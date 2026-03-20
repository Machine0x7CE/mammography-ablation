"""
CBIS-DDSM Mammography Dataset Preprocessing Pipeline

Pipeline Steps:
    1. DICOM to PNG Conversion
    2. ROI Cropping with 10% Padding
    3. CLAHE Enhancement (clip limit 2.0, tile grid 8x8)
    4. Image Resizing to 512x512
    5. Patient-wise Train/Test Split (80/20) to prevent data leakage

"""

import os
import sys
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
from datetime import datetime
import traceback

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import pydicom
from pydicom.errors import InvalidDicomError
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)



# CONFIGURATION


@dataclass
class PreprocessingConfig:
    """
    Configuration for the preprocessing pipeline.
    
    Attributes:
        data_root: Root directory containing the CBIS-DDSM dataset
        output_root: Directory where preprocessed images will be saved
        target_size: Output image dimensions (height, width)
        roi_padding: Padding around ROI as fraction (0.10 = 10%)
        clahe_clip_limit: CLAHE clip limit for contrast enhancement
        clahe_tile_grid_size: CLAHE tile grid size
        train_ratio: Fraction of data for training (0.80 = 80%)
        test_ratio: Fraction of data for testing (0.20 = 20%)
        random_seed: Random seed for reproducible splits
        use_roi_cropping: Whether to use ROI masks for cropping
    """
    # Input/Output paths - MODIFY THESE FOR YOUR SYSTEM
    data_root: str = r"D:\Project\data\manifest-ZkhPvrLo5216730872708713142"
    output_root: str = r"D:\Project\data\preprocessed"
    
    # Image processing parameters
    target_size: Tuple[int, int] = (512, 512)
    roi_padding: float = 0.10  # 10% padding around ROI
    
    # CLAHE parameters (optimized for mammography)
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    
    # Split parameters
    train_ratio: float = 0.80
    test_ratio: float = 0.20
    random_seed: int = 42
    
    # Processing options
    use_roi_cropping: bool = True  # Use pre-cropped images if available
    
    def __post_init__(self):
        """Initialize derived paths and create output directories."""
        self.cbis_ddsm_path = os.path.join(self.data_root, "CBIS-DDSM")
        self.output_train = os.path.join(self.output_root, "train")
        self.output_test = os.path.join(self.output_root, "test")
        
        # Create output directory structure for PyTorch ImageFolder compatibility
        for split in [self.output_train, self.output_test]:
            for label in ['benign', 'malignant']:
                os.makedirs(os.path.join(split, label), exist_ok=True)
                
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for JSON serialization."""
        return {
            'data_root': self.data_root,
            'output_root': self.output_root,
            'target_size': list(self.target_size),
            'roi_padding': self.roi_padding,
            'clahe_clip_limit': self.clahe_clip_limit,
            'clahe_tile_grid_size': list(self.clahe_tile_grid_size),
            'train_ratio': self.train_ratio,
            'test_ratio': self.test_ratio,
            'random_seed': self.random_seed,
            'use_roi_cropping': self.use_roi_cropping
        }



# DATA STRUCTURES


@dataclass
class ImageRecord:
    """
    Data class to store information about each mammography image.
    
    Attributes:
        patient_id: Unique patient identifier
        image_type: Type of abnormality ('mass' or 'calc')
        laterality: Breast side ('LEFT' or 'RIGHT')
        view: Mammography view ('CC' or 'MLO')
        abnormality_id: Unique ID for the abnormality within the image
        pathology: Diagnosis ('MALIGNANT', 'BENIGN', 'BENIGN_WITHOUT_CALLBACK')
        full_mammogram_path: Path to full mammogram DICOM file
        cropped_image_path: Path to pre-cropped ROI DICOM file
        roi_mask_path: Path to ROI mask DICOM file
        original_set: Original train/test designation from CBIS-DDSM
        processed_path: Path to saved preprocessed PNG file
        split: Assigned split after patient-wise splitting ('train' or 'test')
    """
    patient_id: str
    image_type: str
    laterality: str
    view: str
    abnormality_id: int
    pathology: str
    full_mammogram_path: str
    cropped_image_path: Optional[str] = None
    roi_mask_path: Optional[str] = None
    original_set: str = ""
    processed_path: str = ""
    split: str = ""
    
    @property
    def label(self) -> str:
        """Get binary label for classification."""
        return 'malignant' if 'MALIGNANT' in self.pathology else 'benign'
    
    @property
    def label_binary(self) -> int:
        """Get binary label as integer (1 = malignant, 0 = benign)."""
        return 1 if 'MALIGNANT' in self.pathology else 0



# IMAGE PROCESSING FUNCTIONS


def read_dicom(dicom_path: str) -> Optional[np.ndarray]:
    """
    Read a DICOM file and return the pixel array as a normalized uint8 image.
    
    Handles:
        - Various DICOM transfer syntaxes
        - MONOCHROME1 (inverted) photometric interpretation
        - Window/level adjustment if available
    
    Args:
        dicom_path: Path to the DICOM file
        
    Returns:
        Normalized uint8 image array, or None if reading fails
    """
    if not dicom_path or not os.path.exists(dicom_path):
        return None
        
    try:
        dcm = pydicom.dcmread(dicom_path, force=True)
        
        if not hasattr(dcm, 'pixel_array'):
            return None
            
        img = dcm.pixel_array.astype(np.float32)
        
        # Handle MONOCHROME1 (inverted grayscale)
        if hasattr(dcm, 'PhotometricInterpretation'):
            if dcm.PhotometricInterpretation == 'MONOCHROME1':
                img = img.max() - img
        
        # Apply window/level if available for better contrast
        if hasattr(dcm, 'WindowCenter') and hasattr(dcm, 'WindowWidth'):
            center = dcm.WindowCenter
            width = dcm.WindowWidth
            
            # Handle multi-value window settings
            if isinstance(center, pydicom.multival.MultiValue):
                center = center[0]
            if isinstance(width, pydicom.multival.MultiValue):
                width = width[0]
            
            img_min = center - width // 2
            img_max = center + width // 2
            img = np.clip(img, img_min, img_max)
        
        # Normalize to 0-255 range
        if img.max() > img.min():
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
            
        return img
        
    except Exception as e:
        logger.debug(f"Error reading DICOM {dicom_path}: {str(e)}")
        return None


def read_roi_mask(mask_path: str) -> Optional[np.ndarray]:
    """
    Read ROI mask from DICOM file.
    
    Args:
        mask_path: Path to the mask DICOM file
        
    Returns:
        Binary mask as uint8 array (255 for ROI, 0 for background)
    """
    if not mask_path or not os.path.exists(mask_path):
        return None
        
    try:
        dcm = pydicom.dcmread(mask_path, force=True)
        if hasattr(dcm, 'pixel_array'):
            mask = dcm.pixel_array
            # Convert to binary mask
            mask = (mask > 0).astype(np.uint8) * 255
            return mask
        return None
    except Exception as e:
        logger.debug(f"Error reading mask {mask_path}: {str(e)}")
        return None


def crop_roi_with_padding(
    image: np.ndarray, 
    mask: Optional[np.ndarray] = None,
    padding: float = 0.10
) -> np.ndarray:
    """
    Crop image to Region of Interest with specified padding.
    
    If a mask is provided, crops to the mask bounding box.
    Otherwise, uses automatic breast region detection.
    
    Args:
        image: Input grayscale image
        mask: Optional ROI mask (same dimensions as image)
        padding: Padding around ROI as fraction (0.10 = 10%)
        
    Returns:
        Cropped image with padding
    """
    if mask is not None and mask.shape == image.shape:
        # Use provided ROI mask
        coords = cv2.findNonZero(mask)
    else:
        # Auto-detect breast region using thresholding
        _, binary = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(binary)
    
    if coords is None:
        return image
        
    # Get bounding box
    x, y, w, h = cv2.boundingRect(coords)
    
    # Calculate padding in pixels
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    
    # Apply padding with boundary checks
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image.shape[1], x + w + pad_x)
    y2 = min(image.shape[0], y + h + pad_y)
    
    return image[y1:y2, x1:x2]


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    CLAHE is particularly effective for mammography images as it:
        - Enhances local contrast
        - Reveals subtle tissue density differences
        - Improves visibility of microcalcifications
    
    Args:
        image: Input grayscale image
        clip_limit: Contrast limiting threshold (2.0 recommended for mammography)
        tile_grid_size: Size of grid for histogram equalization (8x8 default)
        
    Returns:
        CLAHE-enhanced image
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        if image.max() > 255:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Create and apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def resize_image(
    image: np.ndarray, 
    target_size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    """
    Resize image to target size while preserving aspect ratio.
    
    Uses REFLECTION PADDING instead of black borders to:
        - Preserve edge texture for CNNs
        - Avoid sharp black edge artifacts that cause false positives
        - Maintain natural tissue appearance at boundaries
    
    Args:
        image: Input grayscale image
        target_size: Target dimensions (height, width)
        
    Returns:
        Resized image with preserved aspect ratio using reflection padding
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling factor to fit within target size
    scale = min(target_w / w, target_h / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize with high-quality interpolation
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Calculate padding needed on each side
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    
    # Use REFLECTION PADDING instead of black borders
    # This mirrors the image at boundaries, preserving texture
    # and preventing the CNN from learning "black edges = benign/malignant"
    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_REFLECT
    )
    
    return padded



# MAIN PREPROCESSOR CLASS


class CBISDDSMPreprocessor:
    """
    Comprehensive preprocessing pipeline for CBIS-DDSM mammography dataset.
    
    This class orchestrates the entire preprocessing workflow:
        1. Loading and parsing case description CSV files
        2. Patient-wise train/test splitting
        3. Image preprocessing (DICOM → ROI crop → CLAHE → Resize → PNG)
        4. Metadata generation
        5. Sample visualization
    
    All progress bars use leave=True to remain visible after completion.
    """
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize the preprocessor.
        
        Args:
            config: PreprocessingConfig instance with all parameters
        """
        self.config = config
        self.records: List[ImageRecord] = []
        self.patient_records: Dict[str, List[ImageRecord]] = defaultdict(list)
        self.processing_stats = {
            'total_images': 0,
            'successful': 0,
            'failed': 0,
            'malignant': 0,
            'benign': 0,
            'train': 0,
            'test': 0,
            'errors': []
        }
        
    def _resolve_dicom_path(self, relative_path: str) -> str:
        """
        Resolve the DICOM file path from CSV relative path.
        
        The CSV contains paths like:
            Mass-Training_P_00001_LEFT_CC/UID1/UID2/000000.dcm
        
        This method searches the folder structure to find the actual file.
        
        Args:
            relative_path: Relative path from CSV
            
        Returns:
            Absolute path to DICOM file, or empty string if not found
        """
        if pd.isna(relative_path) or str(relative_path) == 'nan':
            return ""
            
        relative_path = str(relative_path).strip().strip('"')
        parts = relative_path.replace('\\', '/').split('/')
        folder_name = parts[0]
        
        folder_path = os.path.join(self.config.cbis_ddsm_path, folder_name)
        
        if os.path.exists(folder_path):
            for root, dirs, files in os.walk(folder_path):
                for f in files:
                    if f.endswith('.dcm'):
                        return os.path.join(root, f)
        
        return ""
    
    def _parse_csv(self, csv_path: str, image_type: str, original_set: str) -> int:
        """
        Parse a single case description CSV file.
        
        Args:
            csv_path: Path to CSV file
            image_type: Type of abnormality ('mass' or 'calc')
            original_set: Original dataset designation ('training' or 'test')
            
        Returns:
            Number of records parsed
        """
        count = 0
        try:
            df = pd.read_csv(csv_path)
            
            for _, row in df.iterrows():
                patient_id = str(row['patient_id'])
                laterality = str(row['left or right breast']).upper()
                view = str(row['image view']).upper()
                abnormality_id = int(row['abnormality id'])
                pathology = str(row['pathology']).upper()
                
                full_mammo_path = self._resolve_dicom_path(str(row['image file path']))
                
                cropped_path = None
                roi_mask_path = None
                
                if 'cropped image file path' in row and pd.notna(row['cropped image file path']):
                    cropped_path = self._resolve_dicom_path(str(row['cropped image file path']))
                    
                if 'ROI mask file path' in row and pd.notna(row['ROI mask file path']):
                    roi_mask_path = self._resolve_dicom_path(str(row['ROI mask file path']))
                
                record = ImageRecord(
                    patient_id=patient_id,
                    image_type=image_type,
                    laterality=laterality,
                    view=view,
                    abnormality_id=abnormality_id,
                    pathology=pathology,
                    full_mammogram_path=full_mammo_path,
                    cropped_image_path=cropped_path,
                    roi_mask_path=roi_mask_path,
                    original_set=original_set
                )
                
                self.records.append(record)
                count += 1
                
        except Exception as e:
            logger.error(f"Error parsing CSV {csv_path}: {str(e)}")
            
        return count
    
    def load_case_descriptions(self) -> None:
        """
        Load and parse all case description CSV files.
        
        Reads mass and calcification case descriptions for both
        training and test sets as provided by CBIS-DDSM.
        """
        logger.info("=" * 60)
        logger.info("STEP 1: LOADING CASE DESCRIPTIONS")
        logger.info("=" * 60)
        
        csv_files = [
            ('mass_case_description_train_set.csv', 'mass', 'training'),
            ('mass_case_description_test_set.csv', 'mass', 'test'),
            ('calc_case_description_train_set.csv', 'calc', 'training'),
            ('calc_case_description_test_set.csv', 'calc', 'test'),
        ]
        
        # Progress bar for loading CSV files
        for csv_name, image_type, original_set in tqdm(csv_files, desc="Loading CSV files", leave=True):
            csv_path = os.path.join(self.config.data_root, csv_name)
            if os.path.exists(csv_path):
                count = self._parse_csv(csv_path, image_type, original_set)
                logger.info(f"  Loaded {count} records from {csv_name}")
            else:
                logger.warning(f"  CSV file not found: {csv_name}")
        
        # Group records by patient for patient-wise splitting
        for record in self.records:
            self.patient_records[record.patient_id].append(record)
            
        logger.info(f"\nTotal: {len(self.records)} image records from {len(self.patient_records)} patients")
        
    def perform_patient_wise_split(self) -> Tuple[List[str], List[str]]:
        """
        Split patients into train/test sets to prevent data leakage.
        
        Uses stratified splitting based on majority pathology per patient
        to maintain class balance across splits.
        
        Returns:
            Tuple of (train_patient_ids, test_patient_ids)
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: PATIENT-WISE TRAIN/TEST SPLIT")
        logger.info("=" * 60)
        
        patients = list(self.patient_records.keys())
        
        # Determine majority label for each patient (for stratification)
        patient_labels = []
        for patient_id in tqdm(patients, desc="Analyzing patient labels", leave=True):
            records = self.patient_records[patient_id]
            malignant_count = sum(1 for r in records if 'MALIGNANT' in r.pathology)
            benign_count = len(records) - malignant_count
            patient_labels.append(1 if malignant_count > benign_count else 0)
        
        # Stratified split to maintain class balance
        train_patients, test_patients = train_test_split(
            patients,
            test_size=self.config.test_ratio,
            random_state=self.config.random_seed,
            stratify=patient_labels
        )
        
        # Assign split to all records
        for patient_id in tqdm(train_patients, desc="Assigning train patients", leave=True):
            for record in self.patient_records[patient_id]:
                record.split = 'train'
                
        for patient_id in tqdm(test_patients, desc="Assigning test patients", leave=True):
            for record in self.patient_records[patient_id]:
                record.split = 'test'
        
        # Log statistics
        train_images = sum(1 for r in self.records if r.split == 'train')
        test_images = sum(1 for r in self.records if r.split == 'test')
        
        logger.info(f"\nSplit complete:")
        logger.info(f"  Train: {len(train_patients)} patients, {train_images} images")
        logger.info(f"  Test: {len(test_patients)} patients, {test_images} images")
        
        # Verify no overlap
        overlap = set(train_patients) & set(test_patients)
        if len(overlap) == 0:
            logger.info("  ✓ Verified: No patient overlap between splits")
        else:
            logger.warning(f"  ✗ WARNING: {len(overlap)} patients appear in both splits!")
        
        return train_patients, test_patients
    
    def process_single_image(self, record: ImageRecord) -> bool:
        """
        Process a single image through the entire pipeline.
        
        Pipeline steps:
            1. Read DICOM image (use pre-cropped if available)
            2. Read ROI mask if available (for full mammograms)
            3. Crop to ROI with padding
            4. Apply CLAHE enhancement
            5. Resize to target size
            6. Save as PNG
        
        Args:
            record: ImageRecord with source paths and metadata
            
        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            # Determine which image to use
            if self.config.use_roi_cropping and record.cropped_image_path:
                image_path = record.cropped_image_path
                mask_path = None  # Cropped images don't need additional cropping
            else:
                image_path = record.full_mammogram_path
                mask_path = record.roi_mask_path
            
            # Step 1: Read DICOM image
            image = read_dicom(image_path)
            if image is None:
                # Try fallback to full mammogram
                if record.full_mammogram_path:
                    image = read_dicom(record.full_mammogram_path)
                if image is None:
                    return False
            
            # Step 2: Read mask if available
            mask = None
            if mask_path and not record.cropped_image_path:
                mask = read_roi_mask(mask_path)
                if mask is not None and mask.shape != image.shape:
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            # Step 3: Crop ROI with padding
            if not record.cropped_image_path:
                image = crop_roi_with_padding(image, mask, self.config.roi_padding)
            
            # Step 4: Apply CLAHE enhancement
            image = apply_clahe(
                image, 
                self.config.clahe_clip_limit, 
                self.config.clahe_tile_grid_size
            )
            
            # Step 5: Resize to target size
            image = resize_image(image, self.config.target_size)
            
            # Determine output path
            label = record.label
            split_dir = self.config.output_train if record.split == 'train' else self.config.output_test
            
            # Generate unique filename
            filename = f"{record.patient_id}_{record.image_type}_{record.laterality}_{record.view}_{record.abnormality_id}.png"
            output_path = os.path.join(split_dir, label, filename)
            
            # Step 6: Save as PNG
            cv2.imwrite(output_path, image)
            record.processed_path = output_path
            
            return True
            
        except Exception as e:
            self.processing_stats['errors'].append({
                'patient_id': record.patient_id,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            return False
    
    def process_all_images(self) -> None:
        """
        Process all images in the dataset with detailed progress tracking.
        
        Uses separate progress bars for:
            - Overall progress
            - Train set processing
            - Test set processing
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: PROCESSING ALL IMAGES")
        logger.info("=" * 60)
        
        self.processing_stats['total_images'] = len(self.records)
        
        # Separate records by split
        train_records = [r for r in self.records if r.split == 'train']
        test_records = [r for r in self.records if r.split == 'test']
        
        logger.info(f"\nProcessing {len(train_records)} training images...")
        
        # Process training images
        for record in tqdm(train_records, desc="Processing TRAIN images", leave=True):
            success = self.process_single_image(record)
            
            if success:
                self.processing_stats['successful'] += 1
                self.processing_stats['train'] += 1
                
                if 'MALIGNANT' in record.pathology:
                    self.processing_stats['malignant'] += 1
                else:
                    self.processing_stats['benign'] += 1
            else:
                self.processing_stats['failed'] += 1
        
        logger.info(f"\nProcessing {len(test_records)} test images...")
        
        # Process test images
        for record in tqdm(test_records, desc="Processing TEST images", leave=True):
            success = self.process_single_image(record)
            
            if success:
                self.processing_stats['successful'] += 1
                self.processing_stats['test'] += 1
                
                if 'MALIGNANT' in record.pathology:
                    self.processing_stats['malignant'] += 1
                else:
                    self.processing_stats['benign'] += 1
            else:
                self.processing_stats['failed'] += 1
        
        # Log results
        logger.info(f"\nProcessing complete:")
        logger.info(f"  ✓ Successful: {self.processing_stats['successful']}/{self.processing_stats['total_images']}")
        logger.info(f"  ✗ Failed: {self.processing_stats['failed']}")
        
    def generate_metadata_csv(self) -> pd.DataFrame:
        """
        Generate metadata CSV files documenting the preprocessed dataset.
        
        Creates:
            - metadata.csv: Complete dataset metadata with class weights
            - train_metadata.csv: Training set only with class weights
            - test_metadata.csv: Test set only
        
        Class weights are calculated from TRAINING SET distribution:
            Weight(Class) = Total_Train_Samples / (2 * Count(Class))
        
        This gives higher weight to the minority class, helping to
        address class imbalance during model training.
            
        Returns:
            DataFrame with complete metadata including class_weight column
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: GENERATING METADATA WITH CLASS WEIGHTS")
        logger.info("=" * 60)
        
        metadata = []
        
        for record in tqdm(self.records, desc="Generating metadata", leave=True):
            if record.processed_path:
                metadata.append({
                    'patient_id': record.patient_id,
                    'image_type': record.image_type,
                    'laterality': record.laterality,
                    'view': record.view,
                    'abnormality_id': record.abnormality_id,
                    'pathology': record.pathology,
                    'label': record.label,
                    'label_binary': record.label_binary,
                    'split': record.split,
                    'original_set': record.original_set,
                    'processed_path': record.processed_path,
                    'original_full_path': record.full_mammogram_path,
                    'original_cropped_path': record.cropped_image_path or '',
                    'original_mask_path': record.roi_mask_path or ''
                })
        
        df = pd.DataFrame(metadata)
        
        # Calculate class weights from TRAINING SET distribution
        # This addresses class imbalance by giving minority class higher weight
        train_df = df[df['split'] == 'train']
        total_train = len(train_df)
        
        train_benign_count = len(train_df[train_df['label'] == 'benign'])
        train_malignant_count = len(train_df[train_df['label'] == 'malignant'])
        
        # Weight formula: Total / (2 * Count)
        # Higher weight for minority class
        if train_benign_count > 0:
            weight_benign = total_train / (2 * train_benign_count)
        else:
            weight_benign = 1.0
            
        if train_malignant_count > 0:
            weight_malignant = total_train / (2 * train_malignant_count)
        else:
            weight_malignant = 1.0
        
        logger.info(f"\nClass distribution in training set:")
        logger.info(f"  Benign:    {train_benign_count} samples, weight = {weight_benign:.4f}")
        logger.info(f"  Malignant: {train_malignant_count} samples, weight = {weight_malignant:.4f}")
        
        # Add class_weight column based on label
        df['class_weight'] = df['label'].apply(
            lambda x: weight_malignant if x == 'malignant' else weight_benign
        )
        
        # Save complete metadata
        metadata_path = os.path.join(self.config.output_root, 'metadata.csv')
        df.to_csv(metadata_path, index=False)
        logger.info(f"\n  Complete metadata: {metadata_path}")
        
        # Save separate train/test metadata
        train_df = df[df['split'] == 'train']
        test_df = df[df['split'] == 'test']
        
        train_path = os.path.join(self.config.output_root, 'train_metadata.csv')
        test_path = os.path.join(self.config.output_root, 'test_metadata.csv')
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"  Train metadata: {train_path} (includes class_weight)")
        logger.info(f"  Test metadata: {test_path}")
        
        return df
    
    def print_summary(self, df: pd.DataFrame) -> None:
        """
        Print comprehensive summary statistics.
        
        Args:
            df: Metadata DataFrame
        """
        print("\n" + "=" * 60)
        print("PREPROCESSING SUMMARY")
        print("=" * 60)
        
        print(f"\nTotal processed images: {len(df)}")
        print(f"Total unique patients: {df['patient_id'].nunique()}")
        
        train_df = df[df['split'] == 'train']
        test_df = df[df['split'] == 'test']
        
        print("\n--- Split Distribution ---")
        print(f"Train: {len(train_df)} images from {train_df['patient_id'].nunique()} patients")
        print(f"Test:  {len(test_df)} images from {test_df['patient_id'].nunique()} patients")
        
        print("\n--- Label Distribution ---")
        print("\nTrain set:")
        print(f"  Benign:    {len(train_df[train_df['label'] == 'benign']):>5}")
        print(f"  Malignant: {len(train_df[train_df['label'] == 'malignant']):>5}")
        
        print("\nTest set:")
        print(f"  Benign:    {len(test_df[test_df['label'] == 'benign']):>5}")
        print(f"  Malignant: {len(test_df[test_df['label'] == 'malignant']):>5}")
        
        print("\n--- Image Type Distribution ---")
        print(df['image_type'].value_counts().to_string())
        
        print("\n--- View Distribution ---")
        print(df['view'].value_counts().to_string())
        
        print("\n" + "=" * 60)
    
    def save_summary_json(self, df: pd.DataFrame) -> None:
        """
        Save detailed summary to JSON file.
        
        Args:
            df: Metadata DataFrame
        """
        train_df = df[df['split'] == 'train']
        test_df = df[df['split'] == 'test']
        
        summary = {
            'total_images': len(df),
            'unique_patients': int(df['patient_id'].nunique()),
            'train_images': len(train_df),
            'train_patients': int(train_df['patient_id'].nunique()),
            'test_images': len(test_df),
            'test_patients': int(test_df['patient_id'].nunique()),
            'train_malignant': int(len(train_df[train_df['label'] == 'malignant'])),
            'train_benign': int(len(train_df[train_df['label'] == 'benign'])),
            'test_malignant': int(len(test_df[test_df['label'] == 'malignant'])),
            'test_benign': int(len(test_df[test_df['label'] == 'benign'])),
            'processing_stats': {
                k: v for k, v in self.processing_stats.items() if k != 'errors'
            },
            'config': self.config.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = os.path.join(self.config.output_root, 'preprocessing_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"  Summary JSON: {summary_path}")
    
    def visualize_samples(self, num_samples: int = 4) -> None:
        """
        Generate visualization of sample preprocessed images.
        
        Creates a grid showing malignant and benign samples.
        
        Args:
            num_samples: Number of samples per category
        """
        try:
            import matplotlib.pyplot as plt
            
            logger.info("\n" + "=" * 60)
            logger.info("STEP 5: GENERATING VISUALIZATIONS")
            logger.info("=" * 60)
            
            # Get samples from each category
            train_malignant = [r for r in self.records 
                             if r.split == 'train' and 'MALIGNANT' in r.pathology and r.processed_path]
            train_benign = [r for r in self.records 
                          if r.split == 'train' and 'MALIGNANT' not in r.pathology and r.processed_path]
            
            fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
            fig.suptitle('Sample Preprocessed Mammography Images (512×512)', 
                        fontsize=14, fontweight='bold')
            
            # Plot malignant samples
            for i, record in enumerate(train_malignant[:num_samples]):
                if os.path.exists(record.processed_path):
                    img = cv2.imread(record.processed_path, cv2.IMREAD_GRAYSCALE)
                    axes[0, i].imshow(img, cmap='gray')
                    axes[0, i].set_title(f'Malignant\n{record.patient_id}', 
                                        fontsize=10, color='#e74c3c')
                    axes[0, i].axis('off')
            
            # Plot benign samples
            for i, record in enumerate(train_benign[:num_samples]):
                if os.path.exists(record.processed_path):
                    img = cv2.imread(record.processed_path, cv2.IMREAD_GRAYSCALE)
                    axes[1, i].imshow(img, cmap='gray')
                    axes[1, i].set_title(f'Benign\n{record.patient_id}', 
                                        fontsize=10, color='#2ecc71')
                    axes[1, i].axis('off')
            
            plt.tight_layout()
            
            viz_path = os.path.join(self.config.output_root, 'sample_visualization.png')
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"  Sample visualization: {viz_path}")
            
        except Exception as e:
            logger.warning(f"Could not generate visualization: {str(e)}")
    
    def run_pipeline(self) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline.
        
        Returns:
            DataFrame with complete metadata
        """
        logger.info("\n" + "=" * 60)
        logger.info("CBIS-DDSM PREPROCESSING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Load case descriptions
        self.load_case_descriptions()
        
        # Step 2: Perform patient-wise split
        self.perform_patient_wise_split()
        
        # Step 3: Process all images
        self.process_all_images()
        
        # Step 4: Generate metadata
        df = self.generate_metadata_csv()
        
        # Step 5: Generate visualizations
        self.visualize_samples()
        
        # Save summary JSON
        self.save_summary_json(df)
        
        # Print final summary
        self.print_summary(df)
        
        logger.info("\n" + "=" * 60)
        logger.info("PREPROCESSING PIPELINE COMPLETE!")
        logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        return df



# PYTORCH DATASET CLASS


class CBISDDSMDataset:
    """
    PyTorch-compatible dataset class for the preprocessed CBIS-DDSM data.
    
    This class can be used directly with PyTorch DataLoader for training.
    
    Example:
        >>> from torch.utils.data import DataLoader
        >>> import torchvision.transforms as transforms
        >>>
        >>> transform = transforms.Compose([
        ...     transforms.ToTensor(),
        ...     transforms.Normalize(mean=[0.5], std=[0.5])
        ... ])
        >>>
        >>> dataset = CBISDDSMDataset(
        ...     metadata_csv='D:/Project/data/preprocessed/train_metadata.csv',
        ...     transform=transform
        ... )
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    
    def __init__(
        self, 
        metadata_csv: str, 
        transform=None,
        target_transform=None
    ):
        """
        Initialize the dataset.
        
        Args:
            metadata_csv: Path to metadata CSV file (train or test)
            transform: Optional transform to apply to images
            target_transform: Optional transform to apply to labels
        """
        self.df = pd.read_csv(metadata_csv)
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image, label)
        """
        row = self.df.iloc[idx]
        
        # Load image
        image = cv2.imread(row['processed_path'], cv2.IMREAD_GRAYSCALE)
        
        # Convert to PIL Image for compatibility with torchvision transforms
        image = Image.fromarray(image)
        
        # Get label
        label = row['label_binary']
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label
    
    def get_labels(self) -> np.ndarray:
        """Return all labels for computing class weights."""
        return self.df['label_binary'].values
    
    def get_class_counts(self) -> Dict[str, int]:
        """Return count of each class."""
        return {
            'benign': len(self.df[self.df['label'] == 'benign']),
            'malignant': len(self.df[self.df['label'] == 'malignant'])
        }



# MAIN ENTRY POINT


def main():
    """Main entry point for the preprocessing script."""
    # Create configuration
    config = PreprocessingConfig()
    
    # Print configuration
    print("\n" + "=" * 60)
    print("CBIS-DDSM PREPROCESSING CONFIGURATION")
    print("=" * 60)
    print(f"Data Root:        {config.data_root}")
    print(f"Output Root:      {config.output_root}")
    print(f"Target Size:      {config.target_size}")
    print(f"ROI Padding:      {config.roi_padding * 100}%")
    print(f"CLAHE Clip Limit: {config.clahe_clip_limit}")
    print(f"CLAHE Tile Grid:  {config.clahe_tile_grid_size}")
    print(f"Train/Test Split: {config.train_ratio * 100}%/{config.test_ratio * 100}%")
    print(f"Random Seed:      {config.random_seed}")
    print(f"Use ROI Cropping: {config.use_roi_cropping}")
    print("=" * 60 + "\n")
    
    # Create and run preprocessor
    preprocessor = CBISDDSMPreprocessor(config)
    df = preprocessor.run_pipeline()
    
    # Print output structure
    print("\n" + "=" * 60)
    print("OUTPUT STRUCTURE (PyTorch ImageFolder Compatible)")
    print("=" * 60)
    print(f"""
{config.output_root}/
├── train/
│   ├── benign/     ({len(df[(df['split']=='train') & (df['label']=='benign')])} images)
│   └── malignant/  ({len(df[(df['split']=='train') & (df['label']=='malignant')])} images)
├── test/
│   ├── benign/     ({len(df[(df['split']=='test') & (df['label']=='benign')])} images)
│   └── malignant/  ({len(df[(df['split']=='test') & (df['label']=='malignant')])} images)
├── metadata.csv
├── train_metadata.csv
├── test_metadata.csv
├── preprocessing_summary.json
└── sample_visualization.png
    """)


if __name__ == "__main__":
    main()

