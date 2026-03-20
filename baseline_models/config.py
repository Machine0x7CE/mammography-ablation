"""
Configuration settings for baseline model comparison.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingConfig:
    """Configuration for baseline model comparison."""
    
    # Data paths
    data_root: str = r"D:\Project\data\preprocessed"
    output_root: str = r"D:\Project\results"
    models_dir: str = r"D:\Project\models"
    viz_dir: str = r"D:\Project\visualizations"
    
    # Training parameters
    epochs: int = 65
    batch_size: int = 64
    num_workers: int = 0  # Set to 0 for Windows compatibility
    
    # Default optimizer parameters
    learning_rate: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 1e-4
    
    # Scheduler parameters
    scheduler_factor: float = 0.1
    scheduler_patience: int = 5
    
    # Other settings
    random_seed: int = 42
    use_amp: bool = True
    gradient_clip: float = 1.0
    early_stopping_patience: int = 50
    
    # Models to compare
    models_to_train: List[str] = field(default_factory=lambda: [
        'ResNet34', 'ResNet50', 'VGG16', 'DenseNet121', 
        'EfficientNet-B0', 'MobileNetV2'
    ])
    
    def __post_init__(self):
        """Create output directories."""
        for dir_path in [self.output_root, self.models_dir, self.viz_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def get_train_dir(self) -> str:
        return os.path.join(self.data_root, 'train')
    
    def get_test_dir(self) -> str:
        return os.path.join(self.data_root, 'test')
    
    def print_config(self):
        """Print configuration summary."""
        print(f"Configuration:")
        print(f"  Data Root: {self.data_root}")
        print(f"  Output: {self.output_root}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Models: {', '.join(self.models_to_train)}")
