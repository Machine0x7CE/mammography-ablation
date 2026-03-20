# Architectural Ablation and Systematic Hyperparameter Optimization for Mammography

A three-stage comparative study of CNN and Transformer-based architectures for binary mammography classification (benign vs. malignant) on the CBIS-DDSM dataset.

We investigate a question that most published comparisons skip: how much of the performance gap between architectures comes from the model itself, and how much comes from the training recipe? The answer, based on the results here, is that training configuration matters at least as much as architecture choice — and sometimes more.

## Key Findings

- **Stage 1** benchmarks six CNN backbones (VGG16, DenseNet121, EfficientNet-B0, MobileNetV2, ResNet34, ResNet50). DenseNet121 leads with AUC 0.7897 using only 7.48M parameters.
- **Stage 2** compares ResNet50, CBAM-ResNet50, and a Hybrid Vision Transformer under identical training conditions (same optimizer, scheduler, weight decay). Hybrid-ViT leads on AUC (0.7931) but has only 52.1% sensitivity — clinically unacceptable.
- **Stage 3** runs Bayesian hyperparameter optimization (Optuna/TPE, 40 trials per architecture). CBAM-ResNet50, which ranked *last* in Stage 2, becomes the best model: **AUC 0.8176, 87.5% sensitivity**. The ranking inversion is the central finding.
- **Stage 4** (exploratory, in appendix) tests higher resolution (768×768) and multi-view fusion. Performance dropped across the board — a useful negative result showing that SOTA techniques from large-dataset studies don't transfer freely to small academic cohorts.

## Project Structure

```
├── baseline_models/            # Shared Python package (models, training, metrics, visualization)
│   ├── __init__.py
│   ├── config.py               # Training configuration and paths
│   ├── data.py                 # Dataset loaders and transforms
│   ├── models.py               # All model definitions (Stage 1 + Stage 2 architectures)
│   ├── trainer.py              # Training loop with early stopping
│   ├── metrics.py              # Classification metrics (AUC, sensitivity, specificity, F1, MCC)
│   ├── benchmark.py            # Result tracking and reporting
│   ├── visualize.py            # Plotting (ROC curves, confusion matrices, training curves)
│   └── utils.py                # Logging, seeding, GPU utilities
│
├── preprocess_cbis_ddsm_v2.ipynb   # Data preprocessing (DICOM → PNG, ROI crop, CLAHE, split)
├── baseline_comparison.ipynb       # Stage 1: Baseline CNN benchmarking
├── stage2_notebook.ipynb           # Stage 2: Controlled architectural comparison
├── stage3_notebook.ipynb           # Stage 3: Bayesian hyperparameter optimization
├── stage4_preprocessing.ipynb      # Stage 4: High-resolution preprocessing (768×768)
├── stage4_notebook.ipynb           # Stage 4: SOTA training experiment
│
├── inbreast_preprocessing.ipynb    # INbreast external validation preprocessing
├── inbreast_validation.ipynb       # External validation on INbreast dataset
│
├── preprocess_cbis_ddsm.py         # Preprocessing script (CLI version)
├── baseline_comparison.py          # Stage 1 script (CLI version)
├── stage2.py                       # Stage 2 script (CLI version)
│
├── models/                     # Saved model checkpoints
│   ├── *.pth                   # Stage 1 best checkpoints
│   ├── stage2/                 # Stage 2 best checkpoints
│   └── stage3/                 # Stage 3 trial checkpoints (40 per architecture)
│
├── results/                    # Training logs, metrics CSVs, benchmark reports
│   ├── stage2/
│   ├── stage3/
│   └── inbreast_validation/
│
├── visualizations/             # Generated plots and figures
│   ├── stage2/
│   └── stage3/
│
├── stage4_results/             # Stage 4 checkpoints and plots
│
├── FinalReport.tex             # Research paper (LaTeX)
└── requirements.txt            # Python dependencies
```

## Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (training was done in FP32; batch size 32 at 224×224 fits in ~6 GB VRAM)

### Installation

```bash
git clone https://github.com/Machine0x7CE/mammography-ablation.git
cd mammography-ablation
pip install -r requirements.txt
```

### Dataset

This project uses the **CBIS-DDSM** dataset. It is not included in this repository due to size.

1. Download CBIS-DDSM from [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)
2. Place the downloaded DICOM files in `data/manifest-ZkhPvrLo5216730872708713142/`
3. Run the preprocessing notebook:

```bash
jupyter notebook preprocess_cbis_ddsm_v2.ipynb
```

This converts DICOM images to PNG, extracts ROIs with 10% padding, applies CLAHE, resizes to 224×224, and creates patient-wise train/test splits. Preprocessed images are saved to `data/preprocessed/`.

For Stage 4 (optional), run `stage4_preprocessing.ipynb` separately — it produces 768×768 images in `data/stage4_preprocessed/`.

For INbreast external validation (optional), download the [INbreast dataset](http://medicalresearch.inescporto.pt/breastresearch/index.php/Get_INbreast_Database) and run `inbreast_preprocessing.ipynb`.

## Running the Experiments

The stages should be run in order. Each notebook is self-contained once preprocessing is complete.

### Stage 1 — Baseline CNN Benchmarking

```bash
jupyter notebook baseline_comparison.ipynb
```

Trains six architectures (ResNet34, ResNet50, VGG16, DenseNet121, EfficientNet-B0, MobileNetV2) with architecture-specific optimizers and schedulers. Results and checkpoints are saved to `results/` and `models/`.

### Stage 2 — Controlled Architectural Comparison

```bash
jupyter notebook stage2_notebook.ipynb
```

Trains ResNet50, CBAM-ResNet50, and Hybrid-ViT with identical AdamW + CosineAnnealingLR configuration so any performance gap is attributable to architecture alone.

### Stage 3 — Bayesian Hyperparameter Optimization

```bash
jupyter notebook stage3_notebook.ipynb
```

Runs 40 Optuna/TPE trials per architecture with Median Pruner. Searches over learning rate, weight decay, dropout, and scheduler type. Requires `optuna`, `plotly`, and `kaleido` for visualization.

### Stage 4 — Exploratory SOTA Experiment (Optional)

```bash
jupyter notebook stage4_preprocessing.ipynb   # Run first for 768×768 preprocessing
jupyter notebook stage4_notebook.ipynb
```

Tests Focal Loss, MixUp, balanced batch sampling, and multi-view fusion at 768×768 resolution. This experiment produced negative results and is documented in the appendix of the paper.

### External Validation (Optional)

```bash
jupyter notebook inbreast_preprocessing.ipynb
jupyter notebook inbreast_validation.ipynb
```

## License

This project is released for academic and research purposes. The CBIS-DDSM dataset has its own licensing terms through TCIA.
