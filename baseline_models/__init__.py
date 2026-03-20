"""
Baseline Models Package for Mammography Classification

This package provides a modular framework for training and comparing
CNN architectures on breast cancer detection tasks.

Modules:
    config      - Configuration settings
    models      - Model architectures and hyperparameters
    data        - Dataset and data loading utilities
    metrics     - Performance metrics calculation
    trainer     - Training logic
    benchmark   - Benchmarking and reporting
    visualize   - Visualization functions
"""

from .config import TrainingConfig
from .models import get_model, get_hyperparams, count_parameters, MODEL_HYPERPARAMS
from .data import MammographyDataset, get_data_loaders
from .metrics import EpochMetrics, calculate_metrics
from .trainer import ModelTrainer
from .benchmark import ModelBenchmark, ModelResult
from .visualize import Visualizer

__version__ = '1.0.0'
__author__ = 'Breast Cancer Detection Research Project'
