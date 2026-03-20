"""
Baseline Model Comparison for Mammography Classification

Main entry point for comparing CNN architectures on breast cancer detection.
This script orchestrates the training and evaluation of multiple models.

Models compared:
    - ResNet34
    - ResNet50
    - VGG16
    - DenseNet121
    - EfficientNet-B0
    - MobileNetV2

Usage:
    python baseline_comparison.py
"""

from datetime import datetime

from baseline_models.config import TrainingConfig
from baseline_models.data import get_data_loaders, print_data_info
from baseline_models.benchmark import ModelBenchmark
from baseline_models.trainer import ModelTrainer
from baseline_models.visualize import Visualizer
from baseline_models.utils import setup_logging, set_seed, print_system_info, clear_gpu_memory


def main():
    """Main execution function for baseline model comparison."""
    
    # Setup
    logger = setup_logging()
    
    print("\nBaseline Model Comparison for Mammography Classification")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_system_info()
    
    # Initialize configuration
    config = TrainingConfig()
    config.print_config()
    
    # Set random seed for reproducibility
    set_seed(config.random_seed)
    
    # Load data
    print("\nLoading data...")
    train_loader, test_loader, class_weights = get_data_loaders(config)
    print_data_info(train_loader, test_loader, class_weights)
    
    # Initialize benchmark system
    benchmark = ModelBenchmark(config)
    
    # Initialize trainer
    trainer = ModelTrainer(config, benchmark)
    
    # Train all models
    for model_name in config.models_to_train:
        try:
            result = trainer.train_model(model_name, train_loader, test_loader, class_weights)
            benchmark.add_result(model_name, result)
            
        except Exception as e:
            print(f"\nERROR: Failed to train {model_name}: {str(e)}")
            logger.exception(f"Error training {model_name}")
            benchmark.update_model_status(model_name, "failed")
            continue
        
        clear_gpu_memory()
    
    # Generate final report
    benchmark.print_final_report()
    benchmark.generate_markdown_report()
    
    # Generate all visualizations
    completed = benchmark.get_completed_results()
    sorted_models = benchmark.get_sorted_models()
    
    if completed:
        visualizer = Visualizer(config)
        visualizer.generate_all_visualizations(completed, sorted_models)
    
    # Final summary
    print("\n")
    print("=" * 100)
    print("BASELINE COMPARISON COMPLETE")
    print("=" * 100)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {config.output_root}")
    print(f"Models saved to: {config.models_dir}")
    print(f"Visualizations saved to: {config.viz_dir}")
    
    return benchmark.create_summary_dataframe()


if __name__ == "__main__":
    summary = main()
