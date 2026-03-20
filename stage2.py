"""
Stage 2: Advanced Model Training and Benchmarking

This script trains and compares advanced architectures:
- ResNet50Stage2: Clean ResNet50 backbone (same hyperparams as Stage 1 for fair comparison)
- CBAMResNet50: ResNet50 with Channel and Spatial Attention
- HybridViT: CNN backbone + Transformer encoder

All three models use IDENTICAL hyperparameters matching Stage 1 ResNet50
for fair architectural comparison before hyperparameter tuning.

Features:
- Standardized hyperparameters across all models for fair comparison
- Support for hyperparameter experiments (multiple configs per model)
- Same visualizations as Stage 1 baseline comparison
- Results tracking and comparison

Usage:
    python stage2.py                    # Run with default configs
    python stage2.py --epochs 10        # Override epochs
    python stage2.py --experiment       # Run hyperparameter experiment
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional

import torch
import numpy as np
import pandas as pd

# Add baseline_models to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_models.config import TrainingConfig
from baseline_models.data import get_data_loaders, print_data_info
from baseline_models.models import (
    get_model, get_hyperparams, count_parameters, 
    create_optimizer_from_config, create_scheduler_from_config,
    print_model_config, get_experiment_name, MODEL_HYPERPARAMS
)
from baseline_models.benchmark import ModelBenchmark, ModelResult
from baseline_models.visualize import Visualizer



# Stage 2 Configuration


@dataclass
class Stage2Config(TrainingConfig):
    """Extended configuration for Stage 2 experiments."""
    
    # Override output directories for Stage 2
    output_root: str = r"D:\Project\results\stage2"
    models_dir: str = r"D:\Project\models\stage2"
    viz_dir: str = r"D:\Project\visualizations\stage2"
    
    # Stage 2 models (all use identical hyperparams for fair architectural comparison)
    models_to_train: List[str] = field(default_factory=lambda: [
        'ResNet50Stage2',
        'CBAMResNet50', 
        'HybridViT'
    ])
    
    # Default epochs (can be overridden)
    epochs: int = 8
    
    def __post_init__(self):
        """Create output directories."""
        for path in [self.output_root, self.models_dir, self.viz_dir]:
            os.makedirs(path, exist_ok=True)



# Hyperparameter Experiment Configurations


# Define experiment configurations for hyperparameter tuning
# Each model can have multiple configurations to compare
#
# NOTE: For fair architectural comparison, all models use the SAME default
# hyperparameters matching Stage 1 ResNet50. Experiments test variations.

# Standard hyperparameters (matching Stage 1 ResNet50)
STANDARD_CONFIG = {
    'optimizer': 'AdamW',
    'learning_rate': 1e-4,
    'betas': (0.9, 0.999),
    'weight_decay': 1e-2,
    'dropout': 0.0,
    'scheduler': 'CosineAnnealingLR',
    'T_max': 60,
    'eta_min': 1e-6,
}

EXPERIMENT_CONFIGS = {
    'ResNet50Stage2': [
        # Default configuration (matches Stage 1 ResNet50)
        {
            'name': 'default',
            **STANDARD_CONFIG,
        },
        # Higher learning rate experiment
        {
            'name': 'high_lr',
            'optimizer': 'AdamW',
            'learning_rate': 3e-4,
            'betas': (0.9, 0.999),
            'weight_decay': 1e-2,
            'dropout': 0.0,
            'scheduler': 'CosineAnnealingLR',
            'T_max': 60,
            'eta_min': 1e-6,
        },
        # SGD with momentum experiment
        {
            'name': 'sgd',
            'optimizer': 'SGD',
            'learning_rate': 1e-3,
            'momentum': 0.9,
            'nesterov': True,
            'weight_decay': 1e-4,
            'dropout': 0.0,
            'scheduler': 'StepLR',
            'step_size': 15,
            'gamma': 0.1,
        },
    ],
    
    'CBAMResNet50': [
        # Default configuration (matches Stage 1 ResNet50)
        {
            'name': 'default',
            **STANDARD_CONFIG,
        },
        # With dropout experiment
        {
            'name': 'with_dropout',
            'optimizer': 'AdamW',
            'learning_rate': 1e-4,
            'betas': (0.9, 0.999),
            'weight_decay': 1e-2,
            'dropout': 0.3,
            'scheduler': 'CosineAnnealingLR',
            'T_max': 60,
            'eta_min': 1e-6,
        },
        # Lower learning rate experiment
        {
            'name': 'low_lr',
            'optimizer': 'AdamW',
            'learning_rate': 5e-5,
            'betas': (0.9, 0.999),
            'weight_decay': 1e-2,
            'dropout': 0.0,
            'scheduler': 'CosineAnnealingWarmRestarts',
            'T_0': 10,
            'T_mult': 2,
        },
    ],
    
    'HybridViT': [
        # Default configuration (matches Stage 1 ResNet50)
        {
            'name': 'default',
            **STANDARD_CONFIG,
        },
        # Higher weight decay experiment (ViTs often need more regularization)
        {
            'name': 'high_wd',
            'optimizer': 'AdamW',
            'learning_rate': 1e-4,
            'betas': (0.9, 0.999),
            'weight_decay': 0.05,
            'dropout': 0.0,
            'scheduler': 'CosineAnnealingLR',
            'T_max': 60,
            'eta_min': 1e-6,
        },
        # OneCycleLR experiment (warmup)
        {
            'name': 'warmup',
            'optimizer': 'AdamW',
            'learning_rate': 1e-4,
            'betas': (0.9, 0.999),
            'weight_decay': 1e-2,
            'dropout': 0.0,
            'scheduler': 'OneCycleLR',
            'max_lr': 1e-4,
            'pct_start': 0.1,
        },
    ],
}



# Stage 2 Trainer (supports custom hyperparameters)


class Stage2Trainer:
    """
    Trainer for Stage 2 models with support for custom hyperparameter configs.
    """
    
    def __init__(self, config: Stage2Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Mixed precision (using new API to avoid deprecation warnings)
        if config.use_amp and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        print(f"\nStage 2 Training Configuration")
        print(f"{'='*50}")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Epochs: {config.epochs}")
        print(f"Batch Size: {config.batch_size}")
        print(f"AMP Enabled: {config.use_amp}")
        print(f"{'='*50}\n")
    
    def train_model(self, model_name: str, train_loader, test_loader, 
                    class_weights: torch.Tensor,
                    custom_config: Optional[Dict[str, Any]] = None) -> ModelResult:
        """
        Train a single model with optional custom hyperparameters.
        
        Args:
            model_name: Name of the model architecture
            train_loader: Training data loader
            test_loader: Test data loader  
            class_weights: Class weights for loss function
            custom_config: Optional custom hyperparameter config
        
        Returns:
            ModelResult with training results
        """
        from tqdm import tqdm
        from sklearn.metrics import roc_auc_score
        import time
        
        from baseline_models.metrics import calculate_metrics
        
        # Get hyperparameter config
        if custom_config is not None:
            hp_config = custom_config
            config_name = custom_config.get('name', 'custom')
            experiment_name = f"{model_name}_{config_name}"
        else:
            hp_config = get_hyperparams(model_name)
            config_name = 'default'
            experiment_name = model_name
        
        print(f"\n{'='*70}")
        print(f"TRAINING: {experiment_name}")
        print(f"{'='*70}")
        print_model_config(model_name, hp_config)
        
        # Initialize model
        dropout = hp_config.get('dropout', 0.3)
        model = get_model(model_name, num_classes=2, pretrained=True, dropout=dropout)
        model = model.to(self.device)
        
        num_params = count_parameters(model)
        print(f"Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        
        # Loss function with class weights
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        
        # Create optimizer and scheduler from config
        optimizer = create_optimizer_from_config(model, hp_config)
        
        steps_per_epoch = len(train_loader)
        scheduler = create_scheduler_from_config(
            optimizer, hp_config, 
            num_epochs=self.config.epochs, 
            steps_per_epoch=steps_per_epoch
        )
        
        # Determine scheduler type for proper stepping
        scheduler_type = hp_config.get('scheduler', 'CosineAnnealingLR')
        is_epoch_scheduler = scheduler_type in ['ReduceLROnPlateau', 'StepLR', 
                                                  'CosineAnnealingWarmRestarts', 
                                                  'CosineAnnealingLR']
        is_batch_scheduler = scheduler_type == 'OneCycleLR'
        
        # Training tracking
        best_val_auc = 0.0
        best_epoch = 0
        epochs_without_improvement = 0
        training_start_time = time.time()
        
        # History
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        val_aucs = []
        learning_rates = []
        
        # Training loop
        for epoch in range(self.config.epochs):
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            all_train_probs = []
            all_train_labels = []
            
            pbar = tqdm(
                train_loader,
                desc=f"Ep {epoch+1:02d}/{self.config.epochs} [Train]",
                leave=True,
                ncols=140
            )
            
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                if self.config.use_amp and self.scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    self.scaler.scale(loss).backward()
                    
                    if self.config.gradient_clip > 0:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                    
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    
                    if self.config.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                    
                    optimizer.step()
                
                # Step batch-level scheduler
                if is_batch_scheduler:
                    scheduler.step()
                
                running_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_train_probs.extend(probs[:, 1].detach().cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())
                
                current_loss = running_loss / (pbar.n + 1)
                current_acc = 100.0 * correct / total
                
                # Calculate AUC safely
                train_auc = 0.5
                if len(set(all_train_labels)) > 1:
                    try:
                        train_auc = roc_auc_score(all_train_labels, all_train_probs)
                    except:
                        pass
                
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / 1e9
                    pbar.set_postfix_str(f"loss={current_loss:.4f}, acc={current_acc:.1f}%, auc={train_auc:.4f}, lr={current_lr:.1e}, gpu={gpu_mem:.1f}GB")
                else:
                    pbar.set_postfix_str(f"loss={current_loss:.4f}, acc={current_acc:.1f}%, auc={train_auc:.4f}, lr={current_lr:.1e}")
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100.0 * correct / total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_val_preds = []
            all_val_labels = []
            all_val_probs = []
            
            pbar = tqdm(
                test_loader,
                desc=f"Ep {epoch+1:02d}/{self.config.epochs} [Val]  ",
                leave=True,
                ncols=140
            )
            
            with torch.no_grad():
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    if self.config.use_amp:
                        with torch.amp.autocast('cuda'):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    all_val_preds.extend(predicted.cpu().numpy())
                    all_val_labels.extend(labels.cpu().numpy())
                    all_val_probs.extend(probs[:, 1].cpu().numpy())
                    
                    current_loss = val_loss / (pbar.n + 1)
                    current_acc = 100.0 * val_correct / val_total
                    
                    # Calculate AUC safely
                    val_auc = 0.5
                    if len(set(all_val_labels)) > 1:
                        try:
                            val_auc = roc_auc_score(all_val_labels, all_val_probs)
                        except:
                            pass
                    
                    if torch.cuda.is_available():
                        gpu_mem = torch.cuda.memory_allocated() / 1e9
                        pbar.set_postfix_str(f"loss={current_loss:.4f}, acc={current_acc:.1f}%, auc={val_auc:.4f}, gpu={gpu_mem:.1f}GB")
                    else:
                        pbar.set_postfix_str(f"loss={current_loss:.4f}, acc={current_acc:.1f}%, auc={val_auc:.4f}")
            
            val_loss = val_loss / len(test_loader)
            val_acc = 100.0 * val_correct / val_total
            
            # Calculate final validation AUC
            val_auc = 0.5
            if len(set(all_val_labels)) > 1:
                try:
                    val_auc = roc_auc_score(all_val_labels, all_val_probs)
                except:
                    pass
            
            # Update scheduler (epoch-level)
            if is_epoch_scheduler:
                if scheduler_type == 'ReduceLROnPlateau':
                    scheduler.step(val_auc)
                else:
                    scheduler.step()
            
            # Record history
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            val_aucs.append(val_auc)
            
            # Check for improvement
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                
                # Save best model
                checkpoint_path = os.path.join(self.config.models_dir, f'{experiment_name}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': best_val_auc,
                    'val_acc': val_acc,
                    'config': hp_config
                }, checkpoint_path)
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Load best model for final evaluation
        training_time = (time.time() - training_start_time) / 60
        
        checkpoint = torch.load(
            os.path.join(self.config.models_dir, f'{experiment_name}_best.pth'),
            weights_only=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final test evaluation
        model.eval()
        test_preds = []
        test_labels = []
        test_probs = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                test_preds.extend(predicted.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
                test_probs.extend(probs[:, 1].cpu().numpy())
        
        # Calculate final metrics
        test_metrics = calculate_metrics(
            np.array(test_labels), 
            np.array(test_preds), 
            np.array(test_probs)
        )
        
        # Peak GPU memory
        peak_memory = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        
        # Create result
        result = ModelResult(model_name=experiment_name)
        result.status = "complete"
        result.timestamp = datetime.now().isoformat()
        result.parameters = num_params
        result.val_accuracy = val_accs[-1] if val_accs else 0.0
        result.val_sensitivity = test_metrics.sensitivity
        result.val_specificity = test_metrics.specificity
        result.val_auc = best_val_auc
        result.val_f1 = test_metrics.f1
        result.test_accuracy = test_metrics.accuracy
        result.test_sensitivity = test_metrics.sensitivity
        result.test_specificity = test_metrics.specificity
        result.test_auc = test_metrics.auc_roc
        result.test_f1 = test_metrics.f1
        result.best_epoch = best_epoch
        result.final_lr = current_lr
        result.train_accuracy = train_accs[-1] if train_accs else 0.0
        result.overfitting_gap = (train_accs[-1] - val_accs[-1]) if train_accs and val_accs else 0.0
        result.training_time_minutes = training_time
        result.peak_gpu_memory_gb = peak_memory
        result.train_losses = train_losses
        result.val_losses = val_losses
        result.train_accs = train_accs
        result.val_accs = val_accs
        result.val_aucs = val_aucs
        result.learning_rates = learning_rates
        result.optimizer = hp_config.get('optimizer', 'AdamW')
        result.scheduler = scheduler_type
        result.test_predictions = test_preds
        result.test_labels = test_labels
        result.test_probs = test_probs
        
        # Save training history
        history_df = pd.DataFrame({
            'epoch': range(1, len(train_losses) + 1),
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_acc': train_accs,
            'val_acc': val_accs,
            'val_auc': val_aucs,
            'lr': learning_rates
        })
        history_df.to_csv(
            os.path.join(self.config.output_root, f'{experiment_name}_history.csv'),
            index=False
        )
        
        # Print summary
        print(f"\n{'-'*60}")
        print(f"{experiment_name} TRAINING COMPLETE")
        print(f"{'-'*60}")
        print(f"Optimizer: {hp_config.get('optimizer', 'AdamW')} | Scheduler: {scheduler_type}")
        print(f"Best Validation Score at Epoch {best_epoch} (out of {self.config.epochs})")
        print(f"  Val AUC: {best_val_auc:.4f}")
        print(f"  Val Acc: {val_accs[-1] if val_accs else 0:.1f}%")
        print(f"{'-'*60}")
        print(f"Final Test Results:")
        print(f"  Test AUC: {test_metrics.auc_roc:.4f}")
        print(f"  Test Acc: {test_metrics.accuracy:.1f}%")
        print(f"  Sensitivity: {test_metrics.sensitivity:.1f}%")
        print(f"  Specificity: {test_metrics.specificity:.1f}%")
        print(f"  F1 Score: {test_metrics.f1:.1f}%")
        print(f"{'-'*60}")
        print(f"Training Time: {training_time:.1f} minutes")
        print(f"{'-'*60}\n")
        
        return result



# Stage 2 Benchmark Runner


class Stage2Benchmark:
    """Manages Stage 2 model benchmarking and comparison."""
    
    def __init__(self, config: Stage2Config):
        self.config = config
        self.results: Dict[str, ModelResult] = {}
        self.benchmark_start_time = datetime.now()
    
    def add_result(self, name: str, result: ModelResult):
        """Add a model result."""
        self.results[name] = result
        self._save_log()
    
    def _save_log(self):
        """Save benchmark log to JSON."""
        log_path = os.path.join(self.config.output_root, 'stage2_benchmark_log.json')
        
        log_data = {
            'benchmark_started': self.benchmark_start_time.isoformat(),
            'last_updated': datetime.now().isoformat(),
            'models_completed': len(self.results),
            'results': {name: asdict(r) for name, r in self.results.items()}
        }
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, default=str)
    
    def get_sorted_results(self) -> List[str]:
        """Get model names sorted by validation AUC."""
        return sorted(self.results.keys(), key=lambda n: self.results[n].val_auc, reverse=True)
    
    def print_summary(self):
        """Print summary of all results."""
        if not self.results:
            print("No results to display.")
            return
        
        sorted_names = self.get_sorted_results()
        
        print("\n")
        print("=" * 100)
        print("STAGE 2 MODEL COMPARISON - BENCHMARK RESULTS")
        print("=" * 100)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Models Evaluated: {len(self.results)}")
        print("=" * 100)
        
        # Create summary table
        print("\nPERFORMANCE RANKING (sorted by Validation AUC)")
        print("-" * 140)
        print(f"{'Rank':<5} {'Model':<30} {'Val AUC':<10} {'Val Acc':<10} {'Test AUC':<10} {'Test Acc':<10} {'Sens':<8} {'Spec':<8} {'F1':<8} {'Time':<8}")
        print("-" * 140)
        
        for rank, name in enumerate(sorted_names, 1):
            r = self.results[name]
            print(f"{rank:<5} {name:<30} {r.val_auc:<10.4f} {r.val_accuracy:<10.1f} {r.test_auc:<10.4f} {r.test_accuracy:<10.1f} {r.test_sensitivity:<8.1f} {r.test_specificity:<8.1f} {r.test_f1:<8.1f} {r.training_time_minutes:<8.1f}")
        
        print("-" * 140)
        
        # Best model
        best_name = sorted_names[0]
        best = self.results[best_name]
        
        print("\nBEST MODEL RECOMMENDATION")
        print("-" * 50)
        print(f"Model: {best_name}")
        print(f"Validation AUC: {best.val_auc:.4f}")
        print(f"Test AUC: {best.test_auc:.4f}")
        print(f"Sensitivity: {best.test_sensitivity:.1f}%")
        print(f"Specificity: {best.test_specificity:.1f}%")
        print("-" * 50)
        
        # Save CSV
        summary_data = []
        for rank, name in enumerate(sorted_names, 1):
            r = self.results[name]
            summary_data.append({
                'Rank': rank,
                'Model': name,
                'Val AUC': r.val_auc,
                'Val Acc': r.val_accuracy,
                'Test AUC': r.test_auc,
                'Test Acc': r.test_accuracy,
                'Sensitivity': r.test_sensitivity,
                'Specificity': r.test_specificity,
                'F1': r.test_f1,
                'Best Epoch': r.best_epoch,
                'Time (min)': r.training_time_minutes
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(self.config.output_root, 'stage2_summary.csv'), index=False)


def run_stage2_default(config: Stage2Config):
    """Run Stage 2 with default hyperparameters."""
    print("\n" + "=" * 70)
    print("STAGE 2: ADVANCED MODEL TRAINING")
    print("Using default hyperparameters for each model")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    train_loader, test_loader, class_weights = get_data_loaders(config)
    print_data_info(train_loader, test_loader, class_weights)
    
    # Initialize
    trainer = Stage2Trainer(config)
    benchmark = Stage2Benchmark(config)
    
    # Train each model with default config
    for model_name in config.models_to_train:
        result = trainer.train_model(model_name, train_loader, test_loader, class_weights)
        benchmark.add_result(model_name, result)
    
    # Print summary
    benchmark.print_summary()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    visualizer = Visualizer(config)
    sorted_models = benchmark.get_sorted_results()
    visualizer.generate_all_visualizations(benchmark.results, sorted_models)
    
    print(f"\nStage 2 complete! Results saved to: {config.output_root}")
    
    return benchmark


def run_stage2_experiment(config: Stage2Config):
    """Run Stage 2 with hyperparameter experiments."""
    print("\n" + "=" * 70)
    print("STAGE 2: HYPERPARAMETER EXPERIMENT MODE")
    print("Testing multiple configurations for each model")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    train_loader, test_loader, class_weights = get_data_loaders(config)
    print_data_info(train_loader, test_loader, class_weights)
    
    # Initialize
    trainer = Stage2Trainer(config)
    benchmark = Stage2Benchmark(config)
    
    # Train each model with each configuration
    for model_name in config.models_to_train:
        configs = EXPERIMENT_CONFIGS.get(model_name, [])
        
        if not configs:
            # Use default if no experiment configs defined
            result = trainer.train_model(model_name, train_loader, test_loader, class_weights)
            benchmark.add_result(model_name, result)
        else:
            for hp_config in configs:
                experiment_name = f"{model_name}_{hp_config.get('name', 'custom')}"
                result = trainer.train_model(
                    model_name, train_loader, test_loader, class_weights,
                    custom_config=hp_config
                )
                benchmark.add_result(experiment_name, result)
    
    # Print summary
    benchmark.print_summary()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    visualizer = Visualizer(config)
    sorted_models = benchmark.get_sorted_results()
    visualizer.generate_all_visualizations(benchmark.results, sorted_models)
    
    print(f"\nExperiment complete! Results saved to: {config.output_root}")
    
    return benchmark


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Stage 2: Advanced Model Training')
    parser.add_argument('--epochs', type=int, default=8, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--experiment', action='store_true', 
                        help='Run hyperparameter experiment mode')
    parser.add_argument('--models', nargs='+', 
                        default=['ResNet50Stage2', 'CBAMResNet50', 'HybridViT'],
                        help='Models to train')
    
    args = parser.parse_args()
    
    # Create config
    config = Stage2Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        models_to_train=args.models
    )
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run
    if args.experiment:
        run_stage2_experiment(config)
    else:
        run_stage2_default(config)


if __name__ == '__main__':
    main()

