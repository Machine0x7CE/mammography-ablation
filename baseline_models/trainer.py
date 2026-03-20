"""
Model training logic for mammography classification.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import roc_auc_score

from .config import TrainingConfig
from .models import get_model, get_hyperparams, count_parameters, create_optimizer, create_scheduler, print_model_config
from .metrics import calculate_metrics
from .benchmark import ModelBenchmark, ModelResult


logger = logging.getLogger(__name__)


class ModelTrainer:
    """Model trainer with progress tracking."""
    
    def __init__(self, config: TrainingConfig, benchmark: ModelBenchmark):
        self.config = config
        self.benchmark = benchmark
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = GradScaler() if config.use_amp else None
        
        logger.info(f"Training on device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def train_model(self, model_name: str, train_loader: DataLoader, 
                    test_loader: DataLoader, class_weights: torch.Tensor) -> ModelResult:
        """
        Train a single model with full tracking.
        
        Args:
            model_name: Name of the model to train
            train_loader: Training data loader
            test_loader: Test data loader
            class_weights: Class weights for loss function
        
        Returns:
            ModelResult with all training results
        """
        
        print(f"\n{'='*70}")
        print(f"TRAINING: {model_name}")
        print(f"{'='*70}")
        
        # Print model-specific configuration
        print_model_config(model_name)
        print(f"Progress: {self.benchmark.get_models_progress()}")
        
        self.benchmark.update_model_status(model_name, "training")
        
        # Initialize model
        model = get_model(model_name, num_classes=2, pretrained=True)
        model = model.to(self.device)
        
        num_params = count_parameters(model)
        logger.info(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        
        # Initialize result
        result = ModelResult(model_name=model_name)
        result.parameters = num_params
        result.status = "training"
        
        # Loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        
        # Create model-specific optimizer (each model uses its recommended optimizer)
        optimizer = create_optimizer(model, model_name)
        
        # Create model-specific scheduler
        model_config = get_hyperparams(model_name)
        steps_per_epoch = len(train_loader)
        scheduler = create_scheduler(optimizer, model_name, self.config.epochs, steps_per_epoch)
        
        # Check scheduler type for proper stepping
        scheduler_type = model_config.get('scheduler', 'ReduceLROnPlateau')
        is_epoch_scheduler = scheduler_type in ['ReduceLROnPlateau', 'StepLR', 'CosineAnnealingWarmRestarts', 'CosineAnnealingLR']
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
            epoch_start_time = time.time()
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            # Training phase (pass scheduler for batch-level stepping like OneCycleLR)
            train_metrics, train_loss = self._train_epoch(
                model, train_loader, criterion, optimizer, epoch,
                scheduler=scheduler if is_batch_scheduler else None
            )
            
            # Validation phase
            val_metrics, val_loss, val_preds, val_labels, val_probs = self._validate_epoch(
                model, test_loader, criterion, epoch
            )
            
            # Update scheduler (epoch-level schedulers)
            if is_epoch_scheduler:
                if scheduler_type == 'ReduceLROnPlateau':
                    # ReduceLROnPlateau needs a metric to monitor
                    scheduler.step(val_metrics.auc_roc)
                else:
                    # StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts don't need args
                    scheduler.step()
            
            # Record history
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_metrics.accuracy)
            val_accs.append(val_metrics.accuracy)
            val_aucs.append(val_metrics.auc_roc)
            
            # Check for improvement
            is_best = val_metrics.auc_roc > best_val_auc
            if is_best:
                best_val_auc = val_metrics.auc_roc
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                
                # Save best model
                checkpoint_path = os.path.join(self.config.models_dir, f'{model_name}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': best_val_auc,
                    'val_acc': val_metrics.accuracy
                }, checkpoint_path)
            else:
                epochs_without_improvement += 1
            
            epoch_time = time.time() - epoch_start_time
            
            # Early stopping
            if epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate final metrics
        training_time = (time.time() - training_start_time) / 60
        
        # Load best model for final evaluation
        checkpoint = torch.load(
            os.path.join(self.config.models_dir, f'{model_name}_best.pth'), 
            weights_only=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final test evaluation
        test_metrics, test_loss, test_preds, test_labels, test_probs = self._validate_epoch(
            model, test_loader, criterion, -1, desc=f"{model_name} Final Test"
        )
        
        # Measure inference time
        inference_time = self._measure_inference_time(model, test_loader)
        
        # Peak GPU memory
        peak_memory = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        
        # Populate result
        result.status = "complete"
        result.timestamp = datetime.now().isoformat()
        result.val_accuracy = val_metrics.accuracy
        result.val_sensitivity = val_metrics.sensitivity
        result.val_specificity = val_metrics.specificity
        result.val_auc = best_val_auc
        result.val_f1 = val_metrics.f1
        result.test_accuracy = test_metrics.accuracy
        result.test_sensitivity = test_metrics.sensitivity
        result.test_specificity = test_metrics.specificity
        result.test_auc = test_metrics.auc_roc
        result.test_f1 = test_metrics.f1
        result.best_epoch = best_epoch
        result.final_lr = current_lr
        result.train_accuracy = train_metrics.accuracy
        result.overfitting_gap = train_metrics.accuracy - val_metrics.accuracy
        result.training_time_minutes = training_time
        result.inference_time_ms = inference_time
        result.samples_per_second = len(train_loader.dataset) / (training_time * 60) * self.config.epochs
        result.peak_gpu_memory_gb = peak_memory
        result.train_losses = train_losses
        result.val_losses = val_losses
        result.train_accs = train_accs
        result.val_accs = val_accs
        result.val_aucs = val_aucs
        result.learning_rates = learning_rates
        result.optimizer = model_config.get('optimizer', 'SGD')
        result.scheduler = scheduler_type
        
        # Store predictions for confusion matrix
        result.test_predictions = test_preds.tolist()
        result.test_labels = test_labels.tolist()
        result.test_probs = test_probs.tolist()
        
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
            os.path.join(self.config.output_root, f'{model_name}_history.csv'), 
            index=False
        )
        
        # Print model completion summary
        print(f"\n{'-'*60}")
        print(f"{model_name} TRAINING COMPLETE")
        print(f"{'-'*60}")
        print(f"Optimizer: {model_config.get('optimizer', 'SGD')} | Scheduler: {scheduler_type}")
        print(f"Best Validation Score at Epoch {best_epoch} (out of {self.config.epochs})")
        print(f"  Val AUC: {best_val_auc:.4f}")
        print(f"  Val Acc: {result.val_accuracy:.1f}%")
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
    
    def _train_epoch(self, model, train_loader, criterion, optimizer, epoch, scheduler=None):
        """Train for one epoch. If scheduler is provided, step it after each batch (for OneCycleLR)."""
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        num_batches = len(train_loader)
        
        pbar = tqdm(
            enumerate(train_loader), 
            total=num_batches,
            desc=f"Ep {epoch+1:02d}/{self.config.epochs} [Train]",
            leave=True,
            ncols=140
        )
        
        for batch_idx, (inputs, labels) in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            if self.config.use_amp and self.scaler:
                with autocast():
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
            
            # Step batch-level scheduler (e.g., OneCycleLR)
            if scheduler is not None:
                scheduler.step()
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
            
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100.0 * correct / total
            current_lr = optimizer.param_groups[0]['lr']
            
            # Calculate running AUC safely
            current_auc = 0.5
            if len(all_labels) > 0:
                try:
                    # Only calculate if we have both classes present
                    if len(set(all_labels)) > 1:
                        current_auc = roc_auc_score(all_labels, all_probs)
                except ValueError:
                    pass

            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1e9
                pbar.set_postfix_str(f"loss={current_loss:.4f}, acc={current_acc:.1f}%, auc={current_auc:.4f}, lr={current_lr:.1e}, gpu={gpu_mem:.1f}GB")
            else:
                pbar.set_postfix_str(f"loss={current_loss:.4f}, acc={current_acc:.1f}%, auc={current_auc:.4f}, lr={current_lr:.1e}")
        
        avg_loss = running_loss / num_batches
        metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
        metrics.loss = avg_loss
        
        return metrics, avg_loss
    
    def _validate_epoch(self, model, val_loader, criterion, epoch, desc=None):
        """Validate for one epoch."""
        model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        if desc is None:
            desc = f"Ep {epoch+1:02d}/{self.config.epochs} [Val]  "
        
        num_batches = len(val_loader)
        
        pbar = tqdm(
            enumerate(val_loader), 
            total=num_batches,
            desc=desc,
            leave=True,
            ncols=140
        )
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                if self.config.use_amp:
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                
                current_loss = running_loss / (batch_idx + 1)
                current_acc = 100.0 * correct / total
                
                # Calculate running AUC safely
                current_auc = 0.5
                if len(all_labels) > 0:
                    try:
                        # Only calculate if we have both classes present
                        if len(set(all_labels)) > 1:
                            current_auc = roc_auc_score(all_labels, all_probs)
                    except ValueError:
                        pass
                
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / 1e9
                    pbar.set_postfix_str(f"loss={current_loss:.4f}, acc={current_acc:.1f}%, auc={current_auc:.4f}, gpu={gpu_mem:.1f}GB")
                else:
                    pbar.set_postfix_str(f"loss={current_loss:.4f}, acc={current_acc:.1f}%, auc={current_auc:.4f}")
        
        avg_loss = running_loss / num_batches
        metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
        metrics.loss = avg_loss
        
        return metrics, avg_loss, np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def _measure_inference_time(self, model, test_loader, num_samples: int = 100) -> float:
        """Measure average inference time in milliseconds."""
        model.eval()
        
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        times = []
        count = 0
        
        with torch.no_grad():
            for inputs, _ in test_loader:
                if count >= num_samples:
                    break
                    
                inputs = inputs.to(self.device)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = model(inputs)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                
                batch_time = (end - start) * 1000 / inputs.size(0)
                times.append(batch_time)
                count += inputs.size(0)
        
        return np.mean(times)