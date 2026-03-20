"""
Visualization functions for model comparison and analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from sklearn.metrics import confusion_matrix, roc_curve

from .config import TrainingConfig
from .benchmark import ModelResult


class Visualizer:
    """Handles all visualization tasks for model comparison."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def generate_all_visualizations(self, completed: Dict[str, ModelResult], sorted_models: List[str]):
        """Generate all comparison visualizations."""
        print("\nGenerating visualizations...")
        
        self.plot_confusion_matrices(completed, sorted_models)
        self.plot_validation_comparison(completed, sorted_models)
        self.plot_cancer_metrics(completed, sorted_models)
        self.plot_training_curves(completed, sorted_models)
        self.plot_roc_curves(completed, sorted_models)
        self.plot_efficiency_analysis(completed, sorted_models)
        self.plot_summary_table(completed, sorted_models)
        self.plot_comparison_overview(completed, sorted_models)
        self.plot_individual_model_summaries(completed, sorted_models)
        
        print(f"Visualizations saved to: {self.config.viz_dir}")
    
    def plot_confusion_matrices(self, completed: Dict[str, ModelResult], sorted_models: List[str]):
        """Plot detailed confusion matrix for each model."""
        n_models = len(sorted_models)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        fig.suptitle('Confusion Matrices - All Models\n(TP=True Positive, TN=True Negative, FP=False Positive, FN=False Negative)', 
                     fontsize=14, fontweight='bold')
        
        axes_flat = axes.flatten() if n_models > 1 else [axes]
        class_names = ['Benign (B)', 'Malignant (M)']
        
        for idx, model_name in enumerate(sorted_models):
            r = completed[model_name]
            ax = axes_flat[idx]
            
            if r.test_predictions and r.test_labels:
                cm = confusion_matrix(r.test_labels, r.test_predictions)
                cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
                
                tn, fp, fn, tp = cm.ravel()
                
                annot = np.array([
                    [f'TN\n{tn}\n({cm_percent[0,0]:.1f}%)', f'FP\n{fp}\n({cm_percent[0,1]:.1f}%)'],
                    [f'FN\n{fn}\n({cm_percent[1,0]:.1f}%)', f'TP\n{tp}\n({cm_percent[1,1]:.1f}%)']
                ])
                
                sns.heatmap(cm_percent, annot=annot, fmt='', cmap='Blues',
                           xticklabels=class_names, yticklabels=class_names,
                           ax=ax, cbar=False, annot_kws={'fontsize': 10})
                
                sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
                
                ax.set_title(f'{model_name}\nAcc: {r.test_accuracy:.1f}% | Sens: {sensitivity:.1f}% | Spec: {specificity:.1f}%',
                            fontsize=10)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(model_name)
        
        for idx in range(len(sorted_models), len(axes_flat)):
            axes_flat[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.viz_dir, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        self.plot_classification_table(completed, sorted_models)
    
    def plot_classification_table(self, completed: Dict[str, ModelResult], sorted_models: List[str]):
        """Create detailed classification metrics table."""
        fig, ax = plt.subplots(figsize=(14, len(sorted_models) * 0.8 + 2))
        ax.axis('off')
        
        headers = ['Model', 'TP', 'TN', 'FP', 'FN', 'TP%', 'TN%', 'FP%', 'FN%', 'Sens%', 'Spec%', 'Acc%']
        table_data = []
        
        for model_name in sorted_models:
            r = completed[model_name]
            if r.test_predictions and r.test_labels:
                cm = confusion_matrix(r.test_labels, r.test_predictions)
                tn, fp, fn, tp = cm.ravel()
                total = tn + fp + fn + tp
                
                table_data.append([
                    model_name, tp, tn, fp, fn,
                    f'{tp/total*100:.1f}', f'{tn/total*100:.1f}', 
                    f'{fp/total*100:.1f}', f'{fn/total*100:.1f}',
                    f'{tp/(tp+fn)*100:.1f}' if (tp+fn) > 0 else '0.0',
                    f'{tn/(tn+fp)*100:.1f}' if (tn+fp) > 0 else '0.0',
                    f'{r.test_accuracy:.1f}'
                ])
        
        table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                        cellLoc='center', colColours=['#4472C4']*len(headers))
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)
        
        for i in range(len(headers)):
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Detailed Classification Metrics', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.viz_dir, 'classification_metrics_table.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_validation_comparison(self, completed: Dict[str, ModelResult], sorted_models: List[str]):
        """Plot validation accuracy comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Validation Performance Comparison', fontsize=14, fontweight='bold')
        
        val_accs = [completed[m].val_accuracy for m in sorted_models]
        val_aucs = [completed[m].val_auc for m in sorted_models]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_models)))
        
        bars1 = axes[0].bar(sorted_models, val_accs, color=colors, edgecolor='black')
        axes[0].set_ylabel('Validation Accuracy (%)')
        axes[0].set_title('Validation Accuracy')
        axes[0].tick_params(axis='x', rotation=45)
        for bar, val in zip(bars1, val_accs):
            axes[0].annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
        
        bars2 = axes[1].bar(sorted_models, val_aucs, color=colors, edgecolor='black')
        axes[1].set_ylabel('Validation AUC-ROC')
        axes[1].set_title('Validation AUC-ROC')
        axes[1].tick_params(axis='x', rotation=45)
        for bar, val in zip(bars2, val_aucs):
            axes[1].annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.viz_dir, 'validation_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_cancer_metrics(self, completed: Dict[str, ModelResult], sorted_models: List[str]):
        """Plot cancer detection specific metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Cancer Detection Metrics Comparison', fontsize=14, fontweight='bold')
        
        sensitivity = [completed[m].test_sensitivity for m in sorted_models]
        specificity = [completed[m].test_specificity for m in sorted_models]
        f1_scores = [completed[m].test_f1 for m in sorted_models]
        
        x = np.arange(len(sorted_models))
        width = 0.6
        
        colors_sens = ['#e74c3c' if s < 70 else '#2ecc71' for s in sensitivity]
        bars1 = axes[0].bar(x, sensitivity, width, color=colors_sens, edgecolor='black')
        axes[0].set_ylabel('Sensitivity (%)')
        axes[0].set_title('Sensitivity (Cancer Detection Rate)')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(sorted_models, rotation=45, ha='right')
        axes[0].axhline(y=70, color='r', linestyle='--', alpha=0.5)
        for bar, val in zip(bars1, sensitivity):
            axes[0].annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
        
        colors_spec = ['#e74c3c' if s < 60 else '#3498db' for s in specificity]
        bars2 = axes[1].bar(x, specificity, width, color=colors_spec, edgecolor='black')
        axes[1].set_ylabel('Specificity (%)')
        axes[1].set_title('Specificity (False Positive Avoidance)')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(sorted_models, rotation=45, ha='right')
        axes[1].axhline(y=60, color='r', linestyle='--', alpha=0.5)
        for bar, val in zip(bars2, specificity):
            axes[1].annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
        
        colors_f1 = plt.cm.RdYlGn(np.array(f1_scores) / 100)
        bars3 = axes[2].bar(x, f1_scores, width, color=colors_f1, edgecolor='black')
        axes[2].set_ylabel('F1 Score (%)')
        axes[2].set_title('F1 Score (Balanced Metric)')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(sorted_models, rotation=45, ha='right')
        for bar, val in zip(bars3, f1_scores):
            axes[2].annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.viz_dir, 'cancer_detection_metrics.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_training_curves(self, completed: Dict[str, ModelResult], sorted_models: List[str]):
        """Plot training curves comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Curves Comparison - All Models', fontsize=14, fontweight='bold')
        
        for name in sorted_models:
            r = completed[name]
            if r.val_accs:
                epochs = range(1, len(r.val_accs) + 1)
                axes[0, 0].plot(epochs, r.train_accs, label=f'{name} (train)', linestyle='--', alpha=0.7)
                axes[0, 1].plot(epochs, r.val_accs, label=name, linewidth=2)
                axes[1, 0].plot(epochs, r.val_losses, label=name, linewidth=2)
                axes[1, 1].plot(epochs, r.val_aucs, label=name, linewidth=2)
        
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Training Accuracy (%)')
        axes[0, 0].set_title('Training Accuracy')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Validation Accuracy (%)')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Validation Loss')
        axes[1, 0].set_title('Validation Loss')
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation AUC-ROC')
        axes[1, 1].set_title('Validation AUC-ROC')
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.viz_dir, 'training_curves_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curves(self, completed: Dict[str, ModelResult], sorted_models: List[str]):
        """Plot ROC curves for all models."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_models)))
        ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)', alpha=0.5)
        
        for idx, model_name in enumerate(sorted_models):
            r = completed[model_name]
            if r.test_labels and r.test_probs:
                fpr, tpr, _ = roc_curve(r.test_labels, r.test_probs)
                ax.plot(fpr, tpr, color=colors[idx], linewidth=2,
                       label=f'{model_name} (AUC = {r.test_auc:.4f})')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves Comparison - All Models', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.viz_dir, 'roc_curves_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_efficiency_analysis(self, completed: Dict[str, ModelResult], sorted_models: List[str]):
        """Plot efficiency analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Model Efficiency Analysis', fontsize=14, fontweight='bold')
        
        test_accs = [completed[m].test_accuracy for m in sorted_models]
        inference_times = [completed[m].inference_time_ms for m in sorted_models]
        params = [completed[m].parameters / 1e6 for m in sorted_models]
        training_times = [completed[m].training_time_minutes for m in sorted_models]
        
        scatter1 = axes[0].scatter(inference_times, test_accs, s=[p*5 for p in params],
                                   c=training_times, cmap='viridis', alpha=0.7, edgecolors='black')
        for i, name in enumerate(sorted_models):
            axes[0].annotate(name, (inference_times[i], test_accs[i]), fontsize=8,
                           xytext=(5, 5), textcoords='offset points')
        axes[0].set_xlabel('Inference Time (ms)')
        axes[0].set_ylabel('Test Accuracy (%)')
        axes[0].set_title('Accuracy vs Speed\n(bubble size = parameters)')
        cbar1 = plt.colorbar(scatter1, ax=axes[0])
        cbar1.set_label('Training Time (min)')
        
        aucs = [completed[m].test_auc for m in sorted_models]
        colors = plt.cm.RdYlGn(np.array(aucs))
        bars = axes[1].barh(sorted_models, params, color=colors, edgecolor='black')
        axes[1].set_xlabel('Parameters (Millions)')
        axes[1].set_title('Model Size\n(color = AUC)')
        for bar, auc in zip(bars, aucs):
            axes[1].annotate(f'AUC: {auc:.3f}', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                           xytext=(5, 0), textcoords='offset points', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.viz_dir, 'efficiency_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_summary_table(self, completed: Dict[str, ModelResult], sorted_models: List[str]):
        """Create visualization of the summary table."""
        fig, ax = plt.subplots(figsize=(18, len(sorted_models) * 0.8 + 3))
        ax.axis('off')
        
        headers = ['Rank', 'Model', 'Params(M)', 'Val Acc%', 'Val AUC', 'Val Sens%', 
                   'Test Acc%', 'Test AUC', 'Sens%', 'Spec%', 'F1%', 'Time(min)']
        
        table_data = []
        for rank, name in enumerate(sorted_models, 1):
            r = completed[name]
            table_data.append([
                rank, name, f'{r.parameters/1e6:.2f}',
                f'{r.val_accuracy:.1f}', f'{r.val_auc:.4f}', f'{r.val_sensitivity:.1f}',
                f'{r.test_accuracy:.1f}', f'{r.test_auc:.4f}', 
                f'{r.test_sensitivity:.1f}', f'{r.test_specificity:.1f}',
                f'{r.test_f1:.1f}', f'{r.training_time_minutes:.1f}'
            ])
        
        table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                        cellLoc='center', colColours=['#2E75B6']*len(headers))
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2.0)
        
        for i in range(len(headers)):
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(len(headers)):
            table[(1, i)].set_facecolor('#C6EFCE')
        
        ax.set_title('BASELINE MODEL COMPARISON - FINAL RESULTS\n(Ranked by Validation AUC-ROC)',
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.viz_dir, 'final_summary_table.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_comparison_overview(self, completed: Dict[str, ModelResult], sorted_models: List[str]):
        """Create comprehensive visual comparison chart."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison Overview', fontsize=16, fontweight='bold')
        
        x = np.arange(len(sorted_models))
        width = 0.35
        
        val_accs = [completed[m].val_accuracy for m in sorted_models]
        test_accs = [completed[m].test_accuracy for m in sorted_models]
        
        axes[0, 0].bar(x - width/2, val_accs, width, label='Validation', color='#3498db', edgecolor='black')
        axes[0, 0].bar(x + width/2, test_accs, width, label='Test', color='#2ecc71', edgecolor='black')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('Validation vs Test Accuracy')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(sorted_models, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, axis='y', alpha=0.3)
        
        val_aucs = [completed[m].val_auc for m in sorted_models]
        test_aucs = [completed[m].test_auc for m in sorted_models]
        
        axes[0, 1].bar(x - width/2, val_aucs, width, label='Validation', color='#9b59b6', edgecolor='black')
        axes[0, 1].bar(x + width/2, test_aucs, width, label='Test', color='#e74c3c', edgecolor='black')
        axes[0, 1].set_ylabel('AUC-ROC')
        axes[0, 1].set_title('Validation vs Test AUC-ROC')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(sorted_models, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, axis='y', alpha=0.3)
        
        sensitivity = [completed[m].test_sensitivity for m in sorted_models]
        specificity = [completed[m].test_specificity for m in sorted_models]
        
        axes[1, 0].bar(x - width/2, sensitivity, width, label='Sensitivity', color='#e74c3c', edgecolor='black')
        axes[1, 0].bar(x + width/2, specificity, width, label='Specificity', color='#3498db', edgecolor='black')
        axes[1, 0].set_ylabel('Percentage (%)')
        axes[1, 0].set_title('Sensitivity vs Specificity')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(sorted_models, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, axis='y', alpha=0.3)
        
        params = [completed[m].parameters / 1e6 for m in sorted_models]
        aucs = [completed[m].test_auc for m in sorted_models]
        times = [completed[m].training_time_minutes for m in sorted_models]
        
        scatter = axes[1, 1].scatter(params, aucs, s=[t*20 for t in times], c=test_accs, 
                                     cmap='RdYlGn', alpha=0.7, edgecolors='black')
        for i, name in enumerate(sorted_models):
            axes[1, 1].annotate(name, (params[i], aucs[i]), fontsize=9,
                               xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('Parameters (Millions)')
        axes[1, 1].set_ylabel('Test AUC-ROC')
        axes[1, 1].set_title('Model Size vs Performance')
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label('Test Accuracy (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.viz_dir, 'model_comparison_overview.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_individual_model_summaries(self, completed: Dict[str, ModelResult], sorted_models: List[str]):
        """Generate individual training summary for each model using plots (no table)."""
        for model_name in sorted_models:
            r = completed[model_name]
            if not r.val_accs:
                continue
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'{model_name} - Training Summary\nBest Epoch: {r.best_epoch} | Final Val AUC: {r.val_auc:.4f}', 
                        fontsize=14, fontweight='bold')
            
            epochs = range(1, len(r.val_accs) + 1)
            
            # Plot 1: Loss Curves
            axes[0, 0].plot(epochs, r.train_losses, 'b-', label='Train Loss', linewidth=2)
            axes[0, 0].plot(epochs, r.val_losses, 'r-', label='Val Loss', linewidth=2)
            axes[0, 0].axvline(x=r.best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({r.best_epoch})')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Accuracy Curves
            axes[0, 1].plot(epochs, r.train_accs, 'b-', label='Train Acc', linewidth=2)
            axes[0, 1].plot(epochs, r.val_accs, 'r-', label='Val Acc', linewidth=2)
            axes[0, 1].axvline(x=r.best_epoch, color='g', linestyle='--', alpha=0.7)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].set_title('Accuracy Curves')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Validation AUC
            axes[0, 2].plot(epochs, r.val_aucs, 'g-', linewidth=2)
            axes[0, 2].axhline(y=r.val_auc, color='r', linestyle='--', alpha=0.7, label=f'Best AUC ({r.val_auc:.4f})')
            axes[0, 2].axvline(x=r.best_epoch, color='g', linestyle='--', alpha=0.5)
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('AUC-ROC')
            axes[0, 2].set_title('Validation AUC')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            # Plot 4: Learning Rate Schedule
            axes[1, 0].plot(epochs, r.learning_rates, 'm-', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 5: Overfitting Gap (Train Acc - Val Acc) - more useful than a table
            gap = [t - v for t, v in zip(r.train_accs, r.val_accs)]
            axes[1, 1].fill_between(epochs, gap, alpha=0.3, color='orange', label='Overfitting Gap')
            axes[1, 1].plot(epochs, gap, 'orange', linewidth=2)
            axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            axes[1, 1].axvline(x=r.best_epoch, color='g', linestyle='--', alpha=0.7)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Train Acc - Val Acc (%)')
            axes[1, 1].set_title('Overfitting Gap (Lower is Better)')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Plot 6: Final Results Summary (text box)
            axes[1, 2].axis('off')
            summary_text = f"""
FINAL TEST METRICS

Accuracy:     {r.test_accuracy:.2f}%
Sensitivity:  {r.test_sensitivity:.2f}%
Specificity:  {r.test_specificity:.2f}%
AUC-ROC:      {r.test_auc:.4f}
F1-Score:     {r.test_f1:.2f}%

TRAINING INFO

Parameters:   {r.parameters/1e6:.2f}M
Best Epoch:   {r.best_epoch}
Train Time:   {r.training_time_minutes:.1f} min
"""
            axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                           fontsize=11, verticalalignment='center', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            axes[1, 2].set_title('Final Results', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.viz_dir, f'{model_name}_training_summary.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
