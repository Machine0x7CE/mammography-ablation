"""
Benchmarking system for model comparison and reporting.
"""

import os
import json
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List

from .config import TrainingConfig


@dataclass
class ModelResult:
    """Complete results for a trained model."""
    model_name: str
    status: str = "pending"
    timestamp: str = ""
    
    # Validation metrics
    val_accuracy: float = 0.0
    val_sensitivity: float = 0.0
    val_specificity: float = 0.0
    val_auc: float = 0.0
    val_f1: float = 0.0
    
    # Test metrics
    test_accuracy: float = 0.0
    test_sensitivity: float = 0.0
    test_specificity: float = 0.0
    test_auc: float = 0.0
    test_f1: float = 0.0
    
    # Training info
    best_epoch: int = 0
    final_lr: float = 0.0
    train_accuracy: float = 0.0
    overfitting_gap: float = 0.0
    optimizer: str = ""
    scheduler: str = ""
    
    # Efficiency metrics
    parameters: int = 0
    training_time_minutes: float = 0.0
    inference_time_ms: float = 0.0
    samples_per_second: float = 0.0
    peak_gpu_memory_gb: float = 0.0
    
    # History
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_accs: List[float] = field(default_factory=list)
    val_accs: List[float] = field(default_factory=list)
    val_aucs: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    
    # For confusion matrix visualization
    test_predictions: List[int] = field(default_factory=list)
    test_labels: List[int] = field(default_factory=list)
    test_probs: List[float] = field(default_factory=list)


class ModelBenchmark:
    """Benchmarking system for model comparison."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.results: Dict[str, ModelResult] = {}
        self.benchmark_start_time = datetime.now()
        
        for model_name in config.models_to_train:
            self.results[model_name] = ModelResult(model_name=model_name)
        
        self.log_path = os.path.join(config.output_root, 'benchmark_log.json')
    
    def update_model_status(self, model_name: str, status: str):
        """Update model training status."""
        self.results[model_name].status = status
        if status == "complete":
            self.results[model_name].timestamp = datetime.now().isoformat()
    
    def add_result(self, model_name: str, result: ModelResult):
        """Add completed model result."""
        self.results[model_name] = result
        self.save_log()
    
    def get_models_progress(self) -> str:
        """Get a simple progress string showing model status."""
        parts = []
        for name, r in self.results.items():
            if r.status == 'complete':
                parts.append(f"[{name} DONE]")
            elif r.status == 'training':
                parts.append(f"[{name} ...]")
            else:
                parts.append(f"[{name}]")
        return " ".join(parts)
    
    def get_completed_results(self) -> Dict[str, ModelResult]:
        """Get dictionary of completed model results."""
        return {name: r for name, r in self.results.items() if r.status == "complete"}
    
    def get_sorted_models(self) -> List[str]:
        """Get model names sorted by validation AUC."""
        completed = self.get_completed_results()
        return sorted(completed.keys(), key=lambda n: completed[n].val_auc, reverse=True)
    
    def save_log(self):
        """Save benchmark log to JSON."""
        completed = [name for name, r in self.results.items() if r.status == "complete"]
        
        current_leader = None
        if completed:
            leader_name = max(completed, key=lambda n: self.results[n].val_auc)
            current_leader = {
                'model': leader_name,
                'val_auc': self.results[leader_name].val_auc
            }
        
        log_data = {
            'benchmark_started': self.benchmark_start_time.isoformat(),
            'last_updated': datetime.now().isoformat(),
            'models_completed': len(completed),
            'models_total': len(self.results),
            'current_leader': current_leader,
            'results': {name: asdict(r) for name, r in self.results.items()}
        }
        
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, default=str)
    
    def create_summary_dataframe(self) -> pd.DataFrame:
        """Create summary DataFrame of all completed models."""
        completed = self.get_completed_results()
        sorted_models = self.get_sorted_models()
        
        summary_data = []
        for rank, name in enumerate(sorted_models, 1):
            r = completed[name]
            summary_data.append({
                'Rank': rank,
                'Model': name,
                'Params (M)': f"{r.parameters/1e6:.2f}",
                'Val Acc (%)': f"{r.val_accuracy:.2f}",
                'Val AUC': f"{r.val_auc:.4f}",
                'Val Sens (%)': f"{r.val_sensitivity:.2f}",
                'Val Spec (%)': f"{r.val_specificity:.2f}",
                'Test Acc (%)': f"{r.test_accuracy:.2f}",
                'Test AUC': f"{r.test_auc:.4f}",
                'Test Sens (%)': f"{r.test_sensitivity:.2f}",
                'Test Spec (%)': f"{r.test_specificity:.2f}",
                'F1 (%)': f"{r.test_f1:.2f}",
                'Time (min)': f"{r.training_time_minutes:.1f}",
                'Best Epoch': r.best_epoch
            })
        
        return pd.DataFrame(summary_data)
    
    def print_final_report(self):
        """Print final benchmark report to console."""
        completed = self.get_completed_results()
        
        if not completed:
            print("WARNING: No completed models to report")
            return
        
        sorted_models = self.get_sorted_models()
        summary_df = self.create_summary_dataframe()
        
        print("\n")
        print("=" * 100)
        print("BASELINE MODEL COMPARISON - FINAL BENCHMARK REPORT")
        print("=" * 100)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Models Evaluated: {len(completed)}")
        print("=" * 100)
        
        print("\n")
        print("PERFORMANCE RANKING (sorted by Validation AUC)")
        print("-" * 140)
        print(summary_df.to_string(index=False))
        print("-" * 140)
        
        # Best model recommendation
        best_model = sorted_models[0]
        best_result = completed[best_model]
        
        print("\n")
        print("BEST MODEL RECOMMENDATION")
        print("-" * 50)
        print(f"Model:              {best_model}")
        print(f"Validation AUC:     {best_result.val_auc:.4f}")
        print(f"Validation Acc:     {best_result.val_accuracy:.2f}%")
        print(f"Test Accuracy:      {best_result.test_accuracy:.2f}%")
        print(f"Sensitivity:        {best_result.test_sensitivity:.2f}%")
        print(f"Specificity:        {best_result.test_specificity:.2f}%")
        print(f"F1-Score:           {best_result.test_f1:.2f}%")
        print("-" * 50)
        
        # Save CSV
        summary_df.to_csv(
            os.path.join(self.config.output_root, 'comparison_summary.csv'), 
            index=False
        )
    
    def generate_markdown_report(self):
        """Generate markdown report file."""
        completed = self.get_completed_results()
        sorted_models = self.get_sorted_models()
        
        if not completed:
            return
        
        best_model = sorted_models[0]
        best_result = completed[best_model]
        
        report = f"""# Baseline Model Comparison Report

## Summary

**Best Model**: {best_model}
**Validation AUC**: {best_result.val_auc:.4f}
**Test Accuracy**: {best_result.test_accuracy:.2f}%
**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Model Rankings (by Validation AUC)

| Rank | Model | Val AUC | Val Acc | Test Acc | Sensitivity | Specificity | F1 | Params (M) |
|------|-------|---------|---------|----------|-------------|-------------|-----|------------|
"""
        
        for rank, name in enumerate(sorted_models, 1):
            r = completed[name]
            report += f"| {rank} | {name} | {r.val_auc:.4f} | {r.val_accuracy:.1f}% | {r.test_accuracy:.1f}% | {r.test_sensitivity:.1f}% | {r.test_specificity:.1f}% | {r.test_f1:.1f}% | {r.parameters/1e6:.2f} |\n"
        
        report += f"""

## Individual Model Details

"""
        from .models import MODEL_HYPERPARAMS
        
        for name in sorted_models:
            r = completed[name]
            model_config = MODEL_HYPERPARAMS.get(name, {})
            report += f"""### {name}

- **Description**: {model_config.get('description', 'N/A')}
- **Parameters**: {r.parameters/1e6:.2f}M
- **Best Epoch**: {r.best_epoch}
- **Training Time**: {r.training_time_minutes:.1f} minutes
- **Validation Accuracy**: {r.val_accuracy:.2f}%
- **Validation AUC**: {r.val_auc:.4f}
- **Test Accuracy**: {r.test_accuracy:.2f}%
- **Test AUC**: {r.test_auc:.4f}
- **Sensitivity**: {r.test_sensitivity:.2f}%
- **Specificity**: {r.test_specificity:.2f}%
- **F1 Score**: {r.test_f1:.2f}%

"""
        
        report += """## Visualizations

- confusion_matrices.png - Confusion matrix for each model
- validation_comparison.png - Validation accuracy and AUC comparison
- cancer_detection_metrics.png - Sensitivity, specificity, and F1 comparison
- training_curves_comparison.png - Training curves for all models
- roc_curves_comparison.png - ROC curves comparison
- final_summary_table.png - Summary table visualization
- model_comparison_overview.png - All metrics comparison

---
*Report generated automatically*
"""
        
        report_path = os.path.join(self.config.output_root, 'final_benchmark_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Markdown report saved to: {report_path}")

