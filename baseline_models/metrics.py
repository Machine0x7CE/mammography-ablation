"""
Performance metrics calculation for classification tasks.
"""

import numpy as np
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""
    loss: float = 0.0
    accuracy: float = 0.0
    sensitivity: float = 0.0
    specificity: float = 0.0
    auc_roc: float = 0.0
    f1: float = 0.0
    precision: float = 0.0


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> EpochMetrics:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities for positive class
    
    Returns:
        EpochMetrics object with all calculated metrics
    """
    metrics = EpochMetrics()
    
    metrics.accuracy = accuracy_score(y_true, y_pred) * 100
    metrics.precision = precision_score(y_true, y_pred, zero_division=0) * 100
    metrics.sensitivity = recall_score(y_true, y_pred, zero_division=0) * 100
    metrics.f1 = f1_score(y_true, y_pred, zero_division=0) * 100
    
    # Calculate specificity from confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.specificity = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0.0
    
    # AUC-ROC
    try:
        metrics.auc_roc = roc_auc_score(y_true, y_prob)
    except:
        metrics.auc_roc = 0.5
    
    return metrics


def get_confusion_matrix_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Get detailed confusion matrix statistics.
    
    Returns dictionary with TP, TN, FP, FN counts and percentages.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    
    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp_pct': tp / total * 100,
        'tn_pct': tn / total * 100,
        'fp_pct': fp / total * 100,
        'fn_pct': fn / total * 100,
        'sensitivity': tp / (tp + fn) * 100 if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) * 100 if (tn + fp) > 0 else 0,
        'accuracy': (tp + tn) / total * 100
    }

