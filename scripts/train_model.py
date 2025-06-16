"""Base script for training any model type."""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

from trading_advisor.models import registry
from trading_advisor.utils import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

def load_data(data_dir: Path, label_type: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test datasets with their labels.
    
    Args:
        data_dir: Directory containing the datasets
        label_type: Type of labels to use (e.g., 'next_day_return', 'volatility_regime')
        
    Returns:
        Tuple of (train_features, train_labels, val_features, val_labels, test_features, test_labels)
    """
    # Load features
    train_features = pd.read_parquet(data_dir / "train_features.parquet")
    val_features = pd.read_parquet(data_dir / "val_features.parquet")
    test_features = pd.read_parquet(data_dir / "test_features.parquet")
    
    # Load labels
    label_dir = data_dir / label_type
    train_labels = pd.read_parquet(label_dir / "train_labels.parquet")
    val_labels = pd.read_parquet(label_dir / "val_labels.parquet")
    test_labels = pd.read_parquet(label_dir / "test_labels.parquet")
    
    logger.info(f"Loaded datasets:")
    logger.info(f"Train: {len(train_features)} samples")
    logger.info(f"Validation: {len(val_features)} samples")
    logger.info(f"Test: {len(test_features)} samples")
    
    return train_features, train_labels, val_features, val_labels, test_features, test_labels

def plot_feature_importance(model, output_dir: Path):
    """Plot feature importance.
    
    Args:
        model: Trained model instance
        output_dir: Directory to save plots
    """
    if not hasattr(model, 'metadata') or 'feature_importance' not in model.metadata:
        return
        
    importance = model.metadata['feature_importance']
    features = list(importance.keys())
    scores = list(importance.values())
    
    # Sort by absolute importance
    sorted_idx = np.argsort(np.abs(scores))
    features = [features[i] for i in sorted_idx]
    scores = [scores[i] for i in sorted_idx]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(features)), scores)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'feature_importance.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_dir: Path):
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_dir: Directory to save plots
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'confusion_matrix.png')
    plt.close()

def save_training_config(config: dict, output_dir: Path):
    """Save training configuration.
    
    Args:
        config: Dictionary containing training configuration
        output_dir: Directory to save config
    """
    with open(output_dir / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2)

def main():
    """Main training function."""
    # Parse command line arguments
    if len(sys.argv) != 5:
        print("Usage: python train_model.py <data_dir> <output_dir> <label_type> <model_type>")
        print("Example: python train_model.py data/ml_datasets models/logistic_v1 next_day_return logistic")
        sys.exit(1)
        
    data_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    label_type = sys.argv[3]
    model_type = sys.argv[4]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training configuration
    config = {
        'data_dir': str(data_dir),
        'label_type': label_type,
        'model_type': model_type,
        'timestamp': datetime.now().isoformat()
    }
    save_training_config(config, output_dir)
    
    # Load data
    train_features, train_labels, val_features, val_labels, test_features, test_labels = load_data(data_dir, label_type)
    
    # Create model
    model = registry.create_model(model_type)
    
    # Train model with hyperparameter tuning
    logger.info("Training model...")
    metrics = model.train(train_features, val_features, tune_hyperparameters=True)
    
    # Log training metrics
    logger.info("Training metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    results = model.predict(test_features)
    
    # Calculate metrics
    test_metrics = {
        'accuracy': (results['predictions'] == test_labels['label']).mean(),
        'auc': roc_auc_score(test_labels['label'], results['probabilities']),
        'precision': precision_score(test_labels['label'], results['predictions']),
        'recall': recall_score(test_labels['label'], results['predictions'])
    }
    
    # Log test metrics
    logger.info("Test metrics:")
    for metric, value in test_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Generate plots
    plot_feature_importance(model, output_dir)
    plot_confusion_matrix(test_labels['label'], results['predictions'], output_dir)
    
    # Save model
    model.save(output_dir / 'model')
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'metric': list(test_metrics.keys()),
        'value': list(test_metrics.values())
    })
    metrics_df.to_csv(output_dir / 'metrics.csv', index=False)
    
    logger.info(f"Model and artifacts saved to {output_dir}")

if __name__ == '__main__':
    main() 