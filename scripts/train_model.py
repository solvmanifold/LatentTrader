"""Base script for training any model type."""

import logging
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import shutil

from trading_advisor.models import registry
from trading_advisor.models.runner import ModelRunner
from trading_advisor.utils import setup_logging

# Set up logger
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a trading model')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Directory containing the datasets (e.g., data/2024_all/swing_trade)')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Directory to save model and artifacts (e.g., output/models/2024_all_logistic)')
    parser.add_argument('--label-type', type=str, required=True,
                      help='Type of labels to use (e.g., swing_trade)')
    parser.add_argument('--model-type', type=str, required=True,
                      help='Type of model to train (e.g., logistic, xgboost)')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level')
    return parser.parse_args()

def load_data(data_dir: Path, label_type: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test datasets with their labels.
    
    Args:
        data_dir: Directory containing the datasets
        label_type: Type of labels to use (e.g., 'swing_trade')
        
    Returns:
        Tuple of (train_features, train_labels, val_features, val_labels, test_features, test_labels)
    """
    # Load features from parent directory
    parent_dir = data_dir.parent
    train_features = pd.read_parquet(parent_dir / "train.parquet")
    val_features = pd.read_parquet(parent_dir / "val.parquet")
    test_features = pd.read_parquet(parent_dir / "test.parquet")
    
    # Load labels from label_type subdirectory
    train_labels = pd.read_parquet(data_dir / "train_labels.parquet")
    val_labels = pd.read_parquet(data_dir / "val_labels.parquet")
    test_labels = pd.read_parquet(data_dir / "test_labels.parquet")
    
    # Ensure the labels have the correct column name
    for df in [train_labels, val_labels, test_labels]:
        if 'label' not in df.columns:
            # Assuming the first column is the label column
            df.rename(columns={df.columns[0]: 'label'}, inplace=True)
    
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
    """Plot confusion matrix for three-class classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_dir: Directory to save plots
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Short (-1)', 'No Trade (0)', 'Long (1)'],
                yticklabels=['Short (-1)', 'No Trade (0)', 'Long (1)'])
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

def generate_readme(
    output_dir: Path,
    config: dict,
    train_metrics: dict,
    test_metrics: dict,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame
) -> None:
    """Generate a README file for the model training run.
    
    Args:
        output_dir: Directory where model and artifacts are saved
        config: Training configuration dictionary
        train_metrics: Dictionary of training metrics
        test_metrics: Dictionary of test metrics
        train_data: Training dataset
        val_data: Validation dataset
        test_data: Test dataset
    """
    # Get command used to run the script
    cmd = ' '.join(sys.argv)
    
    # Calculate class distribution
    def get_class_distribution(df):
        dist = df['label'].value_counts()
        total = len(df)
        return {label: f"{count} ({count/total*100:.1f}%)" for label, count in dist.items()}
    
    train_dist = get_class_distribution(train_data)
    val_dist = get_class_distribution(val_data)
    test_dist = get_class_distribution(test_data)
    
    readme_content = f"""# Model Training Results

## Overview
This model was trained on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using the following command:
```bash
{cmd}
```

## Configuration
- Data Directory: {config['data_dir']}
- Label Type: {config['label_type']}
- Model Type: {config['model_type']}
- Timestamp: {config['timestamp']}

## Dataset Statistics
### Training Set
- Samples: {len(train_data)}
- Date Range: {train_data.index.min().strftime('%Y-%m-%d')} to {train_data.index.max().strftime('%Y-%m-%d')}
- Unique Tickers: {train_data['ticker'].nunique()}
- Class Distribution:
{chr(10).join(f'  - {label}: {count}' for label, count in train_dist.items())}

### Validation Set
- Samples: {len(val_data)}
- Date Range: {val_data.index.min().strftime('%Y-%m-%d')} to {val_data.index.max().strftime('%Y-%m-%d')}
- Unique Tickers: {val_data['ticker'].nunique()}
- Class Distribution:
{chr(10).join(f'  - {label}: {count}' for label, count in val_dist.items())}

### Test Set
- Samples: {len(test_data)}
- Date Range: {test_data.index.min().strftime('%Y-%m-%d')} to {test_data.index.max().strftime('%Y-%m-%d')}
- Unique Tickers: {test_data['ticker'].nunique()}
- Class Distribution:
{chr(10).join(f'  - {label}: {count}' for label, count in test_dist.items())}

## Training Metrics
{chr(10).join(f'- {metric}: {value:.4f}' for metric, value in train_metrics.items())}

## Test Metrics
{chr(10).join(f'- {metric}: {value:.4f}' for metric, value in test_metrics.items())}

## Feature Normalization
The model uses pre-normalized data from the dataset:
- Training data is already normalized during dataset generation
- Normalization parameters are saved in {config['data_dir']}/scalers/ for inference on new data
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    logger.info(f"Generated README at {readme_path}")

def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging with specified level
    setup_logging(log_level=getattr(logging, args.log_level))
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    label_type = args.label_type
    model_type = args.model_type
    
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
    logger.info(f"Loading data from {data_dir}")
    train_features, train_labels, val_features, val_labels, test_features, test_labels = load_data(data_dir, label_type)
    
    # Merge features and labels
    train_data = train_features.copy()
    val_data = val_features.copy()
    test_data = test_features.copy()
    
    train_data['label'] = train_labels['label']
    val_data['label'] = val_labels['label']
    test_data['label'] = test_labels['label']
    
    # Create model
    logger.info(f"Creating {model_type} model")
    model = registry.create_model(model_type)
    
    # Train model with hyperparameter tuning
    logger.info("Training model...")
    train_metrics = model.train(train_data, val_data, tune_hyperparameters=True)
    
    # Log training metrics
    logger.info("Training metrics:")
    for metric, value in train_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    results = model.predict(test_data)
    
    # Calculate metrics
    test_metrics = {
        'accuracy': accuracy_score(test_data['label'], results['predictions']),
        'precision_macro': precision_score(test_data['label'], results['predictions'], average='macro'),
        'precision_micro': precision_score(test_data['label'], results['predictions'], average='micro'),
        'precision_weighted': precision_score(test_data['label'], results['predictions'], average='weighted'),
        'recall_macro': recall_score(test_data['label'], results['predictions'], average='macro'),
        'recall_micro': recall_score(test_data['label'], results['predictions'], average='micro'),
        'recall_weighted': recall_score(test_data['label'], results['predictions'], average='weighted')
    }
    
    # Log test metrics
    logger.info("Test metrics:")
    for metric, value in test_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Generate plots
    logger.info("Generating plots...")
    plot_feature_importance(model, output_dir)
    plot_confusion_matrix(test_data['label'], results['predictions'], output_dir)
    
    # Save model
    logger.info(f"Saving model to {output_dir}")
    model.save(output_dir / 'model')
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'metric': list(test_metrics.keys()),
        'value': list(test_metrics.values())
    })
    metrics_df.to_csv(output_dir / 'metrics.csv', index=False)
    
    # Generate README
    logger.info("Generating README...")
    generate_readme(output_dir, config, train_metrics, test_metrics, train_data, val_data, test_data)
    
    logger.info(f"Model and artifacts saved to {output_dir}")

if __name__ == '__main__':
    main() 