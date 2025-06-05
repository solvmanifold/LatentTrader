import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from trading_advisor.models.logistic_model import LogisticTradingModel
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_split(split_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test sets from a split directory."""
    train_df = pd.read_parquet(split_dir / "train.parquet")
    val_df = pd.read_parquet(split_dir / "val.parquet")
    test_df = pd.read_parquet(split_dir / "test.parquet")
    return train_df, val_df, test_df

def analyze_dataset(df: pd.DataFrame, name: str) -> None:
    """Analyze and log information about a dataset."""
    logger.info(f"\n{name} Dataset Analysis:")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Positive labels: {df['label'].mean():.2%}")
    
    # Check for NaN values
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        logger.warning(f"Found NaN values in columns: {nan_cols}")
        for col in nan_cols:
            nan_count = df[col].isna().sum()
            nan_pct = nan_count / len(df) * 100
            logger.warning(f"Column {col} has {nan_count} NaN values ({nan_pct:.1f}%)")
    
    # Check for infinite values
    inf_cols = df.columns[np.isinf(df.select_dtypes(include=np.number)).any()].tolist()
    if inf_cols:
        logger.warning(f"Found infinite values in columns: {inf_cols}")
    
    # Check for zero variance features
    zero_var_cols = df.columns[df.nunique() == 1].tolist()
    if zero_var_cols:
        logger.warning(f"Found zero variance features: {zero_var_cols}")

def preprocess_features(df: pd.DataFrame, scaler: StandardScaler = None, fit: bool = False) -> Tuple[np.ndarray, StandardScaler]:
    """Preprocess features by scaling and handling missing values."""
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['label']]
    
    # Handle missing values
    df_clean = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Scale features
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(df_clean)
    else:
        X = scaler.transform(df_clean)
    
    return X, scaler

def plot_feature_importance(model: LogisticTradingModel, output_dir: Path, split_num: int) -> pd.DataFrame:
    """Plot feature importance from the trained model."""
    if hasattr(model.model, 'coef_'):
        feature_importance = np.abs(model.model.coef_[0])
        features = model.feature_columns
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df.head(20), x='importance', y='feature')
        plt.title(f'Top 20 Most Important Features (Split {split_num})')
        plt.tight_layout()
        plt.savefig(output_dir / f'feature_importance_split_{split_num}.png')
        plt.close()
        
        # Log top 10 features
        logger.info(f"\nTop 10 most important features for split {split_num}:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.4f}")
        
        return importance_df
    return pd.DataFrame()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path, split_num: int) -> None:
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Split {split_num})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_matrix_split_{split_num}.png')
    plt.close()

def train_and_evaluate_split(
    split_num: int,
    output_dir: Path,
    model_params: Dict
) -> Tuple[Dict, pd.DataFrame]:
    """Train and evaluate model on a specific split."""
    split_dir = Path(f"data/ml_datasets/split_{split_num}")
    train_df, val_df, test_df = load_split(split_dir)
    
    # Analyze datasets
    logger.info(f"\n=== Split {split_num} ===")
    analyze_dataset(train_df, f"Training (Split {split_num})")
    analyze_dataset(val_df, f"Validation (Split {split_num})")
    analyze_dataset(test_df, f"Test (Split {split_num})")
    
    # Preprocess features
    X_train, scaler = preprocess_features(train_df, fit=True)
    X_val, _ = preprocess_features(val_df, scaler=scaler)
    X_test, _ = preprocess_features(test_df, scaler=scaler)
    
    # Get labels
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    # Initialize and train model
    logger.info(f"\nTraining logistic model for split {split_num}...")
    model = LogisticTradingModel(**model_params)
    
    # Create training DataFrame with preprocessed features
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['label']]
    
    train_processed = pd.DataFrame(X_train, columns=numeric_cols)
    train_processed['label'] = y_train
    val_processed = pd.DataFrame(X_val, columns=numeric_cols)
    val_processed['label'] = y_val
    
    model.train(train_processed, val_processed)
    
    # Evaluate on test set
    logger.info(f"\nEvaluating split {split_num} on test set...")
    test_processed = pd.DataFrame(X_test, columns=numeric_cols)
    test_processed['label'] = y_test
    
    test_results = model.predict(test_processed)
    test_pred = test_results['predictions']
    test_proba = test_results['probabilities']
    
    # Calculate metrics
    test_metrics = {
        'split': split_num,
        'accuracy': np.mean(test_pred == y_test),
        'precision': precision_score(y_test, test_pred, zero_division=0),
        'recall': recall_score(y_test, test_pred, zero_division=0),
        'auc': roc_auc_score(y_test, test_proba)
    }
    
    # Log test metrics
    logger.info(f"\nTest Set Metrics for split {split_num}:")
    for metric, value in test_metrics.items():
        if metric != 'split':
            logger.info(f"{metric}: {value:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, test_pred, output_dir, split_num)
    
    # Plot feature importance
    importance_df = plot_feature_importance(model, output_dir, split_num)
    
    # Save model and scaler
    model_path = output_dir / f"logistic_model_split_{split_num}.pkl"
    model.save(model_path)
    logger.info(f"\nSaved trained model to {model_path}")
    
    return test_metrics, importance_df

def aggregate_results(metrics_list: List[Dict], importance_dfs: List[pd.DataFrame]) -> None:
    """Aggregate and analyze results across all splits."""
    # Aggregate metrics
    metrics_df = pd.DataFrame(metrics_list)
    logger.info("\n=== Aggregated Results Across All Splits ===")
    logger.info("\nMetrics Summary:")
    for metric in ['accuracy', 'precision', 'recall', 'auc']:
        mean_val = metrics_df[metric].mean()
        std_val = metrics_df[metric].std()
        logger.info(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Aggregate feature importance
    all_features = []
    for df in importance_dfs:
        if not df.empty:
            all_features.extend(df['feature'].unique())
    all_features = list(set(all_features))
    
    feature_importance = pd.DataFrame(index=all_features)
    for i, df in enumerate(importance_dfs):
        if not df.empty:
            feature_importance[f'split_{i}'] = df.set_index('feature')['importance']
    
    feature_importance['mean_importance'] = feature_importance.mean(axis=1)
    feature_importance['std_importance'] = feature_importance.std(axis=1)
    feature_importance = feature_importance.sort_values('mean_importance', ascending=False)
    
    logger.info("\nTop 10 Most Important Features (Average Across Splits):")
    for feature, row in feature_importance.head(10).iterrows():
        logger.info(f"{feature}: {row['mean_importance']:.4f} ± {row['std_importance']:.4f}")

def main():
    # Create output directory
    output_dir = Path("model_outputs/logistic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model parameters - try different regularization strengths
    model_params = {
        'C': 0.1,  # Stronger regularization
        'class_weight': 'balanced',
        'max_iter': 1000,
        'random_state': 42,
        'solver': 'liblinear'  # Better for small datasets
    }
    
    # Train and evaluate on all splits
    metrics_list = []
    importance_dfs = []
    
    for split_num in range(6):  # We have splits 0-5
        try:
            metrics, importance_df = train_and_evaluate_split(
                split_num=split_num,
                output_dir=output_dir,
                model_params=model_params
            )
            metrics_list.append(metrics)
            importance_dfs.append(importance_df)
        except Exception as e:
            logger.error(f"Error processing split {split_num}: {str(e)}")
            continue
    
    # Aggregate results
    if metrics_list and importance_dfs:
        aggregate_results(metrics_list, importance_dfs)
    else:
        logger.error("No successful splits to aggregate results from")

if __name__ == "__main__":
    main() 