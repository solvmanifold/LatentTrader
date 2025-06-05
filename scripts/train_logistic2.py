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
import random
import joblib
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set all random seeds for reproducibility
def set_random_seeds(seed: int = 42):
    """Set random seeds for all operations."""
    np.random.seed(seed)
    random.seed(seed)

def load_dataset(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test datasets."""
    train_df = pd.read_parquet(data_dir / 'train.parquet')
    val_df = pd.read_parquet(data_dir / 'val.parquet')
    test_df = pd.read_parquet(data_dir / 'test.parquet')
    return train_df, val_df, test_df

def analyze_dataset(df: pd.DataFrame, name: str):
    """Analyze dataset characteristics."""
    logger.info(f"\n{name} Dataset Analysis:")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Label distribution: {df['label'].value_counts(normalize=True).to_dict()}")
    logger.info(f"Number of unique tickers: {df['ticker'].nunique()}")
    logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

def preprocess_features(df: pd.DataFrame, scaler: StandardScaler = None, fit: bool = False) -> Tuple[np.ndarray, StandardScaler]:
    """Preprocess features by scaling and handling missing values."""
    # Select numeric columns and sort to ensure consistent order
    numeric_cols = sorted(df.select_dtypes(include=[np.number]).columns)
    numeric_cols = [col for col in numeric_cols if col not in ['label', 'ticker']]
    
    # Handle missing values
    df_clean = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Scale features
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(df_clean)
    else:
        X = scaler.transform(df_clean)
    
    return X, scaler

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path, split_num: int):
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

def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Path,
    model_params: Dict
) -> Tuple[LogisticTradingModel, StandardScaler, List[str]]:
    """Train model on the dataset and return the trained model, scaler, and feature columns."""
    # Analyze datasets
    analyze_dataset(train_df, "Training")
    analyze_dataset(val_df, "Validation")

    # Preprocess features
    X_train, scaler = preprocess_features(train_df, fit=True)
    X_val, _ = preprocess_features(val_df, scaler=scaler)

    # Get feature columns (actual order used for training)
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['label', 'ticker']]

    # Save scaler and feature columns
    scaler_path = output_dir / "scaler.pkl"
    features_path = output_dir / "features.pkl"
    joblib.dump(scaler, scaler_path)
    joblib.dump(numeric_cols, features_path)

    # Get labels
    y_train = train_df['label'].values
    y_val = val_df['label'].values

    # Initialize and train model
    logger.info("\nTraining logistic model...")
    model = LogisticTradingModel(**model_params)

    # Create training DataFrame with preprocessed features
    train_processed = pd.DataFrame(X_train, columns=numeric_cols)
    train_processed['label'] = y_train
    val_processed = pd.DataFrame(X_val, columns=numeric_cols)
    val_processed['label'] = y_val

    model.train(train_processed, val_processed)

    # Save model
    model_path = output_dir / "logistic_model.pkl"
    model.save(model_path)
    logger.info(f"\nSaved trained model to {model_path}")

    return model, scaler, numeric_cols

def evaluate(
    model: LogisticTradingModel,
    test_df: pd.DataFrame,
    scaler: StandardScaler,
    numeric_cols: List[str],
    output_dir: Path
) -> Tuple[Dict, pd.DataFrame]:
    """Evaluate model on the test dataset."""
    # Analyze test dataset
    analyze_dataset(test_df, "Test")

    # Preprocess features
    X_test, _ = preprocess_features(test_df, scaler=scaler)
    y_test = test_df['label'].values

    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_processed = pd.DataFrame(X_test, columns=numeric_cols)
    test_processed['label'] = y_test

    test_results = model.predict(test_processed)
    test_pred = test_results['predictions']
    test_proba = test_results['probabilities']

    # Calculate metrics
    test_metrics = {
        'accuracy': np.mean(test_pred == y_test),
        'precision': precision_score(y_test, test_pred, zero_division=0),
        'recall': recall_score(y_test, test_pred, zero_division=0),
        'auc': roc_auc_score(y_test, test_proba)
    }

    # Log test metrics
    logger.info("\nTest Set Metrics:")
    for metric, value in test_metrics.items():
        logger.info(f"{metric}: {value:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(y_test, test_pred, output_dir, 0)

    # Plot feature importance
    importance_df = plot_feature_importance(model, output_dir, 0)

    return test_metrics, importance_df

def train_and_evaluate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    model_params: Dict
) -> Tuple[Dict, pd.DataFrame]:
    """Train and evaluate model on the dataset."""
    # Train the model
    model, scaler, numeric_cols = train(train_df, val_df, output_dir, model_params)
    
    # Evaluate the model
    test_metrics, importance_df = evaluate(model, test_df, scaler, numeric_cols, output_dir)
    
    return test_metrics, importance_df

RUN_EVALUATE_ONLY = True  # Set to True to run just evaluate

if __name__ == "__main__":
    if RUN_EVALUATE_ONLY:
        # Only run evaluate
        from trading_advisor.models.logistic_model import LogisticTradingModel
        output_dir = Path("model_outputs/logistic2/method2")
        data_dir = Path("data/ml_datasets/test_run")
        _, _, test_df = load_dataset(data_dir)
        # Load model, scaler, and features
        model = LogisticTradingModel.load(output_dir / "logistic_model.pkl")
        scaler = joblib.load(output_dir / "scaler.pkl")
        numeric_cols = joblib.load(output_dir / "features.pkl")
        metrics, importance_df = evaluate(
            model=model,
            test_df=test_df,
            scaler=scaler,
            numeric_cols=numeric_cols,
            output_dir=output_dir
        )
        logger.info("\nResults from evaluate-only run:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
    else:
        main() 