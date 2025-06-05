import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from trading_advisor.models.logistic_model import LogisticTradingModel
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
import random
import joblib
import hashlib
import json
import argparse

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
    
    # Drop rows with NaN labels
    df_clean = df.dropna(subset=['label'])
    
    # Handle missing values in features
    df_clean = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    
    # Scale features
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(df_clean)
        # Store feature names in scaler
        scaler.feature_names_in_ = numeric_cols
    else:
        # Ensure we're using the same feature order as during training
        df_clean = df_clean[scaler.feature_names_in_]
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

    # Save per-row predictions for comparison
    test_processed['predicted_label'] = test_results['predictions']
    test_processed['predicted_proba'] = test_results['probabilities']
    test_processed.to_parquet(output_dir / "test_predictions.parquet")

    return test_metrics, importance_df

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train logistic regression model')
    parser.add_argument(
        '--dataset',
        type=str,
        default='test_run',
        help='Name of the dataset directory under data/ml_datasets/'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate a previously trained model instead of training a new one'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed logging information'
    )
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Run prediction for a specific date and ticker'
    )
    parser.add_argument(
        '--date',
        type=str,
        help='Date to predict for (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--ticker',
        type=str,
        help='Ticker symbol to predict for'
    )
    return parser.parse_args()

def evaluate_existing_model(dataset_name: str):
    """Evaluate a previously trained model."""
    # Set up paths
    data_dir = Path("data/ml_datasets") / dataset_name
    model_dir = Path("model_outputs/logistic2") / dataset_name
    
    # Check if model exists
    model_path = model_dir / "logistic_model.pkl"
    scaler_path = model_dir / "scaler.pkl"
    features_path = model_dir / "features.pkl"
    
    if not all(p.exists() for p in [model_path, scaler_path, features_path]):
        logger.error(f"Model files not found in {model_dir}")
        return
    
    # Load model and artifacts
    model = LogisticTradingModel.load(model_path)
    scaler = joblib.load(scaler_path)
    numeric_cols = joblib.load(features_path)
    
    # Load dataset
    train_df, val_df, test_df = load_dataset(data_dir)
    
    # Evaluate on test set
    logger.info(f"\n=== Evaluating model on {dataset_name} dataset ===")
    metrics, importance_df = evaluate(
        model=model,
        test_df=test_df,
        scaler=scaler,
        numeric_cols=numeric_cols,
        output_dir=model_dir
    )
    
    # Save metrics to a JSON file
    metrics_path = model_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("\nFinal Results:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")

def load_trained_model(dataset_name: str) -> Tuple[LogisticTradingModel, StandardScaler, List[str]]:
    """Load a trained model and its artifacts.
    
    Args:
        dataset_name: Name of the dataset directory under data/ml_datasets/
        
    Returns:
        Tuple of (model, scaler, feature_columns)
    """
    model_dir = Path("model_outputs/logistic2") / dataset_name
    
    # Check if model exists
    model_path = model_dir / "logistic_model.pkl"
    scaler_path = model_dir / "scaler.pkl"
    features_path = model_dir / "features.pkl"
    
    if not all(p.exists() for p in [model_path, scaler_path, features_path]):
        raise FileNotFoundError(f"Model files not found in {model_dir}")
    
    # Load model and artifacts
    model = LogisticTradingModel.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(features_path)
    
    return model, scaler, feature_columns

def predict_single_row(
    model: LogisticTradingModel,
    scaler: StandardScaler,
    feature_columns: List[str],
    row: pd.DataFrame
) -> Dict[str, Any]:
    """Run inference on a single row of data.
    
    Args:
        model: Trained model
        scaler: Feature scaler
        feature_columns: List of feature column names in the order they were used during training
        row: DataFrame containing a single row of features
        
    Returns:
        Dictionary containing prediction and probability
    """
    # Ensure we have all required features
    missing_cols = set(feature_columns) - set(row.columns)
    if missing_cols:
        raise ValueError(f"Missing required features: {missing_cols}")
    
    # Select and order features exactly as they were during training
    X = row[feature_columns].copy()
    
    # Use scaler's feature_names_in_ to ensure correct order
    if hasattr(scaler, 'feature_names_in_'):
        X = X[scaler.feature_names_in_]
    
    # Log feature order
    logger.info("Feature order during prediction:")
    logger.info(f"Expected order: {feature_columns}")
    logger.info(f"Actual order: {X.columns.tolist()}")
    if hasattr(scaler, 'feature_names_in_'):
        logger.info(f"Scaler feature names: {scaler.feature_names_in_}")
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Get prediction
    results = model.predict(pd.DataFrame(X_scaled, columns=X.columns))
    
    return {
        'prediction': results['predictions'][0],
        'probability': results['probabilities'][0]
    }

def _load_market_features(date: pd.Timestamp) -> pd.DataFrame:
    """Load market features for a specific date.
    
    Args:
        date: Date to load features for
        
    Returns:
        DataFrame containing market features
    """
    market_features = {}
    market_features_dir = Path("data/market_features")
    
    for path in market_features_dir.glob("*.parquet"):
        if path.is_dir() or path.name.startswith("metadata"):
            continue
        feature_type = path.stem
        df = pd.read_parquet(path)
        
        # Ensure date is the index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df = df.set_index('Date')
            elif 'date' in df.columns:
                df = df.set_index('date')
            else:
                continue
        df.index = pd.to_datetime(df.index)
        
        # Filter for target date or nearest previous date
        if date in df.index:
            row = df.loc[date]
        else:
            # Find the nearest previous date
            mask = df.index <= date
            if not mask.any():
                continue
            nearest_date = df.index[mask].max()
            row = df.loc[nearest_date]
        
        row = row.to_frame().T
        row = row.add_suffix(f'_{feature_type}')
        market_features[feature_type] = row
    
    if market_features:
        return pd.concat(market_features.values(), axis=1)
    else:
        return pd.DataFrame()

def _load_sector_features(ticker: str, date: pd.Timestamp) -> pd.DataFrame:
    """Load sector features for a specific ticker and date.
    
    Args:
        ticker: Ticker symbol
        date: Date to load features for
        
    Returns:
        DataFrame containing sector features
    """
    # Load sector mapping
    sector_mapping_file = Path("data/market_features/metadata/sector_mapping.parquet")
    if not sector_mapping_file.exists():
        raise FileNotFoundError(f"Sector mapping file not found: {sector_mapping_file}")
    
    sector_mapping = pd.read_parquet(sector_mapping_file)
    
    # Find ticker's sector
    ticker_row = sector_mapping[sector_mapping['ticker'] == ticker]
    if ticker_row.empty:
        raise ValueError(f"No sector found for ticker {ticker}")
    
    ticker_sector = ticker_row['sector'].iloc[0]
    
    # Load sector features
    sector_file = Path("data/market_features/sectors") / f"{ticker_sector}.parquet"
    if not sector_file.exists():
        raise FileNotFoundError(f"Sector features not found: {sector_file}")
    
    sector_df = pd.read_parquet(sector_file)
    
    # Ensure date is the index
    if not isinstance(sector_df.index, pd.DatetimeIndex):
        if 'Date' in sector_df.columns:
            sector_df = sector_df.set_index('Date')
        elif 'date' in sector_df.columns:
            sector_df = sector_df.set_index('date')
        else:
            raise ValueError("No date column found in sector features")
    
    sector_df.index = pd.to_datetime(sector_df.index)
    
    # Filter for target date or nearest previous date
    if date in sector_df.index:
        row = sector_df.loc[date]
    else:
        # Find the nearest previous date
        mask = sector_df.index <= date
        if not mask.any():
            raise ValueError(f"No sector features found for {ticker_sector} before {date}")
        nearest_date = sector_df.index[mask].max()
        row = sector_df.loc[nearest_date]
    
    row = row.to_frame().T
    row = row.add_suffix('_sector')
    return row

def get_prediction_for_date_ticker(dataset_name: str, date: str, ticker: str) -> Dict[str, Any]:
    """Get prediction for a specific date and ticker.
    
    Args:
        dataset_name: Name of the dataset directory
        date: Date to predict for (YYYY-MM-DD)
        ticker: Ticker symbol to predict for
        
    Returns:
        Dictionary containing prediction and probability
    """
    # Load model and artifacts
    model, scaler, feature_columns = load_trained_model(dataset_name)
    
    # Load ticker features
    features_dir = Path("data/ticker_features")
    feature_file = features_dir / f"{ticker}_features.parquet"
    
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    
    # Load and prepare all features
    ticker_df = pd.read_parquet(feature_file)
    
    # Ensure date is the index
    if not isinstance(ticker_df.index, pd.DatetimeIndex):
        if 'Date' in ticker_df.columns:
            ticker_df = ticker_df.set_index('Date')
        elif 'date' in ticker_df.columns:
            ticker_df = ticker_df.set_index('date')
        else:
            raise ValueError("No date column found in ticker features")
    
    ticker_df.index = pd.to_datetime(ticker_df.index)
    
    # Convert date string to datetime
    target_date = pd.to_datetime(date)
    
    # Filter for target date or nearest previous date
    if target_date in ticker_df.index:
        ticker_row = ticker_df.loc[target_date]
    else:
        # Find the nearest previous date
        mask = ticker_df.index <= target_date
        if not mask.any():
            raise ValueError(f"No ticker features found for {ticker} before {date}")
        nearest_date = ticker_df.index[mask].max()
        ticker_row = ticker_df.loc[nearest_date]
    
    ticker_row = ticker_row.to_frame().T
    
    # Get market features
    market_df = _load_market_features(target_date)
    
    # Get sector features
    sector_df = _load_sector_features(ticker, target_date)
    
    # Combine all features
    features = pd.concat([ticker_row, market_df, sector_df], axis=1)
    
    # Ensure we have all required features
    missing_cols = set(feature_columns) - set(features.columns)
    if missing_cols:
        raise ValueError(f"Missing required features: {missing_cols}")
    
    # Get prediction
    result = predict_single_row(model, scaler, feature_columns, features)
    
    # Add metadata
    result.update({
        'ticker': ticker,
        'date': date,
        'dataset': dataset_name
    })
    
    return result

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
        # Keep important messages
        logging.getLogger(__name__).setLevel(logging.INFO)
    
    if args.predict:
        if not args.date or not args.ticker:
            logger.error("Both --date and --ticker are required for prediction")
            return
        
        try:
            result = get_prediction_for_date_ticker(args.dataset, args.date, args.ticker)
            print(f"\nPrediction for {result['ticker']} on {result['date']}:")
            print(f"Prediction: {result['prediction']}")
            print(f"Probability: {result['probability']:.4f}")
        except Exception as e:
            logger.error(f"Error getting prediction: {str(e)}")
        return
    
    if args.evaluate:
        evaluate_existing_model(args.dataset)
        return
    
    # Set random seeds
    set_random_seeds(42)
    
    # Get dataset name from path
    data_dir = Path("data/ml_datasets") / args.dataset
    dataset_name = data_dir.name
    
    # Create output directory
    output_dir = Path("model_outputs/logistic2") / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model parameters
    model_params = {
        'C': 0.1,  # Stronger regularization
        'class_weight': 'balanced',
        'max_iter': 1000,
        'random_state': 42,
        'solver': 'liblinear'  # Better for small datasets
    }
    
    # Load dataset
    train_df, val_df, test_df = load_dataset(data_dir)
    
    # Train and evaluate
    logger.info(f"\n=== Training and Evaluating on {dataset_name} dataset ===")
    model, scaler, numeric_cols = train(
        train_df=train_df,
        val_df=val_df,
        output_dir=output_dir,
        model_params=model_params
    )
    
    metrics, importance_df = evaluate(
        model=model,
        test_df=test_df,
        scaler=scaler,
        numeric_cols=numeric_cols,
        output_dir=output_dir
    )
    
    # Save metrics to a JSON file
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("\nFinal Results:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 