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
import json
import argparse
import os
from trading_advisor.dataset import DatasetGenerator

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
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

def preprocess_features(df: pd.DataFrame, scaler: StandardScaler = None, fit: bool = False, feature_order: list = None) -> Tuple[np.ndarray, StandardScaler]:
    """Preprocess features by scaling and handling missing values."""
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['label', 'ticker']]
    if feature_order is not None:
        numeric_cols = feature_order
    else:
        numeric_cols = sorted(numeric_cols)
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

    # Determine feature order ONCE and use everywhere
    feature_order = sorted([col for col in train_df.select_dtypes(include=[np.number]).columns if col not in ['label', 'ticker']])

    # Preprocess features using the same order
    X_train, scaler = preprocess_features(train_df, fit=True, feature_order=feature_order)
    X_val, _ = preprocess_features(val_df, scaler=scaler, feature_order=feature_order)

    # Save scaler and feature columns
    scaler_path = output_dir / "scaler.pkl"
    features_path = output_dir / "features.pkl"
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_order, features_path)

    # Get labels
    y_train = train_df['label'].values
    y_val = val_df['label'].values

    # Initialize and train model
    logger.info("\nTraining logistic model...")
    model = LogisticTradingModel(**model_params)

    # Create training DataFrame with preprocessed features
    train_processed = pd.DataFrame(X_train, columns=feature_order)
    train_processed['label'] = y_train
    val_processed = pd.DataFrame(X_val, columns=feature_order)
    val_processed['label'] = y_val

    model.train(train_processed, val_processed)

    # Save model and feature means
    model_path = output_dir / "logistic_model.pkl"
    feature_means_path = output_dir / "feature_means.pkl"
    model.save(model_path)
    joblib.dump(model.feature_means, feature_means_path)
    logger.info(f"\nSaved trained model to {model_path}")
    logger.info(f"Saved feature means to {feature_means_path}")

    return model, scaler, feature_order

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

    # Save test predictions
    test_predictions = test_df.copy()
    test_predictions['predicted_label'] = test_pred
    test_predictions['predicted_proba'] = test_proba
    test_predictions.to_parquet(output_dir / 'test_predictions.parquet')
    logger.info(f"Saved test predictions to {output_dir / 'test_predictions.parquet'}")

    return test_metrics, importance_df

def get_ticker_code(ticker: str) -> int:
    """Convert a ticker symbol to its numeric code.
    Args:
        ticker: Ticker symbol (e.g., 'AMD')
    Returns:
        Integer code for the ticker, or None if not found
    """
    # Load ticker mapping from dataset directory
    mapping_file = Path("data/ml_datasets/large_test/feature_mappings.json")
    if not mapping_file.exists():
        raise FileNotFoundError(f"Ticker mapping file not found: {mapping_file}")
    with open(mapping_file, 'r') as f:
        mappings = json.load(f)
    # Look up ticker in the mappings (key is 'ticker')
    ticker_dict = mappings.get('ticker', {})
    return ticker_dict.get(ticker, None)

def _add_prefix_from_filename(df, file_path, skip_cols=None):
    """Add prefix to columns based on the base filename (without .parquet)."""
    if skip_cols is None:
        skip_cols = ['Date', 'ticker']
    prefix = os.path.splitext(os.path.basename(str(file_path)))[0] + '_'
    rename_dict = {col: prefix + col for col in df.columns if col not in skip_cols}
    return df.rename(columns=rename_dict)

def _load_market_features(date):
    """Load market features for a given date."""
    market_features_dir = Path("data/market_features")
    if not market_features_dir.exists():
        logger.error(f"Market features directory not found at {market_features_dir}")
        return None
        
    market_features = {}
    
    # List of market feature files
    market_files = [
        "daily_breadth.parquet",
        "market_volatility.parquet",
        "market_sentiment.parquet",
        "gdelt_raw.parquet"
    ]
    for fname in market_files:
        fpath = market_features_dir / fname
        if fpath.exists():
            df = pd.read_parquet(fpath)
            # If Date is the index, reset it to a column
            if 'Date' not in df.columns and df.index.name == 'Date':
                df = df.reset_index()
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                row = df[df['Date'] == pd.to_datetime(date)]
                if not row.empty:
                    row = row.iloc[0:1]  # keep as DataFrame
                    row = _add_prefix_from_filename(row, fpath)
                    # Remove 'Date' column except for the first file
                    if market_features:
                        row = row.drop(columns=['Date'], errors='ignore')
                    market_features[fname] = row
    # Combine all market features
    if market_features:
        market_df = pd.concat(market_features.values(), axis=1)
        market_df['Date'] = pd.to_datetime(date)
        return market_df
    return None

def _load_sector_features(ticker, date):
    """Load sector features for a given ticker and date."""
    sector_map_file = Path("data/market_features/metadata/sector_mapping.parquet")
    if not sector_map_file.exists():
        return None
        
    sector_map = pd.read_parquet(sector_map_file)
    if 'ticker' in sector_map.columns and 'sector' in sector_map.columns:
        sector_row = sector_map[sector_map['ticker'] == ticker]
        if sector_row.empty:
            return None
        sector = sector_row['sector'].iloc[0]
    else:
        return None
        
    sector_file = Path(f"data/market_features/sectors/{sector}.parquet")
    if not sector_file.exists():
        return None
        
    sector_df = pd.read_parquet(sector_file)
    # If Date is the index, reset it to a column
    if 'Date' not in sector_df.columns and sector_df.index.name == 'Date':
        sector_df = sector_df.reset_index()
    if 'Date' in sector_df.columns:
        sector_df['Date'] = pd.to_datetime(sector_df['Date'])
        sector_df = sector_df[sector_df['Date'] == pd.to_datetime(date)]
        if not sector_df.empty:
            sector_df = sector_df.iloc[0:1]
            sector_df = _add_prefix_from_filename(sector_df, sector_file)
            return sector_df.iloc[0]
    return None

def load_features_for_date(date, ticker=None, verbose=False):
    """Load all features for a given date (and optionally ticker symbol as string).
    Returns a DataFrame with all features for that date (optionally filtered by ticker),
    and ensures all expected features are present in the correct order.
    """
    logger.info(f"[DEBUG] load_features_for_date called with date={date}, ticker={ticker}")
    # First load the expected features to ensure correct order
    features_path = Path("model_outputs/logistic2/large_test/features.pkl")
    if not features_path.exists():
        logger.error(f"Features file not found at {features_path}")
        return None
    expected_features = joblib.load(features_path)
    
    # Load feature means for imputation
    feature_means_path = Path("model_outputs/logistic2/large_test/feature_means.pkl")
    if not feature_means_path.exists():
        logger.error(f"Feature means file not found at {feature_means_path}")
        return None
    feature_means = joblib.load(feature_means_path)
    
    # Load ticker features
    ticker_features_dir = Path("data/ticker_features")
    if not ticker_features_dir.exists():
        logger.error(f"Ticker features directory not found at {ticker_features_dir}")
        return None
        
    ticker_dfs = []
    for file in ticker_features_dir.glob("*_features.parquet"):
        df = pd.read_parquet(file)
        # If Date is the index, reset it to a column
        if 'Date' not in df.columns and df.index.name == 'Date':
            df = df.reset_index()
        # Extract ticker symbol from filename
        ticker_symb = file.stem.split('_')[0]
        if ticker is None or ticker_symb == ticker:
            # Convert date column to datetime if it's not already
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df[df['Date'] == pd.to_datetime(date)]
                if not df.empty:
                    df['ticker'] = ticker_symb  # keep as string for now
                    df = _add_prefix_from_filename(df, file)
                    ticker_dfs.append(df)
    if not ticker_dfs:
        logger.error(f"[DEBUG] No ticker features found for ticker={ticker} on {date}")
        return None
    # Combine all ticker features
    ticker_df = pd.concat(ticker_dfs, ignore_index=True)
    # Convert ticker symbol to numeric code for model input
    ticker_df['ticker'] = ticker_df['ticker'].apply(lambda x: get_ticker_code(x) if isinstance(x, str) else x)
    # Load market features
    market_features = _load_market_features(date)
    if market_features is not None:
        # Ensure date column is datetime
        market_features['Date'] = pd.to_datetime(market_features['Date'])
        ticker_df = ticker_df.merge(market_features, on='Date', how='left')
    # Load sector features for each ticker
    for ticker_code in ticker_df['ticker'].unique():
        sector_features = _load_sector_features(ticker_code, date)
        if sector_features is not None:
            # Ensure date column is datetime
            sector_features['Date'] = pd.to_datetime(sector_features['Date'])
            ticker_mask = ticker_df['ticker'] == ticker_code
            for col in sector_features.index:
                if col != 'Date':
                    ticker_df.loc[ticker_mask, col] = sector_features[col]
    # Ensure all expected features are present and in the correct order
    missing_features = [feat for feat in expected_features if feat not in ticker_df.columns]
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
        for feat in missing_features:
            # Use feature mean for imputation if available, otherwise use 0
            ticker_df[feat] = feature_means.get(feat, 0.0)
    # Reorder columns to match expected features
    ticker_df = ticker_df[expected_features + [col for col in ticker_df.columns if col not in expected_features]]
    return ticker_df

def predict_single_row(model, date, ticker, verbose=False):
    """Make a prediction for a single row of data."""
    try:
        # Load and prepare features
        features_df = load_features_for_date(date, ticker, verbose)
        if features_df is None or features_df.empty:
            logger.error(f"No features found for {ticker} on {date}")
            return None
        
        # Load expected features from the features.pkl file
        features_path = Path("model_outputs/logistic2/large_test/features.pkl")
        if not features_path.exists():
            logger.error(f"Features file not found at {features_path}")
            return None
        expected_features = joblib.load(features_path)
        
        # Ensure all expected features are present
        missing_features = [feat for feat in expected_features if feat not in features_df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            for feat in missing_features:
                features_df[feat] = np.nan
        
        # Reorder columns to match expected features
        features_df = features_df[expected_features]
        
        # Check for NaN values before scaling
        if features_df.isna().any().any():
            # Load feature means for imputation
            feature_means_path = Path("model_outputs/logistic2/large_test/feature_means.pkl")
            if feature_means_path.exists():
                feature_means = joblib.load(feature_means_path)
                for col in features_df.columns:
                    if features_df[col].isna().any():
                        features_df[col] = features_df[col].fillna(feature_means.get(col, 0.0))
            else:
                logger.warning("Feature means file not found, using 0.0 for imputation")
                features_df = features_df.fillna(0.0)
        
        # Scale the features
        X_scaled = model.scaler.transform(features_df)
        
        # Check for NaN values after scaling
        if np.isnan(X_scaled).any():
            logger.warning("NaN values found after scaling, replacing with 0.0")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        
        # Create DataFrame for prediction
        features_df_scaled = pd.DataFrame(X_scaled, columns=expected_features)
        
        # Get predictions using model's predict method
        results = model.predict(features_df_scaled)
        if results is None:
            logger.error("Model prediction returned None")
            return None
            
        if 'predictions' not in results or 'probabilities' not in results:
            logger.error(f"Invalid prediction results format: {results}")
            return None
            
        # Get the raw probabilities from the model
        try:
            raw_probs = model.model.predict_proba(X_scaled)
            if raw_probs is None or len(raw_probs) == 0:
                logger.error("Model predict_proba returned None or empty array")
                return None
                
            # Create a DataFrame with all features and predictions
            prediction_df = features_df.copy()
            prediction_df['predicted_label'] = results['predictions'][0]
            prediction_df['predicted_proba'] = raw_probs[0, 1]  # Probability of class 1 (buy)
            prediction_df['Date'] = pd.to_datetime(date)
            prediction_df['ticker'] = ticker
            
            # Add label column (will be NaN for predictions)
            prediction_df['label'] = np.nan
            
            return prediction_df
            
        except Exception as e:
            logger.error(f"Error getting probabilities: {str(e)}")
            return None
        
    except Exception as e:
        logger.error(f"Error getting prediction: {str(e)}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return None

def load_trained_model(dataset_name: str):
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

def get_prediction_for_date_ticker(dataset_name: str, date: str, ticker: str) -> dict:
    """Get prediction for a specific date and ticker.
    Args:
        dataset_name: Name of the dataset directory
        date: Date to predict for (YYYY-MM-DD)
        ticker: Ticker symbol to predict for
    Returns:
        Dictionary containing prediction and probability
    """
    try:
        # Convert ticker symbol to numeric code (for later use)
        ticker_code = get_ticker_code(ticker)
        if ticker_code is None:
            logger.error(f"Could not find numeric code for ticker {ticker}")
            return None
            
        # Load model and artifacts
        model, scaler, feature_columns = load_trained_model(dataset_name)
        if model is None or scaler is None:
            logger.error("Failed to load model or scaler")
            return None
        
        # Convert date string to datetime
        target_date = pd.to_datetime(date)
        
        # Use generate-dataset to prepare features for the given date and ticker
        generator = DatasetGenerator(
            market_features_dir="data/market_features",
            ticker_features_dir="data/ticker_features",
            output_dir="data/ml_datasets"
        )
        features_df = generator.prepare_features(ticker, date, include_sector=True)
        if features_df.empty:
            logger.error(f"No features found for {ticker} on {date}")
            return None
        
        # Ensure all expected features are present
        missing_features = [feat for feat in feature_columns if feat not in features_df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            for feat in missing_features:
                features_df[feat] = np.nan
        
        # Reorder columns to match expected features
        features_df = features_df[feature_columns]
        
        # Check for NaN values before scaling
        if features_df.isna().any().any():
            # Load feature means for imputation
            feature_means_path = Path("model_outputs/logistic2") / dataset_name / "feature_means.pkl"
            if feature_means_path.exists():
                feature_means = joblib.load(feature_means_path)
                for col in features_df.columns:
                    if features_df[col].isna().any():
                        features_df[col] = features_df[col].fillna(feature_means.get(col, 0.0))
            else:
                logger.warning("Feature means file not found, using 0.0 for imputation")
                features_df = features_df.fillna(0.0)
        
        # Scale the features
        X_scaled = scaler.transform(features_df)
        
        # Check for NaN values after scaling
        if np.isnan(X_scaled).any():
            logger.warning("NaN values found after scaling, replacing with 0.0")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        
        # Create DataFrame for prediction
        features_df_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
        
        # Get predictions using model's predict method
        results = model.predict(features_df_scaled)
        if results is None:
            logger.error("Model prediction returned None")
            return None
            
        if 'predictions' not in results or 'probabilities' not in results:
            logger.error(f"Invalid prediction results format: {results}")
            return None
            
        # Get the raw probabilities from the model
        try:
            raw_probs = model.model.predict_proba(X_scaled)
            if raw_probs is None or len(raw_probs) == 0:
                logger.error("Model predict_proba returned None or empty array")
                return None
                
            # Create a DataFrame with all features and predictions
            prediction_df = features_df.copy()
            prediction_df['predicted_label'] = results['predictions'][0]
            prediction_df['predicted_proba'] = raw_probs[0, 1]  # Probability of class 1 (buy)
            prediction_df['Date'] = pd.to_datetime(date)
            prediction_df['ticker'] = ticker
            
            # Add label column (will be NaN for predictions)
            prediction_df['label'] = np.nan
            
            return prediction_df
            
        except Exception as e:
            logger.error(f"Error getting probabilities: {str(e)}")
            return None
        
    except Exception as e:
        logger.error(f"Error getting prediction: {str(e)}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return None

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a logistic regression model")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--predict", action="store_true", help="Make a prediction for a single date and ticker")
    parser.add_argument("--date", type=str, help="Date for prediction")
    parser.add_argument("--ticker", type=str, help="Ticker symbol for prediction")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate an existing model")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()

def evaluate_existing_model(dataset_name: str):
    """Evaluate an existing trained model on the test dataset."""
    try:
        # Load dataset
        data_dir = Path("data/ml_datasets") / dataset_name
        train_df, val_df, test_df = load_dataset(data_dir)

        # Load model and artifacts
        model, scaler, feature_columns = load_trained_model(dataset_name)
        if model is None or scaler is None:
            logger.error("Failed to load model or scaler")
            return

        # Create output directory
        output_dir = Path("model_outputs/logistic2") / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Evaluate model
        logger.info(f"\n=== Evaluating existing model on {dataset_name} dataset ===")
        metrics, importance_df = evaluate(
            model=model,
            test_df=test_df,
            scaler=scaler,
            numeric_cols=feature_columns,
            output_dir=output_dir
        )

        # Save metrics to a JSON file
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info("\nFinal Results:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")

def main():
    # Parse command line arguments
    args = parse_args()

    # If --predict is set, run prediction and exit
    if args.predict:
        if not args.date or not args.ticker:
            logger.error("Both --date and --ticker are required for prediction")
            return
        try:
            result = get_prediction_for_date_ticker(args.dataset, args.date, args.ticker)
            print(f"\nPrediction for {args.ticker} on {args.date}:")
            print(f"Prediction: {result['prediction']}")
            print(f"Buy Probability: {result['buy_probability']:.4f}")
            print(f"Sell Probability: {result['sell_probability']:.4f}")
        except Exception as e:
            logger.error(f"Error getting prediction: {str(e)}")
        return

    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
        # Keep important messages
        logging.getLogger(__name__).setLevel(logging.INFO)

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
    model, scaler, feature_order = train(
        train_df=train_df,
        val_df=val_df,
        output_dir=output_dir,
        model_params=model_params
    )

    metrics, importance_df = evaluate(
        model=model,
        test_df=test_df,
        scaler=scaler,
        numeric_cols=feature_order,
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
