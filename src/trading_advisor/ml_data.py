"""ML dataset preparation functionality."""

import logging
from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from trading_advisor.utils import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

def generate_swing_trade_labels(
    df: pd.DataFrame, 
    lookback_days: int = 3,
    forward_days: int = 5,
    min_return: float = 0.02,
    max_drawdown: float = 0.01
) -> pd.DataFrame:
    """Generate swing trade labels based on short-term price action and risk management.
    
    Args:
        df: DataFrame with price data
        lookback_days: Number of days to look back for trend confirmation
        forward_days: Number of days to look forward for profit target
        min_return: Minimum return required for a trade (e.g., 0.02 for 2%)
        max_drawdown: Maximum allowed drawdown (e.g., 0.01 for 1%)
        
    Returns:
        DataFrame with generated labels
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()
    
    # Calculate returns over lookback period
    df['returns_lookback'] = df.groupby('ticker')['adj_close'].pct_change(lookback_days)
    
    # Calculate forward returns
    df['forward_returns'] = df.groupby('ticker')['adj_close'].shift(-forward_days) / df['adj_close'] - 1
    
    # Calculate drawdown using a different approach
    df['rolling_max'] = df.groupby('ticker')['adj_close'].transform(
        lambda x: x.rolling(forward_days, min_periods=1).max()
    )
    df['drawdown'] = (df['adj_close'] - df['rolling_max']) / df['rolling_max']
    
    # Generate labels
    labels = pd.DataFrame(index=df.index)
    labels['label'] = 0  # Default to no trade
    
    # Long setup conditions
    long_condition = (
        (df['returns_lookback'] > min_return) &  # Confirmed uptrend
        (df['forward_returns'] > min_return) &   # Expected profit
        (df['drawdown'] > -max_drawdown)         # Risk management
    )
    
    # Short setup conditions
    short_condition = (
        (df['returns_lookback'] < -min_return) &  # Confirmed downtrend
        (df['forward_returns'] < -min_return) &   # Expected profit
        (df['drawdown'] > -max_drawdown)          # Risk management
    )
    
    # Assign labels
    labels.loc[long_condition, 'label'] = 1
    labels.loc[short_condition, 'label'] = -1
    
    return labels

def generate_labels(df: pd.DataFrame, label_type: str = 'swing_trade', **kwargs) -> pd.DataFrame:
    """Generate labels based on the specified type.
    
    Args:
        df: DataFrame with price data
        label_type: Type of label to generate (only 'swing_trade' supported)
        **kwargs: Additional parameters for label generation
        
    Returns:
        DataFrame with only the label column
    """
    if label_type != 'swing_trade':
        raise ValueError("Only 'swing_trade' label type is supported")
    
    return generate_swing_trade_labels(df, **kwargs)

def prepare_ml_datasets(
    input_dir: Path,
    output_dir: Path,
    label_types: Optional[List[str]] = None
) -> None:
    """Prepare ML datasets by adding labels and splitting into train/val/test sets.
    
    Args:
        input_dir: Directory containing processed data
        output_dir: Directory to save ML datasets
        label_types: List of label types to generate (only 'swing_trade' supported)
    """
    # Default to swing_trade if no label types provided
    if label_types is None:
        label_types = ['swing_trade']
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load processed data
    logger.info("Loading processed data...")
    df = pd.read_parquet(input_dir / 'processed_data.parquet')
    
    # Remove rows with NaN values
    df = df.dropna()
    
    # Split into train/val/test sets
    logger.info("Splitting data into train/val/test sets...")
    # First split into train+val and test
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=False  # Don't shuffle to maintain time series order
    )
    
    # Then split train+val into train and val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.2,
        random_state=42,
        shuffle=False
    )
    
    # Save feature datasets
    logger.info("Saving feature datasets...")
    train_df.to_parquet(output_dir / 'train.parquet')
    val_df.to_parquet(output_dir / 'val.parquet')
    test_df.to_parquet(output_dir / 'test.parquet')
    
    # Generate and save labels
    for label_type in label_types:
        logger.info(f"\nGenerating {label_type} labels...")
        
        # Generate labels for each dataset
        train_labels = generate_labels(train_df, label_type)
        val_labels = generate_labels(val_df, label_type)
        test_labels = generate_labels(test_df, label_type)
        
        # Save labels
        label_dir = output_dir / label_type
        label_dir.mkdir(exist_ok=True)
        
        train_labels.to_parquet(label_dir / 'train_labels.parquet')
        val_labels.to_parquet(label_dir / 'val_labels.parquet')
        test_labels.to_parquet(label_dir / 'test_labels.parquet')
        
        # Log label distribution
        logger.info(f"\nLabel distribution for {label_type}:")
        for name, labels in [('Train', train_labels), ('Validation', val_labels), ('Test', test_labels)]:
            dist = labels['label'].value_counts(normalize=True)
            logger.info(f"{name} set:")
            for label, pct in dist.items():
                label_name = 'Long' if label == 1 else ('Short' if label == -1 else 'No Trade')
                logger.info(f"  {label_name}: {pct:.2%}")
    
    # Log dataset sizes
    logger.info(f"\nDataset sizes:")
    logger.info(f"Train set size: {len(train_df)}")
    logger.info(f"Validation set size: {len(val_df)}")
    logger.info(f"Test set size: {len(test_df)}") 