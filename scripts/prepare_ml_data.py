"""Script to prepare ML datasets by adding labels."""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from trading_advisor.utils import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

def generate_labels(df: pd.DataFrame, label_type: str = 'short_term_profit', **kwargs) -> pd.DataFrame:
    """Generate labels based on different trading strategies.
    
    Args:
        df: DataFrame with price data
        label_type: Type of label to generate
        **kwargs: Additional parameters for label generation
        
    Returns:
        DataFrame with only the label column
    """
    labels = pd.DataFrame(index=df.index)
    
    if label_type == 'short_term_profit':
        # Look for 1% gain within 5 days
        forward_returns = df.groupby('ticker')['adj_close'].pct_change(5).shift(-5)
        # Create binary labels (1 if return > 1%, 0 otherwise)
        labels['label'] = (forward_returns > 0.01).astype(int)
        
    elif label_type == 'risk_adjusted':
        # Calculate 5-day returns and volatility
        returns = df.groupby('ticker')['adj_close'].pct_change()
        forward_returns = returns.shift(-5)
        volatility = returns.rolling(20).std()
        # Create labels based on Sharpe ratio (return/volatility)
        sharpe = forward_returns / volatility
        labels['label'] = (sharpe > 0.5).astype(int)  # Positive Sharpe ratio
        
    elif label_type == 'relative_strength':
        # Compare stock returns to sector performance
        stock_returns = df.groupby('ticker')['adj_close'].pct_change(5).shift(-5)
        sector_returns = df.groupby('ticker')['sector_performance_returns_5d'].first()
        # Label as 1 if stock outperforms sector
        labels['label'] = (stock_returns > sector_returns).astype(int)
        
    elif label_type == 'multi_class':
        # Create multi-class labels based on return thresholds
        forward_returns = df.groupby('ticker')['adj_close'].pct_change(5).shift(-5)
        labels['label'] = pd.cut(
            forward_returns,
            bins=[-np.inf, -0.01, 0.005, 0.02, np.inf],
            labels=[0, 1, 2, 3]  # 0: Strong Sell, 1: Sell, 2: Buy, 3: Strong Buy
        ).astype(int)
        
    elif label_type == 'momentum_reversal':
        # Look for momentum continuation or reversal
        returns_5d = df.groupby('ticker')['adj_close'].pct_change(5)
        returns_1d = df.groupby('ticker')['adj_close'].pct_change(1)
        # Label as 1 if momentum continues (both returns in same direction)
        labels['label'] = ((returns_5d > 0) & (returns_1d > 0) | 
                          (returns_5d < 0) & (returns_1d < 0)).astype(int)
        
    else:
        raise ValueError(f"Unknown label type: {label_type}")
    
    return labels

def main():
    """Main function to prepare ML datasets."""
    # Input and output directories
    input_dir = Path('data/processed')
    output_dir = Path('data/ml_datasets')
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
    train_df.to_parquet(output_dir / 'train_features.parquet')
    val_df.to_parquet(output_dir / 'val_features.parquet')
    test_df.to_parquet(output_dir / 'test_features.parquet')
    
    # Generate and save different types of labels
    label_types = {
        'short_term_profit': {},  # 1% gain in 5 days
        'risk_adjusted': {},      # Based on Sharpe ratio
        'relative_strength': {},  # Outperforming sector
        'multi_class': {},        # Multiple return thresholds
        'momentum_reversal': {}   # Momentum continuation/reversal
    }
    
    for label_type, params in label_types.items():
        logger.info(f"\nGenerating {label_type} labels...")
        
        # Generate labels for each dataset
        train_labels = generate_labels(train_df, label_type, **params)
        val_labels = generate_labels(val_df, label_type, **params)
        test_labels = generate_labels(test_df, label_type, **params)
        
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
                if label_type == 'multi_class':
                    label_names = ['Strong Sell', 'Sell', 'Buy', 'Strong Buy']
                    logger.info(f"  {label_names[label]}: {pct:.2%}")
                else:
                    logger.info(f"  Class {label}: {pct:.2%}")
    
    # Log dataset sizes
    logger.info(f"\nDataset sizes:")
    logger.info(f"Train set size: {len(train_df)}")
    logger.info(f"Validation set size: {len(val_df)}")
    logger.info(f"Test set size: {len(test_df)}")

if __name__ == '__main__':
    main() 