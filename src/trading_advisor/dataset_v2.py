"""Dataset generation functionality for machine learning models.

This module provides a clean implementation for generating machine learning datasets
from market and ticker features. It focuses on:
1. Loading and combining features
2. Proper handling of sector data
3. Train/val/test splitting
4. Feature normalization
5. Data quality validation
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import typer
import os

from trading_advisor.features import load_features
from trading_advisor.sector_mapping import load_sector_mapping
from trading_advisor.data import download_stock_data
from trading_advisor.preprocessing import FeaturePreprocessor

logger = logging.getLogger(__name__)

class DatasetGeneratorV2:
    """Generator for machine learning datasets with improved data handling."""
    
    def __init__(
        self,
        market_features_dir: str = "data/market_features",
        ticker_features_dir: str = "data/ticker_features",
        output_dir: str = "data/ml_datasets",
        train_months: int = 6,
        val_months: int = 2,
        min_samples_per_ticker: int = 30
    ):
        """Initialize the dataset generator.
        
        Args:
            market_features_dir: Directory containing market feature files
            ticker_features_dir: Directory containing ticker feature files
            output_dir: Directory to save generated datasets
            train_months: Number of months to use for training
            val_months: Number of months to use for validation
            min_samples_per_ticker: Minimum number of trading days required per ticker in each split
        """
        self.market_features_dir = Path(market_features_dir)
        self.ticker_features_dir = Path(ticker_features_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_months = train_months
        self.val_months = val_months
        self.min_samples_per_ticker = min_samples_per_ticker
        
        # Load sector mapping
        self.sector_mapping = load_sector_mapping(str(self.market_features_dir))
        
        # Initialize preprocessor
        self.preprocessor = FeaturePreprocessor(
            market_features_dir=market_features_dir,
            ticker_features_dir=ticker_features_dir
        )
        
    def generate_dataset(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        output_dir: Path,
        validate: bool = True,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> None:
        """Generate a dataset for the given tickers and date range.
        
        Args:
            tickers: List of ticker symbols to include
            start_date: Start date for data collection (YYYY-MM-DD)
            end_date: End date for data collection (YYYY-MM-DD)
            output_dir: Directory to save output files
            validate: Whether to perform validation checks (default: True)
            progress_callback: Optional callback function to update progress
        """
        logger.info(f"Generating dataset for {len(tickers)} tickers from {start_date} to {end_date}")
        
        # Convert dates to pandas Timestamps
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        
        # Load all sector data once
        sector_data = self._load_sector_data()
        
        # Process each ticker
        all_data = []
        for i, ticker in enumerate(tickers):
            # Load ticker features
            ticker_features = self._load_ticker_features(ticker, start_date, end_date)
            if ticker_features.empty:
                continue
                
            # Get sector data for this ticker
            sector = self.sector_mapping.get(ticker)
            if sector:
                sector_name = sector.lower().replace(' ', '_')
                sector_df = sector_data.get(sector_name)
                if sector_df is not None:
                    ticker_features = self._add_sector_features(ticker_features, sector_df)
            
            all_data.append(ticker_features)
            
            # Update progress if callback provided
            if progress_callback:
                progress_callback(i + 1)
        
        if not all_data:
            raise ValueError("No data generated for any ticker")
            
        # Combine all data
        df = pd.concat(all_data, axis=0)
        
        # Split into train/val/test
        train_df, val_df, test_df = self._split_dataset(df)
        
        # Normalize features
        train_df, val_df, test_df = self.preprocessor.fit_transform(train_df, val_df, test_df)
        
        # Save datasets
        output_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(output_dir / 'train.parquet')
        val_df.to_parquet(output_dir / 'val.parquet')
        test_df.to_parquet(output_dir / 'test.parquet')
        
        # Generate README
        self._generate_readme(output_dir, train_df, val_df, test_df, validate)
        
        logger.info(f"Dataset generated successfully at {output_dir}")
        
        # Run validation if requested
        if validate:
            logger.info("Running dataset validation...")
            self._validate_datasets(train_df, val_df, test_df)
            logger.info("Dataset validation passed")
        
    def prepare_inference_data(
        self,
        ticker: str,
        date: pd.Timestamp,
        features: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """Prepare a single row of data for inference.
        
        Args:
            ticker: Ticker symbol
            date: Date for inference
            features: Optional pre-computed features
            
        Returns:
            DataFrame with a single row of normalized features
        """
        return self.preprocessor.prepare_single_row(ticker, date, features)
        
    def _load_sector_data(self) -> Dict[str, pd.DataFrame]:
        """Load all sector data files.
        
        Returns:
            Dictionary mapping sector names to their data
        """
        sector_data = {}
        for sector in self.sector_mapping.values():
            sector_name = sector.lower().replace(' ', '_')
            sector_file = self.market_features_dir / 'sectors' / f"{sector_name}.parquet"
            if sector_file.exists():
                try:
                    sector_df = pd.read_parquet(sector_file)
                    if not sector_df.empty:
                        sector_data[sector_name] = sector_df
                except Exception as e:
                    logger.error(f"Error loading sector data for {sector}: {str(e)}")
        return sector_data
        
    def _load_ticker_features(
        self,
        ticker: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Load pre-computed ticker features from parquet file."""
        # Load ticker data from parquet file
        features_path = self.ticker_features_dir / f"{ticker}_features.parquet"
        if not features_path.exists():
            logger.warning(f"No feature file found for {ticker}")
            return pd.DataFrame()
            
        try:
            df = pd.read_parquet(features_path)
            # Ensure date is the index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'date' in df.columns:
                    df.set_index('date', inplace=True)
                elif 'Date' in df.columns:
                    df.set_index('Date', inplace=True)
            df.index = pd.to_datetime(df.index)
            
            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            if df.empty:
                logger.warning(f"No data available for {ticker} in date range {start_date.date()} to {end_date.date()}")
                return pd.DataFrame()
                
            # Add ticker column
            df['ticker'] = ticker
            
            # Handle stock_splits - fill missing values with 0 (no splits)
            if 'stock_splits' in df.columns:
                df['stock_splits'] = df['stock_splits'].fillna(0)
            else:
                df['stock_splits'] = 0
                
            # Handle adj_close - if missing, use close price
            if 'adj_close' in df.columns:
                df['adj_close'] = df['adj_close'].fillna(df['close'])
            else:
                df['adj_close'] = df['close']
                
            # Handle analyst_targets - fill missing values with empty dict
            if 'analyst_targets' in df.columns:
                df['analyst_targets'] = df['analyst_targets'].fillna({})
            else:
                df['analyst_targets'] = {}
                
            return df
            
        except Exception as e:
            logger.error(f"Error loading features for {ticker}: {e}")
            return pd.DataFrame()
            
    def _add_sector_features(
        self,
        ticker_features: pd.DataFrame,
        sector_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add sector features to ticker features.
        
        Args:
            ticker_features: DataFrame containing ticker features
            sector_df: DataFrame containing sector features
            
        Returns:
            DataFrame with added sector features
        """
        # For each date in ticker_features, get the most recent sector data
        for date in ticker_features.index:
            available_dates = sector_df.index[sector_df.index <= date]
            if len(available_dates) > 0:
                latest_date = available_dates.max()
                sector_data = sector_df.loc[latest_date]
                ticker_features.loc[date, sector_data.index] = sector_data.values
                
        return ticker_features
        
    def _split_dataset(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train/val/test sets using a sophisticated time-based approach.
        
        This method:
        1. Ensures each split has sufficient data for each ticker
        2. Uses specific time periods that make sense for market data
        3. Maintains chronological order and prevents data leakage
        
        Args:
            df: DataFrame to split
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Sort by date to ensure chronological order
        df = df.sort_index()
        
        # Get the date range
        start_date = df.index.min()
        end_date = df.index.max()
        
        # Calculate split dates
        # Use specified number of months for training
        train_end = start_date + pd.DateOffset(months=self.train_months)
        # Use specified number of months for validation
        val_end = train_end + pd.DateOffset(months=self.val_months)
        
        # Split chronologically
        train_df = df[df.index <= train_end]
        val_df = df[(df.index > train_end) & (df.index <= val_end)]
        test_df = df[df.index > val_end]
        
        # Verify each ticker has sufficient data in each split
        for name, split_df in [("Training", train_df), ("Validation", val_df), ("Testing", test_df)]:
            ticker_counts = split_df['ticker'].value_counts()
            insufficient_tickers = ticker_counts[ticker_counts < self.min_samples_per_ticker]
            if not insufficient_tickers.empty:
                logger.warning(f"\n{name} set has tickers with insufficient data:")
                for ticker, count in insufficient_tickers.items():
                    logger.warning(f"{ticker}: {count} samples (minimum {self.min_samples_per_ticker} required)")
                raise ValueError(f"{name} set has tickers with insufficient data")
        
        logger.info(f"\nSplit dates and statistics:")
        logger.info(f"Training: {start_date.date()} to {train_end.date()}")
        logger.info(f"- {len(train_df)} total samples")
        logger.info(f"- {len(train_df['ticker'].unique())} unique tickers")
        logger.info(f"- {train_df['ticker'].value_counts().mean():.1f} samples per ticker on average")
        
        logger.info(f"\nValidation: {train_end.date()} to {val_end.date()}")
        logger.info(f"- {len(val_df)} total samples")
        logger.info(f"- {len(val_df['ticker'].unique())} unique tickers")
        logger.info(f"- {val_df['ticker'].value_counts().mean():.1f} samples per ticker on average")
        
        logger.info(f"\nTesting: {val_end.date()} to {end_date.date()}")
        logger.info(f"- {len(test_df)} total samples")
        logger.info(f"- {len(test_df['ticker'].unique())} unique tickers")
        logger.info(f"- {test_df['ticker'].value_counts().mean():.1f} samples per ticker on average")
        
        return train_df, val_df, test_df
        
    def _generate_readme(
        self,
        output_dir: Path,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        validate: bool = True
    ) -> None:
        """Generate a README file for the dataset.
        
        Args:
            output_dir: Path to save the README
            train_df: Training dataset
            val_df: Validation dataset
            test_df: Test dataset
            validate: Whether validation was performed
        """
        # Capture validation output
        import io
        from contextlib import redirect_stdout
        
        validation_output = ""
        if validate:
            f = io.StringIO()
            with redirect_stdout(f):
                self._validate_datasets(train_df, val_df, test_df)
            validation_output = f.getvalue()
        
        readme_content = f"""# Machine Learning Dataset

## Overview
This dataset was generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using the following parameters:
- Number of tickers: {len(train_df['ticker'].unique())}
- Date range: {train_df.index.min().strftime('%Y-%m-%d')} to {train_df.index.max().strftime('%Y-%m-%d')}
- Training months: {self.train_months}
- Validation months: {self.val_months}
- Minimum samples per ticker: {self.min_samples_per_ticker}
- Data validation: {'Enabled' if validate else 'Disabled'}

## Included Tickers
{', '.join(train_df['ticker'].unique())}

## Dataset Structure
The dataset is split into three parts:
1. Training set (train.parquet)
2. Validation set (val.parquet)
3. Test set (test.parquet)

## Features
The dataset includes:
1. Price/Volume Features
2. Technical Indicators
3. Sector Features
4. Market Features

## Data Quality
The dataset has been validated for:
- Consistent ticker distribution across splits
- No overlapping dates between splits
- No infinite values
- No zero variance features (except stock_splits)
- Missing value analysis for all features

## Feature Normalization
All numeric features (except binary features like stock_splits) are normalized using StandardScaler:
1. Mean and standard deviation are calculated from the training set only
2. These parameters are saved to disk in the 'data/scalers' directory
3. The same normalization is applied to validation and test sets
4. For inference, the saved parameters are used to normalize new data

Normalization parameters are saved as:
- One scaler per feature (e.g., 'open_scaler.joblib', 'volume_scaler.joblib')
- Each scaler contains the mean and standard deviation from the training set
- These parameters ensure consistent normalization across training and inference

## Usage
To load the dataset:
```python
import pandas as pd

# Load datasets
train_df = pd.read_parquet('train.parquet')
val_df = pd.read_parquet('val.parquet')
test_df = pd.read_parquet('test.parquet')
```

To prepare a single row for inference:
```python
from trading_advisor.dataset_v2 import DatasetGeneratorV2
from datetime import datetime

# Initialize generator
generator = DatasetGeneratorV2()

# Prepare single row
date = datetime(2024, 1, 2)
inference_data = generator.prepare_inference_data('AAPL', date)
```

## Validation Results
```
{validation_output}
```

## Generation Command
The dataset was generated using:
```bash
python -m trading_advisor.cli generate-dataset-v2 --start-date 2020-01-01 --end-date 2024-12-31 --output-dir data/ml_datasets
```

This command:
1. Uses data from 2020-01-01 to 2024-12-31
2. Saves the dataset to `data/ml_datasets/`
3. Creates train/val/test splits
4. Saves normalization parameters to `data/ml_datasets/scalers/`
"""
        
        readme_path = output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
            
        logger.info(f"Generated README at {readme_path}")
        
    def _validate_datasets(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> None:
        """Validate the generated datasets and provide detailed statistics.
        
        Args:
            train_df: Training dataset
            val_df: Validation dataset
            test_df: Test dataset
        """
        import typer
        
        # Columns that are expected to have missing values due to historical data requirements
        expected_missing = {
            'macd': 'Requires 26 days of historical data',
            'macd_signal': 'Requires 26 days of historical data',
            'macd_hist': 'Requires 26 days of historical data',
            'bb_upper': 'Requires 20 days of historical data',
            'bb_lower': 'Requires 20 days of historical data',
            'bb_middle': 'Requires 20 days of historical data',
            'bb_pband': 'Requires 20 days of historical data',
            'sma_20': 'Requires 20 days of historical data',
            'sma_50': 'Requires 50 days of historical data',
            'sma_100': 'Requires 100 days of historical data',
            'ema_100': 'Requires 100 days of historical data',
            'sector_performance_returns_20d': 'Requires 20 days of historical data',
            'sector_performance_momentum_20d': 'Requires 20 days of historical data',
            'sector_performance_relative_strength': 'Requires S&P 500 data for comparison',
            'sector_performance_relative_strength_ratio': 'Requires S&P 500 data for comparison',
            'analyst_targets': 'Only available for most recent date'
        }
        
        # Print detailed statistics for each dataset
        for name, df in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
            typer.echo(f"\n{name} Dataset Statistics:")
            typer.echo(f"Samples: {len(df)} | Date Range: {df.index.min().date()} to {df.index.max().date()} | Tickers: {len(df['ticker'].unique())} | Avg Samples/Ticker: {df['ticker'].value_counts().mean():.1f}")
            
            # Calculate missing values statistics
            missing = df.isnull().sum()
            missing_pct = (missing / len(df)) * 100
            
            # Group columns by missing value percentage
            high_missing = missing_pct[missing_pct > 10]
            medium_missing = missing_pct[(missing_pct > 0) & (missing_pct <= 10)]
            no_missing = missing_pct[missing_pct == 0]
            
            typer.echo("\nMissing Values Summary:")
            if not high_missing.empty:
                typer.echo("High Missing (>10%):")
                for col in high_missing.index:
                    reason = expected_missing.get(col, "Unexpected")
                    typer.echo(f"- {col}: {missing[col]} ({missing_pct[col]:.1f}%) - {reason}")
            
            if not medium_missing.empty:
                typer.echo("Medium Missing (0-10%):")
                for col in medium_missing.index:
                    reason = expected_missing.get(col, "Unexpected")
                    typer.echo(f"- {col}: {missing[col]} ({missing_pct[col]:.1f}%) - {reason}")
            
            if not no_missing.empty:
                typer.echo(f"No Missing Values ({len(no_missing)} columns)")
            
            # Calculate basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if not numeric_cols.empty:
                typer.echo("\nNumeric Column Statistics:")
                stats = df[numeric_cols].describe()
                for col in numeric_cols:
                    typer.echo(f"{col}: Mean={stats[col]['mean']:.2f} | Std={stats[col]['std']:.2f} | Min={stats[col]['min']:.2f} | Max={stats[col]['max']:.2f}")
        
        # Now do error checks
        for name, df in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
            # Check for infinite values
            inf_mask = np.isinf(df.select_dtypes(include=[np.number]))
            if inf_mask.any().any():
                typer.echo("\nInfinite Values Found:")
                for col in inf_mask.columns[inf_mask.any()]:
                    inf_count = inf_mask[col].sum()
                    typer.echo(f"- {col}: {inf_count} infinite values")
                raise ValueError(f"{name} dataset contains infinite values")
            
            # Check for zero variance features (excluding stock_splits)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = numeric_cols.drop('stock_splits', errors='ignore')
            zero_var_cols = [col for col in numeric_cols if df[col].nunique() <= 1]
            if zero_var_cols:
                typer.echo("\nZero Variance Features Found:")
                for col in zero_var_cols:
                    typer.echo(f"- {col}: {df[col].nunique()} unique values")
                raise ValueError(f"{name} dataset contains zero variance features")
        
        # Check for date range consistency
        train_dates = set(train_df.index)
        val_dates = set(val_df.index)
        test_dates = set(test_df.index)
        
        if train_dates & val_dates:
            raise ValueError("Training and validation sets have overlapping dates")
        if train_dates & test_dates:
            raise ValueError("Training and test sets have overlapping dates")
        if val_dates & test_dates:
            raise ValueError("Validation and test sets have overlapping dates")
            
        # Check for ticker distribution
        train_tickers = set(train_df['ticker'].unique())
        val_tickers = set(val_df['ticker'].unique())
        test_tickers = set(test_df['ticker'].unique())
        
        if train_tickers != val_tickers or train_tickers != test_tickers:
            typer.echo("\nTicker Distribution:")
            typer.echo(f"TRAIN: {len(train_tickers)} tickers | VAL: {len(val_tickers)} tickers | TEST: {len(test_tickers)} tickers")
            raise ValueError("Ticker distribution is not consistent across datasets") 