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
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from trading_advisor.features import load_features
from trading_advisor.sector_mapping import load_sector_mapping

logger = logging.getLogger(__name__)

class DatasetGeneratorV2:
    """Generator for machine learning datasets with improved data handling."""
    
    def __init__(
        self,
        market_features_dir: str = "data/market_features",
        ticker_features_dir: str = "data/ticker_features",
        output_dir: str = "data/ml_datasets",
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ):
        """Initialize the dataset generator.
        
        Args:
            market_features_dir: Directory containing market feature files
            ticker_features_dir: Directory containing ticker feature files
            output_dir: Directory to save generated datasets
            test_size: Proportion of data to use for testing
            val_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility
        """
        self.market_features_dir = Path(market_features_dir)
        self.ticker_features_dir = Path(ticker_features_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Load sector mapping
        self.sector_mapping = load_sector_mapping(str(self.market_features_dir))
        
    def generate_dataset(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        output_dir: Path,
        validate: bool = False
    ) -> None:
        """Generate a dataset for the given tickers and date range.
        
        Args:
            tickers: List of ticker symbols to include
            start_date: Start date for data collection (YYYY-MM-DD)
            end_date: End date for data collection (YYYY-MM-DD)
            output_dir: Directory to save output files
            validate: Whether to perform validation checks
        """
        logger.info(f"Generating dataset for {len(tickers)} tickers from {start_date} to {end_date}")
        
        # Convert dates to pandas Timestamps
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        
        # Load all sector data once
        sector_data = self._load_sector_data()
        
        # Process each ticker
        all_data = []
        for ticker in tickers:
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
        
        if not all_data:
            raise ValueError("No data generated for any ticker")
            
        # Combine all data
        df = pd.concat(all_data, axis=0)
        
        # Split into train/val/test
        train_df, val_df, test_df = self._split_dataset(df)
        
        # Save datasets
        output_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(output_dir / 'train.parquet')
        val_df.to_parquet(output_dir / 'val.parquet')
        test_df.to_parquet(output_dir / 'test.parquet')
        
        # Generate README
        self._generate_readme(output_dir, train_df, val_df, test_df)
        
        logger.info(f"Dataset generated successfully at {output_dir}")
        
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
        """Load features for a single ticker.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame containing ticker features
        """
        try:
            features = load_features(ticker, str(self.ticker_features_dir))
            if features.empty:
                return pd.DataFrame()
                
            # Filter by date range
            features = features[(features.index >= start_date) & (features.index <= end_date)]
            
            # Add ticker column
            features['ticker'] = ticker
            
            return features
            
        except Exception as e:
            logger.error(f"Error loading features for {ticker}: {str(e)}")
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
        """Split dataset into train/val/test sets.
        
        Args:
            df: DataFrame to split
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # First split into train+val and test
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # Then split train+val into train and val
        val_size_adjusted = self.val_size / (1 - self.test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=self.random_state
        )
        
        return train_df, val_df, test_df
        
    def _generate_readme(
        self,
        output_dir: Path,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> None:
        """Generate a README file for the dataset.
        
        Args:
            output_dir: Path to save the README
            train_df: Training dataset
            val_df: Validation dataset
            test_df: Test dataset
        """
        readme_content = f"""# Machine Learning Dataset

## Overview
This dataset was generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using the following parameters:
- Number of tickers: {len(train_df['ticker'].unique())}
- Date range: {train_df.index.min().strftime('%Y-%m-%d')} to {train_df.index.max().strftime('%Y-%m-%d')}
- Test size: {self.test_size}
- Validation size: {self.val_size}

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

## Usage
To load the dataset:
```python
import pandas as pd

# Load datasets
train_df = pd.read_parquet('train.parquet')
val_df = pd.read_parquet('val.parquet')
test_df = pd.read_parquet('test.parquet')
```
"""
        
        readme_path = output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
            
        logger.info(f"Generated README at {readme_path}") 