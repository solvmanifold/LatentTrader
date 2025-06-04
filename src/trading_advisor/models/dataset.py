"""Dataset generation for machine learning models."""

import logging
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DatasetGenerator:
    """Generator for machine learning datasets."""
    
    def __init__(
        self,
        market_features_dir: str = "data/market_features",
        ticker_features_dir: str = "data/ticker_features",
        output_dir: str = "data/ml_datasets"
    ):
        """Initialize the dataset generator.
        
        Args:
            market_features_dir: Directory containing market feature files
            ticker_features_dir: Directory containing ticker feature files
            output_dir: Directory to save generated datasets
        """
        self.market_features_dir = Path(market_features_dir)
        self.ticker_features_dir = Path(ticker_features_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load sector mapping
        self.sector_mapping = pd.read_parquet(
            self.market_features_dir / "metadata" / "sector_mapping.parquet"
        )
    
    def generate_splits(
        self,
        start_date: str,
        end_date: str,
        train_months: int = 10,
        val_months: int = 1,
        test_months: int = 1,
        step_months: int = 1
    ) -> List[Dict[str, pd.Timestamp]]:
        """Generate time-based splits for cross validation.
        
        Args:
            start_date: Start date for splits
            end_date: End date for splits
            train_months: Number of months for training
            val_months: Number of months for validation
            test_months: Number of months for testing
            step_months: Number of months to step forward for each split
            
        Returns:
            List of dictionaries containing split dates
        """
        splits = []
        current_date = pd.Timestamp(start_date)
        end_timestamp = pd.Timestamp(end_date)
        
        while current_date + pd.DateOffset(months=train_months + val_months + test_months) <= end_timestamp:
            split = {
                'train_start': current_date,
                'train_end': current_date + pd.DateOffset(months=train_months),
                'val_start': current_date + pd.DateOffset(months=train_months),
                'val_end': current_date + pd.DateOffset(months=train_months + val_months),
                'test_start': current_date + pd.DateOffset(months=train_months + val_months),
                'test_end': current_date + pd.DateOffset(months=train_months + val_months + test_months)
            }
            splits.append(split)
            current_date += pd.DateOffset(months=step_months)
        
        return splits
    
    def generate_labels(
        self,
        df: pd.DataFrame,
        target_days: int = 5,
        target_return: float = 0.02
    ) -> pd.Series:
        """Generate binary labels based on future returns.
        
        Args:
            df: DataFrame containing price data
            target_days: Number of days to look ahead
            target_return: Target return threshold
            
        Returns:
            Series containing binary labels
        """
        # Calculate future returns
        future_returns = df['close'].shift(-target_days) / df['close'] - 1
        
        # Generate binary labels
        labels = (future_returns >= target_return).astype(int)
        
        return labels
    
    def _load_market_features(self, date: str) -> pd.DataFrame:
        """Load market features for a specific date.
        
        Args:
            date: Date to load features for
            
        Returns:
            DataFrame containing market features
        """
        market_features = {}
        
        # Load each market feature file
        for feature_file in self.market_features_dir.glob("*.parquet"):
            if feature_file.name == "metadata":
                continue
                
            try:
                df = pd.read_parquet(feature_file)
                if 'date' in df.columns:
                    df = df[df['date'] == date]
                    if not df.empty:
                        market_features[feature_file.stem] = df
            except Exception as e:
                logger.warning(f"Error loading {feature_file}: {e}")
                continue
        
        # Combine all market features
        if not market_features:
            return pd.DataFrame()
            
        # Start with the first dataframe
        combined_df = next(iter(market_features.values()))
        
        # Merge with remaining dataframes
        for df in list(market_features.values())[1:]:
            if not df.empty:
                combined_df = combined_df.merge(
                    df,
                    on='date',
                    how='outer',
                    suffixes=('', f'_{df.name}')
                )
        
        return combined_df
    
    def _load_sector_features(self, ticker: str, date: str) -> pd.DataFrame:
        """Load sector features for a ticker.
        
        Args:
            ticker: Ticker symbol
            date: Date to load features for
            
        Returns:
            DataFrame containing sector features
        """
        # Get sector for ticker
        sector_info = self.sector_mapping[self.sector_mapping['ticker'] == ticker]
        if sector_info.empty:
            logger.warning(f"No sector mapping found for {ticker}")
            return pd.DataFrame()
            
        sector = sector_info.iloc[0]['sector']
        sector_file = self.market_features_dir / 'sectors' / f"{sector}.parquet"
        
        if not sector_file.exists():
            logger.warning(f"No sector features found for {sector}")
            return pd.DataFrame()
            
        try:
            df = pd.read_parquet(sector_file)
            df = df[df['date'] == date]
            return df
        except Exception as e:
            logger.error(f"Error loading sector features for {sector}: {e}")
            return pd.DataFrame()
    
    def prepare_features(
        self,
        ticker: str,
        date: str,
        include_sector: bool = True
    ) -> pd.DataFrame:
        """Prepare features for a ticker on a specific date.
        
        Args:
            ticker: Ticker symbol
            date: Date to prepare features for
            include_sector: Whether to include sector features
            
        Returns:
            DataFrame containing prepared features
        """
        # Load ticker features
        ticker_features = pd.read_parquet(
            self.ticker_features_dir / f"{ticker}_features.parquet"
        )
        ticker_features = ticker_features[ticker_features['date'] == date]
        
        if ticker_features.empty:
            return pd.DataFrame()
        
        # Load market features
        market_features = self._load_market_features(date)
        
        # Load sector features if requested
        if include_sector:
            sector_features = self._load_sector_features(ticker, date)
            if not sector_features.empty:
                ticker_features = ticker_features.merge(
                    sector_features,
                    on='date',
                    how='left'
                )
        
        # Combine all features
        features = ticker_features.merge(
            market_features,
            on='date',
            how='left'
        )
        
        return features
    
    def generate_dataset(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        target_days: int = 5,
        target_return: float = 0.02,
        train_months: int = 10,
        val_months: int = 1,
        test_months: int = 1,
        min_samples: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """Generate complete dataset with splits.
        
        Args:
            tickers: List of tickers to include
            start_date: Start date for data
            end_date: End date for data
            target_days: Number of days to look ahead for labels
            target_return: Target return threshold for labels
            train_months: Number of months for training
            val_months: Number of months for validation
            test_months: Number of months for testing
            min_samples: Minimum number of samples required for a ticker
            
        Returns:
            Dictionary containing datasets for each split
        """
        # Generate splits
        splits = self.generate_splits(
            start_date,
            end_date,
            train_months,
            val_months,
            test_months
        )
        
        # Generate datasets for each split
        datasets = {}
        for i, split in enumerate(splits):
            split_data = {
                'train': self._generate_split_data(
                    tickers, split, 'train', target_days, target_return, min_samples
                ),
                'val': self._generate_split_data(
                    tickers, split, 'val', target_days, target_return, min_samples
                ),
                'test': self._generate_split_data(
                    tickers, split, 'test', target_days, target_return, min_samples
                )
            }
            datasets[f'split_{i}'] = split_data
            
            # Save split data
            split_dir = self.output_dir / f"split_{i}"
            split_dir.mkdir(exist_ok=True)
            
            for split_name, data in split_data.items():
                if not data.empty:
                    data.to_parquet(split_dir / f"{split_name}.parquet")
        
        return datasets
    
    def _generate_split_data(
        self,
        tickers: List[str],
        split: Dict[str, pd.Timestamp],
        split_name: str,
        target_days: int,
        target_return: float,
        min_samples: int
    ) -> pd.DataFrame:
        """Generate data for a specific split.
        
        Args:
            tickers: List of tickers to include
            split: Dictionary containing split dates
            split_name: Name of the split (train/val/test)
            target_days: Number of days to look ahead for labels
            target_return: Target return threshold for labels
            min_samples: Minimum number of samples required for a ticker
            
        Returns:
            DataFrame containing the split data
        """
        start_date = split[f'{split_name}_start']
        end_date = split[f'{split_name}_end']
        
        all_data = []
        for ticker in tickers:
            try:
                # Load ticker features
                ticker_features = pd.read_parquet(
                    self.ticker_features_dir / f"{ticker}_features.parquet"
                )
                
                # Filter by date range
                ticker_features = ticker_features[
                    (ticker_features['date'] >= start_date) &
                    (ticker_features['date'] <= end_date)
                ]
                
                if len(ticker_features) < min_samples:
                    logger.warning(f"Insufficient samples for {ticker} in {split_name}")
                    continue
                
                # Generate labels
                labels = self.generate_labels(
                    ticker_features,
                    target_days,
                    target_return
                )
                ticker_features['label'] = labels
                
                # Add ticker column
                ticker_features['ticker'] = ticker
                
                all_data.append(ticker_features)
                
            except Exception as e:
                logger.error(f"Error processing {ticker} for {split_name}: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all ticker data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Remove rows with NaN labels (due to future returns calculation)
        combined_data = combined_data.dropna(subset=['label'])
        
        return combined_data 