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
        target_days: int,
        target_return: float
    ) -> pd.DataFrame:
        """Generate binary labels based on future returns."""
        # Sort by date to ensure correct future return calculation
        df = df.sort_values('Date')
        
        # Calculate future returns
        future_returns = df['Close'].shift(-target_days) / df['Close'] - 1
        
        # Generate binary labels
        df['label'] = (future_returns >= target_return).astype(int)
        
        return df
    
    def _load_market_features(self, date: pd.Timestamp) -> pd.DataFrame:
        """
        Loads and merges all market features for a specific date.
        Returns a single-row DataFrame with all market features for the given date.
        """
        market_features = {}
        for path in self.market_features_dir.glob("*.parquet"):
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
            
            # Filter for target date
            row = df[df.index == date]
            if not row.empty:
                row = row.add_suffix(f'_{feature_type}')
                market_features[feature_type] = row
        
        if market_features:
            merged = pd.concat(market_features.values(), axis=1)
            return merged
        else:
            return pd.DataFrame()
    
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
            
            # Ensure date is the index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'Date' in df.columns:
                    df = df.set_index('Date')
                elif 'date' in df.columns:
                    df = df.set_index('date')
                else:
                    logger.warning(f"No date column found in sector features for {sector}")
                    return pd.DataFrame()
            df.index = pd.to_datetime(df.index)
            
            # Filter for target date
            target_date = pd.to_datetime(date)
            df = df[df.index == target_date]
            
            if df.empty:
                logger.warning(f"No sector features found for {sector} on {date}")
                return pd.DataFrame()
            
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
        ticker_file = self.ticker_features_dir / f"{ticker}_features.parquet"
        if not ticker_file.exists():
            logger.warning(f"No features found for {ticker}")
            return pd.DataFrame()
            
        ticker_features = pd.read_parquet(ticker_file)
        
        # Ensure date is the index
        if not isinstance(ticker_features.index, pd.DatetimeIndex):
            if 'Date' in ticker_features.columns:
                ticker_features = ticker_features.set_index('Date')
            elif 'date' in ticker_features.columns:
                ticker_features = ticker_features.set_index('date')
            else:
                logger.warning(f"No date column found in ticker features for {ticker}")
                return pd.DataFrame()
        ticker_features.index = pd.to_datetime(ticker_features.index)
        
        # Filter for target date
        target_date = pd.to_datetime(date)
        ticker_features = ticker_features[ticker_features.index == target_date]
        if ticker_features.empty:
            logger.warning(f"No ticker features found for {ticker} on {date}")
            return pd.DataFrame()
        
        # Load market features
        market_features = self._load_market_features(target_date)
        if market_features.empty:
            logger.warning(f"No market features found for {date}")
            return pd.DataFrame()
        
        # Load sector features if requested
        if include_sector:
            sector_features = self._load_sector_features(ticker, date)
            if not sector_features.empty:
                logger.info(f"Found sector features for {ticker} on {date}")
                # Combine features
                features = pd.concat([ticker_features, market_features, sector_features], axis=1)
            else:
                logger.warning(f"No sector features found for {ticker} on {date}")
                features = pd.concat([ticker_features, market_features], axis=1)
        else:
            features = pd.concat([ticker_features, market_features], axis=1)
        
        logger.debug(f"Combined features columns: {features.columns.tolist()}")
        logger.debug(f"Combined features head:\n{features.head()}")
        
        # Log feature counts for debugging
        logger.info(f"Feature counts for {ticker} on {date}:")
        logger.info(f"Ticker features: {len(ticker_features.columns)}")
        logger.info(f"Market features: {len(market_features.columns)}")
        if include_sector and not sector_features.empty:
            logger.info(f"Sector features: {len(sector_features.columns)}")
        logger.info(f"Total features: {len(features.columns)}")
        
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
        """Generate complete dataset with splits."""
        # Generate splits
        splits = self.generate_splits(
            start_date,
            end_date,
            train_months,
            val_months,
            test_months
        )
        print(f"[DEBUG] Generated {len(splits)} splits.")
        for i, split in enumerate(splits):
            print(f"Split {i}: train {split['train_start'].date()} to {split['train_end'].date()}, val {split['val_start'].date()} to {split['val_end'].date()}, test {split['test_start'].date()} to {split['test_end'].date()}")
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
            print(f"[DEBUG] Split {i} sample counts: train={len(split_data['train'])}, val={len(split_data['val'])}, test={len(split_data['test'])}")
            datasets[f'split_{i}'] = split_data
            # Save split data (force save even if empty for debugging)
            split_dir = self.output_dir / f"split_{i}"
            split_dir.mkdir(exist_ok=True)
            for split_name, data in split_data.items():
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
                ticker_file = self.ticker_features_dir / f"{ticker}_features.parquet"
                if not ticker_file.exists():
                    continue
                    
                ticker_features = pd.read_parquet(ticker_file)
                
                # Handle date column/index
                if isinstance(ticker_features.index, pd.DatetimeIndex):
                    ticker_features = ticker_features.reset_index()
                    if 'Date' in ticker_features.columns:
                        date_col = 'Date'
                    elif 'date' in ticker_features.columns:
                        date_col = 'date'
                    else:
                        logger.warning(f"No date column found in ticker features for {ticker}")
                        continue
                else:
                    if 'Date' in ticker_features.columns:
                        date_col = 'Date'
                    elif 'date' in ticker_features.columns:
                        date_col = 'date'
                    else:
                        logger.warning(f"No date column found in ticker features for {ticker}")
                        continue
                
                # Convert date column to datetime
                ticker_features[date_col] = pd.to_datetime(ticker_features[date_col])
                
                # Filter by date range
                mask = (ticker_features[date_col] >= start_date) & (ticker_features[date_col] <= end_date)
                ticker_features = ticker_features[mask]
                
                if len(ticker_features) < min_samples:
                    logger.warning(f"Insufficient samples for {ticker} in {split_name}")
                    continue
                
                # Generate labels
                ticker_features = self.generate_labels(ticker_features, target_days, target_return)
                
                # Add ticker column
                ticker_features['ticker'] = ticker
                
                # Prepare features for each date, including market features
                prepared_features = []
                for _, row in ticker_features.iterrows():
                    date_str = str(row[date_col].date())
                    features = self.prepare_features(ticker, date_str, include_sector=True)
                    if not features.empty:
                        logger.debug(f"Features columns before label assignment: {features.columns.tolist()}")
                        logger.debug(f"Row being processed: {row}")
                        # Add label and ticker from the original row
                        features['label'] = row['label']
                        features['ticker'] = ticker
                        prepared_features.append(features)
                
                if prepared_features:
                    all_data.extend(prepared_features)
                
            except Exception as e:
                import traceback
                logger.error(f"Error processing {ticker} for {split_name}: {e}\n{traceback.format_exc()}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all ticker data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Remove rows with NaN labels (due to future returns calculation)
        combined_data = combined_data.dropna(subset=['label'])
        
        return combined_data 