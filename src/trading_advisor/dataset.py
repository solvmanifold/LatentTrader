"""Dataset generation functionality."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import json
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from trading_advisor.features import load_features
from trading_advisor.sector_mapping import load_sector_mapping
from trading_advisor.normalization import FeatureNormalizer

logger = logging.getLogger(__name__)

class DatasetGenerator:
    """Generator for machine learning datasets."""
    
    # Define expected features from documentation
    EXPECTED_TICKER_FEATURES = [
        # Price/Volume
        'open', 'high', 'low', 'close', 'volume', 'volume_prev',
        'dividends', 'stock_splits', 'adj_close',
        
        # Technical Indicators
        'rsi',  # 14-day RSI
        'macd', 'macd_signal', 'macd_hist',  # MACD
        'bb_upper', 'bb_lower', 'bb_middle', 'bb_pband',  # Bollinger Bands
        
        # Moving Averages
        'sma_20', 'sma_50', 'sma_100', 'sma_200',
        'ema_100', 'ema_200',
        
        # Analyst Info
        'analyst_targets'
    ]
    
    EXPECTED_SECTOR_FEATURES = [
        'sector_performance_price',
        'sector_performance_volatility',
        'sector_performance_volume',
        'sector_performance_returns_1d',
        'sector_performance_returns_5d',
        'sector_performance_returns_20d',
        'sector_performance_momentum_5d',
        'sector_performance_momentum_20d',
        'sector_performance_relative_strength',
        'sector_performance_relative_strength_ratio'
    ]
    
    def __init__(
        self,
        market_features_dir: str = "data/market_features",
        ticker_features_dir: str = "data/ticker_features",
        output_dir: str = "data/ml_datasets",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        target_days: int = 5,
        min_samples_per_ticker: int = 100,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        normalization_version: str = "1.0.0"
    ):
        """Initialize the dataset generator.
        
        Args:
            market_features_dir: Directory containing market feature files
            ticker_features_dir: Directory containing ticker feature files
            output_dir: Directory to save generated datasets
            start_date: Start date for data collection
            end_date: End date for data collection
            target_days: Number of days to look ahead for labeling
            min_samples_per_ticker: Minimum number of samples required per ticker
            test_size: Proportion of data to use for testing
            val_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility
            normalization_version: Version of normalization parameters to use
        """
        self.market_features_dir = Path(market_features_dir)
        self.ticker_features_dir = Path(ticker_features_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load sector mapping
        self.sector_mapping = load_sector_mapping(str(self.market_features_dir))
        
        # Initialize feature normalizer
        self.normalizer = FeatureNormalizer(
            output_dir=self.output_dir / "normalization",
            version=normalization_version
        )
        
        self.start_date = start_date or (datetime.now() - timedelta(days=365*2))
        self.end_date = end_date or datetime.now()
        self.target_days = target_days
        self.min_samples_per_ticker = min_samples_per_ticker
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
    
    def _load_sector_features(self, ticker: str, date: datetime) -> pd.DataFrame:
        """Load sector-specific features for a ticker.
        
        Args:
            ticker: Ticker symbol
            date: Date to load features for
            
        Returns:
            DataFrame containing sector features
        """
        # Get sector for ticker
        sector = self.sector_mapping.get(ticker, 'Unknown')
        if sector == 'Unknown':
            logger.warning(f"No sector mapping found for {ticker}")
            return pd.DataFrame()
            
        sector_file = self.market_features_dir / 'sectors' / f"{sector.lower().replace(' ', '_')}.parquet"
        
        if not sector_file.exists():
            logger.warning(f"No sector features found for {sector}")
            return pd.DataFrame()
            
        try:
            df = pd.read_parquet(sector_file)
            # Convert date to pandas Timestamp if it's a datetime
            if isinstance(date, datetime):
                date = pd.Timestamp(date)
            df = df[df.index == date]
            return df
        except Exception as e:
            logger.error(f"Error loading sector features for {sector}: {e}")
            return pd.DataFrame()
    
    def _validate_features(self, df: pd.DataFrame, ticker: str) -> List[str]:
        """Validate that all expected features are present.
        
        Args:
            df: DataFrame to validate
            ticker: Ticker symbol for logging
            
        Returns:
            List of missing features
        """
        missing_features = []
        
        # Check ticker features
        for feature in self.EXPECTED_TICKER_FEATURES:
            if feature not in df.columns:
                missing_features.append(feature)
        
        # Check sector features
        sector = self.sector_mapping.get(ticker, 'Unknown')
        if sector != 'Unknown':
            for feature in self.EXPECTED_SECTOR_FEATURES:
                feature_name = f"{sector.lower().replace(' ', '_')}_{feature}"
                if feature_name not in df.columns:
                    missing_features.append(feature_name)
        
        if missing_features:
            logger.warning(f"Missing features for {ticker}: {missing_features}")
        
        return missing_features
    
    def generate_dataset(
        self,
        tickers: List[str],
        output_name: str = "dataset"
    ) -> None:
        """Generate a machine learning dataset.
        
        Args:
            tickers: List of tickers to include
            output_name: Name for the output dataset
        """
        logger.info(f"Generating dataset for {len(tickers)} tickers")
        
        # Load sector mapping
        sector_mapping = self.sector_mapping
        
        # Initialize storage for all data
        all_data = []
        
        # Load and process data for each ticker
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            transient=True
        ) as progress:
            task = progress.add_task(
                f"Processing {len(tickers)} tickers...",
                total=len(tickers)
            )
            
            for ticker in tickers:
                logger.info(f"Loading features for {ticker} from {self.ticker_features_dir}")
                ticker_features = load_features(ticker, str(self.ticker_features_dir))
                logger.info(f"Loaded features for {ticker}: {ticker_features.shape if not ticker_features.empty else 'empty'}")
                
                if ticker_features.empty:
                    logger.warning(f"No features found for {ticker}")
                    progress.advance(task)
                    continue
                
                # Filter by date range
                ticker_features = ticker_features[(ticker_features.index >= self.start_date) & (ticker_features.index <= self.end_date)]
                if ticker_features.empty:
                    logger.warning(f"No features in date range for {ticker}")
                    progress.advance(task)
                    continue
                
                for date, row in ticker_features.iterrows():
                    # Load sector features for this date
                    sector_features = self._load_sector_features(ticker, date)
                    if not sector_features.empty:
                        # Flatten sector features to a dict with prefixed keys
                        sector_dict = {f"{self.sector_mapping.get(ticker, 'unknown').lower().replace(' ', '_')}_{col}": val for col, val in sector_features.iloc[0].items()}
                    else:
                        sector_dict = {}
                    # Combine ticker and sector features
                    features = dict(row)
                    features.update(sector_dict)
                    features['ticker'] = ticker
                    features['date'] = date
                    all_data.append(features)
                progress.advance(task)
                
        if not all_data:
            raise ValueError("No valid features found for any tickers")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Validate features
        self._validate_features(df, ticker)
        
        # Split into train/val/test
        train_df, val_df, test_df = self._split_dataset(df)
        
        # Identify columns to normalize (numeric, excluding 'ticker', 'date', 'analyst_targets')
        exclude_cols = {'ticker', 'date', 'analyst_targets'}
        numeric_cols = [col for col in train_df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[col])]
        # Fit normalizer only on numeric columns
        self.normalizer.fit(train_df[numeric_cols])
        # Normalize numeric columns, keep others unchanged
        def normalize_df(df):
            df_norm = df.copy()
            df_norm[numeric_cols] = self.normalizer.transform(df[numeric_cols])
            return df_norm
        train_df = normalize_df(train_df)
        val_df = normalize_df(val_df)
        test_df = normalize_df(test_df)
        
        # Save datasets
        output_path = self.output_dir / output_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_df.to_parquet(output_path / "train.parquet")
        val_df.to_parquet(output_path / "val.parquet")
        test_df.to_parquet(output_path / "test.parquet")
        
        # Generate README
        self._generate_readme(output_path, tickers)
        
        logger.info(f"Dataset generated at {output_path}")
    
    def _split_dataset(
        self,
        df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train/val/test sets.
        
        Args:
            df: DataFrame to split
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Group by ticker to ensure proper splitting
        ticker_groups = df.groupby('ticker')
        
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        for _, group in ticker_groups:
            if len(group) < self.min_samples_per_ticker:
                continue
                
            # Calculate split indices
            n = len(group)
            test_size = int(n * self.test_size)
            val_size = int(n * self.val_size)
            
            # Shuffle and split
            group = group.sample(frac=1, random_state=self.random_state)
            
            test_df = group.iloc[:test_size]
            val_df = group.iloc[test_size:test_size + val_size]
            train_df = group.iloc[test_size + val_size:]
            
            train_dfs.append(train_df)
            val_dfs.append(val_df)
            test_dfs.append(test_df)
            
        return (
            pd.concat(train_dfs),
            pd.concat(val_dfs),
            pd.concat(test_dfs)
        )
    
    def _generate_readme(self, output_path: Path, tickers: List[str]) -> None:
        """Generate README file for the dataset.
        
        Args:
            output_path: Path to save README
            tickers: List of tickers in dataset
        """
        readme_content = f"""# Machine Learning Dataset

## Generation Parameters
- Start Date: {self.start_date.strftime('%Y-%m-%d')}
- End Date: {self.end_date.strftime('%Y-%m-%d')}
- Target Days: {self.target_days}
- Minimum Samples per Ticker: {self.min_samples_per_ticker}
- Test Size: {self.test_size:.1%}
- Validation Size: {self.val_size:.1%}
- Random State: {self.random_state}
- Normalization Version: {self.normalizer.version}

## Dataset Statistics
- Number of Tickers: {len(tickers)}
- Tickers: {', '.join(tickers)}
- Training Set Size: {len(pd.read_parquet(output_path / 'train.parquet'))}
- Validation Set Size: {len(pd.read_parquet(output_path / 'val.parquet'))}
- Test Set Size: {len(pd.read_parquet(output_path / 'test.parquet'))}

## Feature Information

### Ticker Features
1. Price/Volume Metrics:
   - open, high, low, close, volume, volume_prev
   - dividends, stock_splits, adj_close

2. Technical Indicators:
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - Bollinger Bands (upper, lower, middle, %b)

3. Moving Averages:
   - SMA (Simple Moving Average): 20, 50, 100, 200 days
   - EMA (Exponential Moving Average): 100, 200 days

4. Analyst Information:
   - analyst_targets: JSON string containing current price and median target

### Sector Features
1. Generic Sector Performance:
   - Price, Volatility, Volume
   - Returns: 1-day, 5-day, 20-day
   - Momentum: 5-day, 20-day
   - Relative Strength and Ratio

2. Sector-Specific Features
For each sector (e.g., healthcare, technology), the following metrics are included:
- Price, Volatility, Volume
- Returns: 1-day, 5-day, 20-day
- Momentum: 5-day, 20-day
- Relative Strength and Ratio

Note: Sector-specific features are prefixed with the sector name (e.g., 'technology_performance_1d').

## Normalization
The dataset uses feature-specific normalization strategies:

1. Price/Volume Features:
   - Robust scaling to handle outliers
   - Preserves relative relationships while reducing impact of extreme values

2. Technical Indicators:
   - Standard scaling (z-score normalization)
   - Centers data around mean with unit variance

3. Bollinger Bands:
   - Min-max scaling to [0,1] range
   - Preserves relative positions within bands

4. Returns/Momentum:
   - Standard scaling
   - Centers around mean with unit variance

Normalization parameters are versioned and stored in the 'normalization' directory.
To normalize new data, use the same version of normalization parameters.

## Usage Notes
1. All features are normalized using version {self.normalizer.version}
2. Missing values are preserved as NaN
3. Sector-specific features may be NaN for some tickers
4. The dataset is split chronologically by ticker to prevent data leakage
"""
        
        # Write README
        readme_path = output_path / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
            
        logger.info(f"Generated README at {readme_path}") 