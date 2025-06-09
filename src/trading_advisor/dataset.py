"""Dataset generation functionality."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from trading_advisor.sector_mapping import load_sector_mapping

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
        self.sector_mapping = load_sector_mapping(str(self.market_features_dir))
    
    def _load_sector_features(self, ticker: str, date: pd.Timestamp) -> pd.DataFrame:
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
        start_date: str,
        end_date: str,
        target_days: int = 5,
        target_return: float = 0.02,
        min_samples: int = 10,
        output: Optional[str] = None,
        force: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Generate dataset for classification.
        
        Args:
            tickers: List of tickers to include
            start_date: Start date for dataset
            end_date: End date for dataset
            target_days: Number of days to look ahead for target
            target_return: Target return threshold
            min_samples: Minimum number of samples required per ticker
            output: Output directory (optional)
            force: Whether to overwrite existing files
            
        Returns:
            Dictionary containing train, validation, and test datasets
        """
        logger.info(f"Starting dataset generation for tickers: {tickers}")
        
        # Set output directory
        output_dir = Path(output) if output else self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if files already exist
        train_path = output_dir / 'train.parquet'
        val_path = output_dir / 'val.parquet'
        test_path = output_dir / 'test.parquet'
        
        if not force and (train_path.exists() or val_path.exists() or test_path.exists()):
            raise ValueError(
                f"Dataset files already exist in {output_dir}. Use --force to overwrite."
            )
        
        # Load and process data for each ticker
        all_data = []
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn()) as progress:
            task = progress.add_task("Loading ticker data...", total=len(tickers))
            for ticker in tickers:
                try:
                    # Load ticker features
                    ticker_file = self.ticker_features_dir / f"{ticker}_features.parquet"
                    if not ticker_file.exists():
                        logger.warning(f"No features found for {ticker}")
                        progress.update(task, advance=1)
                        continue
                    
                    ticker_data = pd.read_parquet(ticker_file)
                    
                    # Ensure date is the index
                    if not isinstance(ticker_data.index, pd.DatetimeIndex):
                        if 'Date' in ticker_data.columns:
                            ticker_data = ticker_data.set_index('Date')
                        elif 'date' in ticker_data.columns:
                            ticker_data = ticker_data.set_index('date')
                        else:
                            logger.warning(f"No date column found in ticker features for {ticker}")
                            progress.update(task, advance=1)
                            continue
                    
                    # Filter date range
                    ticker_data = ticker_data[
                        (ticker_data.index >= pd.Timestamp(start_date)) & 
                        (ticker_data.index <= pd.Timestamp(end_date))
                    ]
                    
                    if len(ticker_data) < min_samples:
                        logger.warning(f"Insufficient samples for {ticker}: {len(ticker_data)} < {min_samples}")
                        progress.update(task, advance=1)
                        continue
                    
                    # Normalize column names to lowercase
                    ticker_data.columns = ticker_data.columns.str.lower()
                    
                    # Calculate future returns using 'close' price
                    if 'close' not in ticker_data.columns:
                        logger.error(f"No 'close' column found in ticker features for {ticker}. Available columns: {ticker_data.columns.tolist()}")
                        progress.update(task, advance=1)
                        continue
                        
                    future_returns = ticker_data['close'].shift(-target_days) / ticker_data['close'] - 1
                    ticker_data['label'] = (future_returns >= target_return).astype(int)
                    
                    # Drop rows with NaN labels
                    ticker_data = ticker_data.dropna(subset=['label'])
                    
                    if len(ticker_data) < min_samples:
                        logger.warning(f"Insufficient samples after label generation for {ticker}: {len(ticker_data)} < {min_samples}")
                        progress.update(task, advance=1)
                        continue
                    
                    # Add ticker column
                    ticker_data['ticker'] = ticker
                    
                    # Load and merge sector features
                    for date in ticker_data.index:
                        sector_features = self._load_sector_features(ticker, date)
                        if not sector_features.empty:
                            for col in sector_features.columns:
                                ticker_data.loc[date, col] = sector_features[col].iloc[0]
                    
                    # Validate features
                    missing_features = self._validate_features(ticker_data, ticker)
                    if missing_features:
                        logger.warning(f"Missing features for {ticker}: {missing_features}")
                        # Fill missing features with NaN
                        for feature in missing_features:
                            ticker_data[feature] = np.nan
                    
                    all_data.append(ticker_data)
                    progress.update(task, advance=1)
                    
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {str(e)}")
                    progress.update(task, advance=1)
                    continue
        
        if not all_data:
            raise ValueError("No valid data was generated for any ticker")
        
        # Combine all data
        combined_data = pd.concat(all_data)
        
        # Split into train/val/test (60/20/20)
        dates = sorted(combined_data.index.unique())
        split_idx = int(len(dates) * 0.6)
        val_idx = int(len(dates) * 0.8)
        
        train_dates = dates[:split_idx]
        val_dates = dates[split_idx:val_idx]
        test_dates = dates[val_idx:]
        
        train_data = combined_data[combined_data.index.isin(train_dates)]
        val_data = combined_data[combined_data.index.isin(val_dates)]
        test_data = combined_data[combined_data.index.isin(test_dates)]
        
        # Save datasets
        logger.info(f"Saving train dataset with shape {train_data.shape}")
        logger.info(f"Saving val dataset with shape {val_data.shape}")
        logger.info(f"Saving test dataset with shape {test_data.shape}")
        
        train_data.to_parquet(train_path)
        val_data.to_parquet(val_path)
        test_data.to_parquet(test_path)
        
        # Generate README
        self._generate_readme(output_dir, {
            'start_date': start_date,
            'end_date': end_date,
            'target_days': target_days,
            'target_return': target_return,
            'min_samples': min_samples,
            'tickers': tickers,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'train_positive_pct': (train_data['label'].mean() * 100),
            'val_positive_pct': (val_data['label'].mean() * 100),
            'test_positive_pct': (test_data['label'].mean() * 100)
        })
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def _generate_readme(self, output_dir: Path, params: Dict) -> None:
        """Generate README.md file with dataset information."""
        readme_content = f"""# Machine Learning Dataset

## Generation Parameters
- Start Date: {params['start_date']}
- End Date: {params['end_date']}
- Target Days: {params['target_days']}
- Target Return: {params['target_return']}
- Minimum Samples per Ticker: {params['min_samples']}
- Tickers: {', '.join(params['tickers'])}

## Dataset Statistics
- Training Set:
  - Samples: {params['train_samples']}
  - Positive Labels: {params['train_positive_pct']:.1f}%
- Validation Set:
  - Samples: {params['val_samples']}
  - Positive Labels: {params['val_positive_pct']:.1f}%
- Test Set:
  - Samples: {params['test_samples']}
  - Positive Labels: {params['test_positive_pct']:.1f}%

## Feature Information
The dataset includes the following features:

### Ticker Features
#### Price/Volume Metrics
- open, high, low, close
- volume, volume_prev
- dividends, stock_splits
- adj_close

#### Technical Indicators
- RSI (14-day)
- MACD (macd, macd_signal, macd_hist)
- Bollinger Bands (bb_upper, bb_lower, bb_middle, bb_pband)

#### Moving Averages
- Simple Moving Averages: sma_20, sma_50, sma_100, sma_200
- Exponential Moving Averages: ema_100, ema_200

#### Analyst Information
- analyst_targets: JSON string containing current price and median target

### Sector Features
#### Generic Sector Performance
- Price, Volatility, Volume
- Returns: 1-day, 5-day, 20-day
- Momentum: 5-day, 20-day
- Relative Strength and Ratio

#### Sector-Specific Features
For each sector (e.g., healthcare, technology), the following metrics are included:
- Price, Volatility, Volume
- Returns: 1-day, 5-day, 20-day
- Momentum: 5-day, 20-day
- Relative Strength and Ratio

### Additional Fields
- label: Binary classification target (1 for positive returns, 0 otherwise)
- ticker: Ticker symbol

## Usage Notes
1. Loading the Dataset:
```python
import pandas as pd
from pathlib import Path

# Load the datasets
data_dir = Path("data/ml_datasets/your_dataset_dir")
train_df = pd.read_parquet(data_dir / "train.parquet")
val_df = pd.read_parquet(data_dir / "val.parquet")
test_df = pd.read_parquet(data_dir / "test.parquet")
```

2. Model Training Considerations:
   - Handle class imbalance (positive labels are typically less frequent)
   - Use appropriate time-series cross-validation
   - Consider feature importance and correlation
   - Account for the time-series nature of the data
   - Note that sector-specific features may contain NaN values for tickers not in that sector
"""
        
        readme_path = output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        logger.info(f"Generated README.md at {readme_path}") 