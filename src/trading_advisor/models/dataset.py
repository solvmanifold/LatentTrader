"""Dataset generation for machine learning models."""

import logging
from typing import List, Dict, Optional, Tuple, Callable
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console
import os

# Configure logging to only show WARNING and above by default
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

def remove_unnecessary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove unnecessary features from the dataset."""
    features_to_remove = [
        'analyst_targets',
        'Dividends',
        'Stock Splits',
        'Volume_Prev',
        'price',
        'volume'
    ]
    
    return df.drop(columns=[col for col in features_to_remove if col in df.columns])

def save_feature_mappings(df: pd.DataFrame, output_dir: Path) -> Dict:
    """Save mappings for categorical features."""
    mappings = {}
    
    # Map tickers to integers
    unique_tickers = sorted(df['ticker'].unique())
    ticker_map = {ticker: idx for idx, ticker in enumerate(unique_tickers)}
    mappings['ticker'] = ticker_map
    
    # Save mappings
    output_path = output_dir / 'feature_mappings.json'
    with open(output_path, 'w') as f:
        json.dump(mappings, f, indent=2)
    
    logger.debug(f"Saved feature mappings to {output_path}")
    return mappings

def apply_feature_mappings(df: pd.DataFrame, mappings: Dict) -> pd.DataFrame:
    """Apply mappings to categorical features."""
    df_mapped = df.copy()
    
    # Map tickers
    df_mapped['ticker'] = df_mapped['ticker'].map(mappings['ticker'])
    
    return df_mapped

class DatasetGenerator:
    """Generator for machine learning datasets."""
    
    def __init__(
        self,
        market_features_dir: str = "data/market_features",
        ticker_features_dir: str = "data/ticker_features",
        output_dir: str = "data/ml_datasets",
        feature_config: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        batch_size: int = 1000  # Number of samples to process at once
    ):
        """Initialize the dataset generator.
        
        Args:
            market_features_dir: Directory containing market feature files
            ticker_features_dir: Directory containing ticker feature files
            output_dir: Directory to save generated datasets
            feature_config: Path to feature configuration JSON file (optional)
            progress_callback: Optional callback function to update progress
            batch_size: Number of samples to process at once for memory efficiency
        """
        self.market_features_dir = Path(market_features_dir)
        self.ticker_features_dir = Path(ticker_features_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_callback = progress_callback
        self.batch_size = batch_size
        
        # Load feature configuration if provided
        self.feature_config = None
        if feature_config:
            with open(feature_config) as f:
                self.feature_config = json.load(f)
        
        # Load sector mapping
        self.sector_mapping = pd.read_parquet(
            self.market_features_dir / "metadata" / "sector_mapping.parquet"
        )
        
        # Initialize feature cache
        self._market_feature_cache = {}
        self._sector_feature_cache = {}
    
    def _clear_feature_cache(self):
        """Clear feature caches to free memory."""
        self._market_feature_cache.clear()
        self._sector_feature_cache.clear()
    
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
        # If 'Date' is not a column, but the index is DatetimeIndex, reset it
        if 'Date' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df['Date'] = df.index
        elif 'Date' in df.columns:
            # Ensure Date column is datetime
            df['Date'] = pd.to_datetime(df['Date'])
        else:
            raise ValueError("DataFrame must have a 'Date' column or DatetimeIndex")

        # Sort by date to ensure correct future return calculation
        df = df.sort_values('Date')
        
        # Calculate future returns
        future_returns = df['Close'].shift(-target_days) / df['Close'] - 1
        
        # Generate binary labels
        df['label'] = (future_returns >= target_return).astype(int)
        
        # Drop rows with NaN labels
        df = df.dropna(subset=['label'])
        
        return df
    
    def _add_prefix_from_filename(self, df, file_path, skip_cols=None):
        """Add prefix to columns based on the base filename (without .parquet)."""
        if skip_cols is None:
            skip_cols = ['Date', 'ticker']
        prefix = os.path.splitext(os.path.basename(str(file_path)))[0] + '_'
        rename_dict = {col: prefix + col for col in df.columns if col not in skip_cols}
        return df.rename(columns=rename_dict)

    def _load_market_features(self, date: pd.Timestamp) -> pd.DataFrame:
        """Load market features with caching."""
        date_str = date.strftime('%Y-%m-%d')
        if date_str in self._market_feature_cache:
            return self._market_feature_cache[date_str]
        market_features = {}
        for path in self.market_features_dir.glob("*.parquet"):
            if path.is_dir() or path.name.startswith("metadata"):
                continue
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
            df = self.drop_date_column(df)
            # Filter for target date or nearest previous date
            if date in df.index:
                row = df.loc[date]
            else:
                mask = df.index <= date
                if not mask.any():
                    continue
                nearest_date = df.index[mask].max()
                row = df.loc[nearest_date]
            row = row.to_frame().T
            row = self._add_prefix_from_filename(row, path)
            market_features[path.stem] = row
        if market_features:
            merged = pd.concat(market_features.values(), axis=1)
            self._market_feature_cache[date_str] = merged
            return merged
        else:
            return pd.DataFrame()

    def _load_sector_features(self, ticker: str, date: pd.Timestamp) -> pd.DataFrame:
        """Load sector features with caching."""
        date_str = date.strftime('%Y-%m-%d')
        cache_key = f"{ticker}_{date_str}"
        if cache_key in self._sector_feature_cache:
            return self._sector_feature_cache[cache_key]
        # Get sector for ticker
        sector = self.sector_mapping[self.sector_mapping['ticker'] == ticker]['sector'].iloc[0]
        sector_file = self.market_features_dir / "sectors" / f"{sector}.parquet"
        if not sector_file.exists():
            return pd.DataFrame()
        sector_features = pd.read_parquet(sector_file)
        # Ensure date is the index
        if not isinstance(sector_features.index, pd.DatetimeIndex):
            if 'Date' in sector_features.columns:
                sector_features = sector_features.set_index('Date')
            elif 'date' in sector_features.columns:
                sector_features = sector_features.set_index('date')
            else:
                return pd.DataFrame()
        sector_features.index = pd.to_datetime(sector_features.index)
        sector_features = self.drop_date_column(sector_features)
        # Filter for target date or nearest previous date
        if date in sector_features.index:
            row = sector_features.loc[date]
        else:
            mask = sector_features.index <= date
            if not mask.any():
                return pd.DataFrame()
            nearest_date = sector_features.index[mask].max()
            row = sector_features.loc[nearest_date]
        row = row.to_frame().T
        row = self._add_prefix_from_filename(row, sector_file)
        self._sector_feature_cache[cache_key] = row
        return row

    def prepare_features(
        self,
        ticker: str,
        date: str,
        include_sector: bool = True
    ) -> pd.DataFrame:
        """Prepare features for a ticker on a specific date."""
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
        ticker_features = self.drop_date_column(ticker_features)
        ticker_features.index.name = None
        # Filter for target date
        target_date = pd.to_datetime(date)
        ticker_features = ticker_features[ticker_features.index == target_date]
        if ticker_features.empty:
            logger.warning(f"No ticker features found for {ticker} on {date}")
            return pd.DataFrame()
        ticker_features = self._add_prefix_from_filename(ticker_features, ticker_file)
        # Load market features (using cache)
        market_features = self._load_market_features(target_date)
        if market_features.empty:
            logger.warning(f"No market features found for {date}")
            return pd.DataFrame()
        # Load sector features if requested (using cache)
        if include_sector:
            sector_features = self._load_sector_features(ticker, target_date)
            if not sector_features.empty:
                features = pd.concat([ticker_features, market_features, sector_features], axis=1)
            else:
                features = pd.concat([ticker_features, market_features], axis=1)
        else:
            features = pd.concat([ticker_features, market_features], axis=1)
        # No suffix or config-based filtering, just return all features
        features = self.drop_date_column(features)
        return features

    def _format_feature_list(self, features: List[str]) -> str:
        """Format a list of features for the README."""
        if not features:
            return "No features specified in configuration."
        
        # Categorize features
        ticker_features = []
        market_features = {
            "daily_breadth": [],
            "market_volatility": [],
            "market_sentiment": []
        }
        sector_features = []
        
        # Categorize features based on their suffixes and names
        for feature in features:
            if '_daily_breadth' in feature:
                market_features["daily_breadth"].append(feature)
            elif '_market_volatility' in feature:
                market_features["market_volatility"].append(feature)
            elif '_market_sentiment' in feature:
                market_features["market_sentiment"].append(feature)
            elif feature.endswith('_sector'):
                sector_features.append(feature)
            else:
                ticker_features.append(feature)
        
        # Format the features
        formatted = []
        
        if ticker_features:
            formatted.append("### Ticker Features")
            formatted.extend([f"- `{feature}`" for feature in sorted(ticker_features)])
        
        if any(market_features.values()):
            formatted.append("\n### Market Features")
            for category, features in market_features.items():
                if features:
                    formatted.append(f"#### {category.replace('_', ' ').title()}")
                    formatted.extend([f"- `{feature}`" for feature in sorted(features)])
        
        if sector_features:
            formatted.append("\n### Sector Features")
            formatted.extend([f"- `{feature}`" for feature in sorted(sector_features)])
        
        return "\n".join(formatted)

    def _generate_readme(
        self,
        output_dir: Path,
        generation_params: Dict,
        split_stats: List[Dict],
        splits: List[Dict[str, pd.Timestamp]]
    ) -> None:
        """Generate README.md with dataset documentation."""
        # Get the actual features from the first split's train data
        train_path = output_dir / 'train.parquet'
        if train_path.exists():
            train_df = pd.read_parquet(train_path)
            actual_features = [col for col in train_df.columns if col not in ['Date', 'ticker', 'label']]
            logger.debug(f"Features in train data: {actual_features}")
        else:
            actual_features = []
        
        readme_content = f"""# Stock Price Prediction Dataset

This dataset is designed for training and evaluating machine learning models for stock price prediction. It uses a binary classification approach where the target is whether a stock's price will increase by a specified threshold within a given prediction window.

## Generation Command

The dataset was generated using the following command:

```bash
trading-advisor generate-dataset \\
    --tickers {','.join(generation_params['tickers'])} \\
    --start-date {generation_params['start_date']} \\
    --end-date {generation_params['end_date']} \\
    --target-days {generation_params['target_days']} \\
    --target-return {generation_params['target_return']} \\
    --train-months {generation_params['train_months']} \\
    --val-months {generation_params['val_months']} \\
    --test-months {generation_params['test_months']} \\
    --min-samples {generation_params['min_samples']} \\
    --output {output_dir}
```

## Dataset Organization

The dataset is organized into time-series splits to prevent data leakage and ensure realistic model evaluation:

- `train.parquet`: Combined training data from all splits
- `val.parquet`: Combined validation data from all splits
- `test.parquet`: Combined test data from all splits
- `feature_mappings.json`: Contains mappings for categorical features (e.g., ticker symbols)

## Label Definition

The binary classification labels are defined as follows:
- `1` (Positive): The stock's price increases by at least {generation_params['target_return']*100:.1f}% within {generation_params['target_days']} trading days
- `0` (Negative): The stock's price does not increase by {generation_params['target_return']*100:.1f}% within {generation_params['target_days']}

## Features

The dataset includes the following features:

{self._format_feature_list(actual_features)}

## Dataset Statistics

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Generation Parameters
- Date Range: {generation_params['start_date']} to {generation_params['end_date']}
- Target Return: {generation_params['target_return']*100:.1f}%
- Prediction Window: {generation_params['target_days']} days
- Training Period: {generation_params['train_months']} months
- Validation Period: {generation_params['val_months']} months
- Test Period: {generation_params['test_months']} months
- Minimum Samples per Ticker: {generation_params['min_samples']}
- Tickers: {', '.join(generation_params['tickers'])}

### Data Cleaning
- Training set: {generation_params['train_dropped']} rows dropped ({generation_params['train_dropped']/generation_params['train_before']*100:.1f}%) due to NaN labels
- Validation set: {generation_params['val_dropped']} rows dropped ({generation_params['val_dropped']/generation_params['val_before']*100:.1f}%) due to NaN labels
- Test set: {generation_params['test_dropped']} rows dropped ({generation_params['test_dropped']/generation_params['test_before']*100:.1f}%) due to NaN labels

### Split Statistics

"""
        # Add statistics for each split
        for i, (stats, split) in enumerate(zip(split_stats, splits)):
            readme_content += f"""#### Split {i}
- Date Ranges:
  - Train: {split['train_start'].strftime('%Y-%m-%d')} to {split['train_end'].strftime('%Y-%m-%d')}
  - Validation: {split['val_start'].strftime('%Y-%m-%d')} to {split['val_end'].strftime('%Y-%m-%d')}
  - Test: {split['test_start'].strftime('%Y-%m-%d')} to {split['test_end'].strftime('%Y-%m-%d')}
- Test Set:
  - Samples: {stats['test_samples']}
  - Positive Labels: {stats['test_positive_pct']:.1f}%
  - Features: {stats['n_features']}
- Train Set:
  - Samples: {stats['train_samples']}
  - Positive Labels: {stats['train_positive_pct']:.1f}%
  - Features: {stats['n_features']}
- Val Set:
  - Samples: {stats['val_samples']}
  - Positive Labels: {stats['val_positive_pct']:.1f}%
  - Features: {stats['n_features']}

"""

        readme_content += """## Usage Notes

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

3. Feature Mappings:
   - Categorical features are mapped to integers
   - Mappings are stored in `feature_mappings.json`
   - Apply the same mappings when using the model for prediction

4. Working with Splits:
   - The combined datasets contain data from all splits
   - To work with individual splits, filter by date ranges:
```python
# Example: Get data for split 0
split_0 = splits[0]  # Get split dates from README
split_0_train = train_df[
    (train_df['Date'] >= split_0['train_start']) & 
    (train_df['Date'] < split_0['train_end'])
]
split_0_val = val_df[
    (val_df['Date'] >= split_0['val_start']) & 
    (val_df['Date'] < split_0['val_end'])
]
split_0_test = test_df[
    (test_df['Date'] >= split_0['test_start']) & 
    (test_df['Date'] < split_0['test_end'])
]
```
   - Each split represents a different time period, allowing for:
     - Time-based cross-validation
     - Testing model performance across different market conditions
     - Validating model robustness over time
"""

        # Write README.md
        readme_path = output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"Generated README.md at {readme_path}")

    def _calculate_split_stats(self, split_data: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate statistics for a split."""
        stats = {
            'train_samples': len(split_data['train']),
            'val_samples': len(split_data['val']),
            'test_samples': len(split_data['test']),
            'train_positive_pct': (split_data['train']['label'].mean() * 100),
            'val_positive_pct': (split_data['val']['label'].mean() * 100),
            'test_positive_pct': (split_data['test']['label'].mean() * 100),
            'n_features': len(split_data['train'].columns) - 2  # Exclude 'Date' and 'label'
        }
        return stats

    def generate_dataset(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        target_days: int = 5,
        target_return: float = 0.02,
        train_months: int = 3,
        val_months: int = 1,
        test_months: int = 1,
        min_samples: int = 10,
        output: Optional[str] = None,
        force: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Generate dataset for classification."""
        self.min_samples = min_samples
        print(f"Starting dataset generation for tickers: {tickers}")
        
        # Set output directory
        output_dir = Path(output) if output else self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if files already exist
        train_path = output_dir / 'train.parquet'
        val_path = output_dir / 'val.parquet'
        test_path = output_dir / 'test.parquet'
        mappings_path = output_dir / 'feature_mappings.json'
        
        if not force and (train_path.exists() or val_path.exists() or test_path.exists() or mappings_path.exists()):
            raise ValueError(
                f"Dataset files already exist in {output_dir}. Use --force to overwrite."
            )
        
        # Generate time splits
        splits = self.generate_splits(
            start_date=start_date,
            end_date=end_date,
            train_months=train_months,
            val_months=val_months,
            test_months=test_months
        )
        print(f"Generated {len(splits)} splits.")
        
        # Calculate total tasks
        total_splits = len(splits)
        total_tickers = len(tickers)
        total_tasks = total_splits * total_tickers
        
        # Initialize progress tracking
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=Console()
        )
        
        # Process each split and ticker
        all_datasets = []
        split_stats = []
        with progress:
            task = progress.add_task("Generating datasets...", total=total_tasks)
            
            for split in splits:
                split_datasets = []
                for ticker in tickers:
                    try:
                        # Load ticker features for the entire split period
                        ticker_file = self.ticker_features_dir / f"{ticker}_features.parquet"
                        if not ticker_file.exists():
                            print(f"No features found for {ticker}")
                            progress.update(task, advance=1)
                            continue
                            
                        ticker_features = pd.read_parquet(ticker_file)
                        
                        # Ensure date is the index
                        if not isinstance(ticker_features.index, pd.DatetimeIndex):
                            if 'Date' in ticker_features.columns:
                                ticker_features = ticker_features.set_index('Date')
                            elif 'date' in ticker_features.columns:
                                ticker_features = ticker_features.set_index('date')
                            else:
                                print(f"No date column found in ticker features for {ticker}")
                                progress.update(task, advance=1)
                                continue
                        ticker_features.index = pd.to_datetime(ticker_features.index)
                        ticker_features = self.drop_date_column(ticker_features)
                        ticker_features.index.name = None
                        
                        # Filter for the split period
                        split_start = split['train_start']
                        split_end = split['test_end']
                        ticker_features = ticker_features[(ticker_features.index >= split_start) & 
                                                        (ticker_features.index <= split_end)]
                        
                        if ticker_features.empty:
                            print(f"No ticker features found for {ticker} in split period")
                            progress.update(task, advance=1)
                            continue
                        
                        # Generate labels for the entire period
                        labels = self.generate_labels(ticker_features, target_days, target_return)
                        
                        # Load market and sector features for each date
                        all_features = []
                        for date in ticker_features.index:
                            market_features = self._load_market_features(date)
                            sector_features = self._load_sector_features(ticker, date)
                            
                            if market_features.empty:
                                continue
                            ticker_row = ticker_features.loc[[date]]
                            ticker_row = self.drop_date_column(ticker_row)
                            market_features = self.drop_date_column(market_features)
                            sector_features = self.drop_date_column(sector_features)
                            features = pd.concat([ticker_row, market_features, sector_features], axis=1)
                            features = self.drop_date_column(features)
                            features['ticker'] = ticker
                            all_features.append(features)
                        
                        if not all_features:
                            print(f"No valid features found for {ticker} in split period")
                            progress.update(task, advance=1)
                            continue
                            
                        features_df = pd.concat(all_features)
                        
                        # Apply feature filtering if configuration exists
                        if self.feature_config:
                            logger.debug(f"Feature configuration: {self.feature_config}")
                            logger.debug(f"Original features: {features_df.columns.tolist()}")
                            
                            # Get all allowed features from config
                            allowed_features = set()
                            
                            # Add ticker features
                            allowed_features.update(self.feature_config['ticker_features'])
                            
                            # Add market features
                            for category in self.feature_config['market_features'].values():
                                allowed_features.update(category)
                            
                            # Add sector features
                            allowed_features.update(self.feature_config['sector_features'])
                            
                            # Filter columns
                            matching_cols = [col for col in features_df.columns if col in allowed_features]
                            # Always include 'ticker' if present
                            if 'ticker' in features_df.columns and 'ticker' not in matching_cols:
                                matching_cols.append('ticker')
                            if 'label' in features_df.columns and 'label' not in matching_cols:
                                matching_cols.append('label')
                            if 'Date' in features_df.columns and 'Date' not in matching_cols:
                                matching_cols.append('Date')
                            logger.debug(f"Allowed features: {allowed_features}")
                            logger.debug(f"Matching columns: {matching_cols}")
                            
                            if matching_cols:
                                features_df = features_df[matching_cols]
                            else:
                                logger.warning("No features remained after filtering")
                                progress.update(task, advance=1)
                                continue
                        
                        # Generate split datasets
                        split_data = self._generate_split_datasets(features_df, labels, split)
                        if split_data:
                            split_datasets.append(split_data)
                        
                    except Exception as e:
                        print(f"Error processing {ticker} for split {split['train_start']}: {str(e)}")
                    
                    progress.update(task, advance=1)
                
                if split_datasets:
                    # Combine datasets for this split
                    split_train = pd.concat([d['train'] for d in split_datasets])
                    split_val = pd.concat([d['val'] for d in split_datasets])
                    split_test = pd.concat([d['test'] for d in split_datasets])
                    
                    # Calculate statistics for this split
                    split_stats.append(self._calculate_split_stats({
                        'train': split_train,
                        'val': split_val,
                        'test': split_test
                    }))
                    
                    all_datasets.extend(split_datasets)
        
        print(f"Finished processing splits. Number of valid datasets: {len(all_datasets)}")
        
        if not all_datasets:
            print("No valid datasets were generated. Exiting.")
            raise ValueError("No valid datasets were generated")
        
        # Combine all datasets
        combined_train = pd.concat([d['train'] for d in all_datasets])
        combined_val = pd.concat([d['val'] for d in all_datasets])
        combined_test = pd.concat([d['test'] for d in all_datasets])
        
        # Remove unnecessary features
        combined_train = remove_unnecessary_features(combined_train)
        combined_val = remove_unnecessary_features(combined_val)
        combined_test = remove_unnecessary_features(combined_test)
        
        # Drop rows with NaN labels and track statistics
        train_before = len(combined_train)
        val_before = len(combined_val)
        test_before = len(combined_test)
        
        combined_train = combined_train.dropna(subset=['label'])
        combined_val = combined_val.dropna(subset=['label'])
        combined_test = combined_test.dropna(subset=['label'])
        
        train_dropped = train_before - len(combined_train)
        val_dropped = val_before - len(combined_val)
        test_dropped = test_before - len(combined_test)
        
        # Save feature mappings
        mappings = save_feature_mappings(combined_train, output_dir)
        
        # Apply mappings
        combined_train = apply_feature_mappings(combined_train, mappings)
        combined_val = apply_feature_mappings(combined_val, mappings)
        combined_test = apply_feature_mappings(combined_test, mappings)
        
        # Save datasets
        print(f"Saving train dataset to {train_path} with shape {combined_train.shape}")
        print(f"Saving val dataset to {val_path} with shape {combined_val.shape}")
        print(f"Saving test dataset to {test_path} with shape {combined_test.shape}")
        print(f"\nRows dropped due to NaN labels:")
        print(f"Training: {train_dropped} rows ({train_dropped/train_before*100:.1f}%)")
        print(f"Validation: {val_dropped} rows ({val_dropped/val_before*100:.1f}%)")
        print(f"Test: {test_dropped} rows ({test_dropped/test_before*100:.1f}%)")
        
        combined_train.to_parquet(train_path)
        combined_val.to_parquet(val_path)
        combined_test.to_parquet(test_path)
        
        # Generate README.md
        generation_params = {
            'start_date': start_date,
            'end_date': end_date,
            'target_days': target_days,
            'target_return': target_return,
            'train_months': train_months,
            'val_months': val_months,
            'test_months': test_months,
            'min_samples': min_samples,
            'tickers': tickers,
            'train_dropped': train_dropped,
            'val_dropped': val_dropped,
            'test_dropped': test_dropped,
            'train_before': train_before,
            'val_before': val_before,
            'test_before': test_before
        }
        self._generate_readme(output_dir, generation_params, split_stats, splits)
        
        print("Dataset generation complete.")
        
        return {
            'train': combined_train,
            'val': combined_val,
            'test': combined_test
        }

    def _generate_split_datasets(self, features: pd.DataFrame, labels: pd.DataFrame, split: Dict[str, pd.Timestamp]) -> Dict[str, pd.DataFrame]:
        """Generate train, validation, and test datasets for a single split using only available dates."""
        try:
            # Get available dates in features
            available_dates = features.index

            # Get intersection of available dates and split ranges
            train_mask = (available_dates >= split['train_start']) & (available_dates < split['train_end'])
            val_mask = (available_dates >= split['val_start']) & (available_dates < split['val_end'])
            test_mask = (available_dates >= split['test_start']) & (available_dates < split['test_end'])

            train_dates = available_dates[train_mask]
            val_dates = available_dates[val_mask]
            test_dates = available_dates[test_mask]

            if len(train_dates) == 0 or len(val_dates) == 0 or len(test_dates) == 0:
                logger.warning(f"No dates available in one or more splits")
                return None

            # Extract data for each split and ensure unique indices
            train_data = features.loc[train_dates].copy()
            val_data = features.loc[val_dates].copy()
            test_data = features.loc[test_dates].copy()

            # Reset index to ensure uniqueness
            train_data = train_data.reset_index()
            val_data = val_data.reset_index()
            test_data = test_data.reset_index()

            # Extract labels for each split
            train_labels = labels.loc[train_dates]
            val_labels = labels.loc[val_dates]
            test_labels = labels.loc[test_dates]

            # Reset index for labels to match features
            train_labels = train_labels.reset_index()
            val_labels = val_labels.reset_index()
            test_labels = test_labels.reset_index()

            # Check if each split has enough samples
            if len(train_data) < self.min_samples:
                logger.warning(f"Train split has insufficient samples: {len(train_data)} < {self.min_samples}")
                return None
            if len(val_data) < self.min_samples:
                logger.warning(f"Validation split has insufficient samples: {len(val_data)} < {self.min_samples}")
                return None
            if len(test_data) < self.min_samples:
                logger.warning(f"Test split has insufficient samples: {len(test_data)} < {self.min_samples}")
                return None

            # Add labels
            train_data['label'] = train_labels['label']
            val_data['label'] = val_labels['label']
            test_data['label'] = test_labels['label']

            # Ensure Date column exists
            train_data['Date'] = train_data['index']
            val_data['Date'] = val_data['index']
            test_data['Date'] = test_data['index']

            # Drop the index column
            train_data = train_data.drop(columns=['index'])
            val_data = val_data.drop(columns=['index'])
            test_data = test_data.drop(columns=['index'])

            # Log the number of samples in each split at debug level
            logger.debug(f"Generated split datasets: Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

            return {
                'train': train_data,
                'val': val_data,
                'test': test_data
            }
        except Exception as e:
            logger.error(f"Error generating split datasets: {str(e)}")
            return None

    def drop_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop the 'Date' column if it exists."""
        if 'Date' in df.columns:
            return df.drop(columns=['Date'])
        return df

    def save_feature_mappings(self, df: pd.DataFrame, output_dir: Path) -> Dict:
        """Save mappings for categorical features."""
        mappings = {}
        
        # Map tickers to integers
        unique_tickers = sorted(df['ticker'].unique())
        ticker_map = {ticker: idx for idx, ticker in enumerate(unique_tickers)}
        mappings['ticker'] = ticker_map
        
        # Save mappings
        output_path = output_dir / 'feature_mappings.json'
        with open(output_path, 'w') as f:
            json.dump(mappings, f, indent=2)
        
        logger.debug(f"Saved feature mappings to {output_path}")
        return mappings

    def apply_feature_mappings(self, df: pd.DataFrame, mappings: Dict) -> pd.DataFrame:
        """Apply mappings to categorical features."""
        df_mapped = df.copy()
        
        # Map tickers
        df_mapped['ticker'] = df_mapped['ticker'].map(mappings['ticker'])
        
        return df_mapped 