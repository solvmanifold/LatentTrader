"""Model runner for executing trading models and saving outputs."""

import os
import logging
import json
from typing import List, Optional, Dict, Any, Set, Union
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler

from .registry import registry
from .base import BaseTradingModel

logger = logging.getLogger(__name__)

class ModelRunner:
    """Runner for executing trading models and saving outputs."""
    
    REQUIRED_MARKET_FEATURES = {
        'daily_breadth': ['adv_dec_line', 'new_highs', 'new_lows', 'above_ma20', 'above_ma50'],
        'market_volatility': ['vix', 'vix_ma20', 'market_volatility'],
        'market_sentiment': ['sentiment_ma5', 'sentiment_ma20', 'sentiment_momentum']
    }
    
    def __init__(
        self,
        output_dir: str = "model_outputs",
        market_features_dir: str = "data/market_features",
        sector_mapping_file: str = "data/market_features/sector_mapping.json",
        model_dir: str = "models"
    ):
        """Initialize the model runner.
        
        Args:
            output_dir: Directory to save model outputs
            market_features_dir: Directory containing market feature files
            sector_mapping_file: Path to sector mapping file
            model_dir: Directory to save trained models
        """
        self.output_dir = Path(output_dir)
        self.market_features_dir = Path(market_features_dir)
        self.sector_mapping_file = Path(sector_mapping_file)
        self.model_dir = Path(model_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Load sector mapping
        with open(self.sector_mapping_file, 'r') as f:
            sector_data = json.load(f)
            self.sector_mapping = pd.DataFrame([
                {'ticker': ticker, 'sector': sector}
                for sector, tickers in sector_data.items()
                for ticker in tickers
            ])
        
        # Initialize feature scaler
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using StandardScaler.
        
        Args:
            df: DataFrame containing features
            
        Returns:
            DataFrame with normalized features
        """
        if self.feature_columns is None:
            # First time normalization, fit the scaler
            # Get numeric columns only, excluding date and ticker
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            self.feature_columns = [col for col in numeric_cols 
                                  if col not in ['date', 'ticker']]
            self.scaler.fit(df[self.feature_columns])
        
        # Create a copy to avoid modifying the original
        df_normalized = df.copy()
        
        # Only normalize the feature columns
        df_normalized[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        
        return df_normalized
    
    def train_model(
        self,
        model_name: str,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        **model_kwargs
    ) -> None:
        """Train a model on the provided data.
        
        Args:
            model_name: Name of the model to train
            train_data: DataFrame containing training data
            val_data: Optional DataFrame containing validation data
            **model_kwargs: Additional arguments to pass to the model
        """
        # Create model instance
        model = registry.create_model(model_name, **model_kwargs)
        
        # Normalize features
        train_data = self._normalize_features(train_data)
        if val_data is not None:
            val_data = self._normalize_features(val_data)
        
        # Train model
        model.train(train_data, val_data)
        
        # Save model
        model_path = self.model_dir / f"{model_name}.pkl"
        model.save(model_path)
        logger.info(f"Saved trained model to {model_path}")
    
    def run_model(
        self,
        model_name: str,
        tickers: List[str],
        features_df: pd.DataFrame,
        date: Optional[str] = None,
        batch_size: Optional[int] = None,
        **model_kwargs
    ) -> Dict[str, Any]:
        """Run a model on the given tickers and features.
        
        Args:
            model_name: Name of the model to run
            tickers: List of tickers to run the model on
            features_df: DataFrame containing features for all tickers
            date: Optional date to run the model for (defaults to latest date)
            batch_size: Optional batch size for processing multiple dates
            **model_kwargs: Additional arguments to pass to the model
            
        Returns:
            Dictionary containing model outputs for each ticker
        """
        # Create model instance
        model = registry.create_model(model_name, **model_kwargs)
        
        # Load trained model if it exists
        model_path = self.model_dir / f"{model_name}.pkl"
        if model_path.exists():
            model.load(model_path)
        
        # Ensure features_df has date index
        if not isinstance(features_df.index, pd.DatetimeIndex):
            if 'date' in features_df.columns:
                features_df.set_index('date', inplace=True)
            features_df.index = pd.to_datetime(features_df.index)
        
        # Get date if not provided
        if date is None:
            date = features_df.index.max().strftime('%Y-%m-%d')
        else:
            # Convert date string to datetime
            target_date = pd.to_datetime(date)
            # Filter features for the requested date
            features_df = features_df[features_df.index == target_date]
            if features_df.empty:
                logger.error(f"No features found for date {date}")
                return {}
        
        # Load market features
        market_features = self._load_market_features(date)
        
        # Initialize results
        results = {}
        
        # Process in batches if specified
        if batch_size is not None:
            # Group by date and process in batches
            dates = features_df.index.unique()
            for i in range(0, len(dates), batch_size):
                batch_dates = dates[i:i + batch_size]
                batch_df = features_df[features_df.index.isin(batch_dates)]
                
                # Process batch
                batch_results = self._process_batch(
                    model, batch_df, market_features, tickers, model_name
                )
                results.update(batch_results)
        else:
            # Process single date
            results = self._process_batch(
                model, features_df, market_features, tickers, model_name
            )
        
        return results
    
    def _process_batch(
        self,
        model: BaseTradingModel,
        features_df: pd.DataFrame,
        market_features: pd.DataFrame,
        tickers: List[str],
        model_name: str
    ) -> Dict[str, Any]:
        """Process a batch of data.
        
        Args:
            model: Model instance
            features_df: DataFrame containing features
            market_features: DataFrame containing market features
            tickers: List of tickers to process
            model_name: Name of the model being run
            
        Returns:
            Dictionary containing model outputs
        """
        results = {}
        
        # Ensure features_df has date index
        if not isinstance(features_df.index, pd.DatetimeIndex):
            if 'date' in features_df.columns:
                features_df.set_index('date', inplace=True)
            features_df.index = pd.to_datetime(features_df.index)
            
        # Get the date for this batch
        batch_date = features_df.index[0]
        batch_date_str = batch_date.strftime('%Y-%m-%d')
        logger.info(f"Processing batch for date: {batch_date_str}")
        
        # Run model for each ticker
        for ticker in tickers:
            try:
                # Get features for this ticker
                ticker_features = features_df[features_df['ticker'] == ticker]
                if ticker_features.empty:
                    logger.warning(f"No features found for {ticker}")
                    continue
                
                # Reset index to make date a column for merging
                ticker_features = ticker_features.reset_index()
                
                # Join with market features if available
                if not market_features.empty:
                    ticker_features = ticker_features.merge(
                        market_features,
                        on='date',
                        how='left'
                    )
                
                # Load and join sector features
                sector_features = self._load_sector_features(ticker, batch_date_str)
                if not sector_features.empty:
                    ticker_features = ticker_features.merge(
                        sector_features,
                        on='date',
                        how='left'
                    )
                
                # Normalize features
                ticker_features = self._normalize_features(ticker_features)
                
                # Run model
                predictions = model.predict(ticker_features)
                
                # Save results
                results[ticker] = predictions
                
                # Save to Parquet file
                self._save_model_output(model_name, ticker, batch_date_str, predictions)
                logger.info(f"Successfully processed {ticker} for {batch_date_str}")
                
            except Exception as e:
                logger.error(f"Error running {model_name} on {ticker}: {str(e)}")
                continue
        
        return results
    
    def _validate_market_features(self, df: pd.DataFrame, feature_type: str) -> bool:
        """Validate that required market features are present and non-NaN.
        
        Args:
            df: DataFrame containing market features
            feature_type: Type of market features (e.g., 'daily_breadth')
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        if feature_type not in self.REQUIRED_MARKET_FEATURES:
            logger.warning(f"Unknown market feature type: {feature_type}")
            return True
            
        required_cols = self.REQUIRED_MARKET_FEATURES[feature_type]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns in {feature_type}: {missing_cols}")
            return False
            
        # Check for NaN values in the most recent row
        if not df.empty:
            latest_row = df.iloc[-1]
            nan_cols = latest_row[required_cols].columns[latest_row[required_cols].isna()]
            if not nan_cols.empty:
                logger.error(f"NaN values found in {feature_type} for columns: {nan_cols.tolist()}")
                return False
                
        return True
    
    def _handle_nan_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle NaN values in market features by using the previous row's values.
        
        Args:
            df: DataFrame containing market features
            
        Returns:
            DataFrame with NaN values filled from previous rows
        """
        if df.empty:
            return df
            
        # Forward fill NaN values
        df = df.fillna(method='ffill')
        
        # If there are still NaN values (e.g., at the start of the DataFrame), backward fill
        df = df.fillna(method='bfill')
        
        return df
    
    def _load_sector_features(self, ticker: str, date: str) -> pd.DataFrame:
        """Load sector-specific features for a ticker.
        
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
    
    def _load_market_features(self, date: str) -> pd.DataFrame:
        """Load and combine all market features for a given date.
        
        Args:
            date: Date to load features for
            
        Returns:
            DataFrame containing all market features for the date
        """
        market_features = {}
        
        # Load each market feature file
        for feature_file in self.market_features_dir.glob("*.parquet"):
            if feature_file.name == "metadata":
                continue
                
            try:
                df = pd.read_parquet(feature_file)
                if 'date' in df.columns:
                    # Get features for the date and previous day
                    df = df[df['date'] <= date].sort_values('date')
                    if not df.empty:
                        # Handle NaN values in the most recent row
                        df = self._handle_nan_values(df)
                        # Get the row for our target date
                        df = df[df['date'] == date]
                        if not df.empty:
                            # Validate features
                            feature_type = feature_file.stem
                            if self._validate_market_features(df, feature_type):
                                market_features[feature_type] = df
                            else:
                                logger.error(f"Market feature validation failed for {feature_type}")
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
    
    def _save_model_output(
        self,
        model_name: str,
        ticker: str,
        date: str,
        predictions: Dict[str, Any]
    ):
        """Save model output to a Parquet file.
        
        Args:
            model_name: Name of the model
            ticker: Ticker symbol
            date: Date of the predictions
            predictions: Model predictions to save
        """
        # Create model output directory with date subdirectory
        model_dir = self.output_dir / model_name / date
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output file path
        output_file = model_dir / f"{ticker}.parquet"
        
        # Convert predictions to DataFrame
        df = pd.DataFrame([{
            'date': date,
            'ticker': ticker,
            **predictions
        }])
        
        # Save to Parquet
        df.to_parquet(output_file)
        logger.info(f"Saved {model_name} output for {ticker} on {date}")
    
    def get_model_output(
        self,
        model_name: str,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Get model outputs for a ticker.
        
        Args:
            model_name: Name of the model
            ticker: Ticker symbol
            start_date: Optional start date (inclusive)
            end_date: Optional end date (inclusive)
            
        Returns:
            DataFrame containing model outputs
        """
        # Get all date directories
        model_dir = self.output_dir / model_name
        if not model_dir.exists():
            return pd.DataFrame()
            
        # Get all date directories and convert names to datetime
        date_dirs = []
        for d in model_dir.iterdir():
            if d.is_dir():
                try:
                    date = pd.to_datetime(d.name)
                    date_dirs.append((d, date))
                except ValueError:
                    logger.warning(f"Skipping invalid date directory: {d.name}")
                    continue
        
        # Filter by date range if provided
        if start_date:
            start_date = pd.to_datetime(start_date)
            date_dirs = [(d, date) for d, date in date_dirs if date >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            date_dirs = [(d, date) for d, date in date_dirs if date <= end_date]
            
        # Sort dates
        date_dirs.sort(key=lambda x: x[1])
        
        # Collect all predictions
        dfs = []
        for date_dir, _ in date_dirs:
            output_file = date_dir / f"{ticker}.parquet"
            if output_file.exists():
                df = pd.read_parquet(output_file)
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
            
        # Combine all predictions
        return pd.concat(dfs, ignore_index=True) 