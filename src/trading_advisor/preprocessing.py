"""Data preprocessing and normalization functionality.

This module handles:
1. Feature normalization for ML models
2. Single-row inference preprocessing
3. Consistent feature engineering across training and inference
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import json

from trading_advisor.features import load_features
from trading_advisor.sector_mapping import load_sector_mapping
from trading_advisor.sector_performance import get_sp500_data

logger = logging.getLogger(__name__)

class FeaturePreprocessor:
    """Handles feature preprocessing and normalization for ML models."""
    
    def __init__(
        self,
        market_features_dir: str = "data/market_features",
        ticker_features_dir: str = "data/ticker_features",
        output_dir: Optional[str] = None
    ):
        """Initialize the preprocessor.
        
        Args:
            market_features_dir: Directory containing market feature files
            ticker_features_dir: Directory containing ticker feature files
            output_dir: Directory to save/load normalization parameters (should be dataset output directory)
        """
        self.market_features_dir = Path(market_features_dir)
        self.ticker_features_dir = Path(ticker_features_dir)
        self.output_dir = Path(output_dir) if output_dir else None
        
        # Load sector mapping
        self.sector_mapping = load_sector_mapping(str(self.market_features_dir))
        
        # Initialize scalers dictionary
        self.scalers: Dict[str, StandardScaler] = {}
        
    def fit_transform(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Fit scalers on training data and transform all datasets.
        
        Args:
            train_df: Training dataset
            val_df: Optional validation dataset
            test_df: Optional test dataset
            
        Returns:
            Tuple of (transformed_train_df, transformed_val_df, transformed_test_df)
        """
        if self.output_dir is None:
            raise ValueError("output_dir must be set to save normalization parameters")
            
        # Create scalers directory in output directory
        scalers_dir = self.output_dir / "scalers"
        scalers_dir.mkdir(parents=True, exist_ok=True)
        
        # Get numeric columns to normalize
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols.drop(['stock_splits'], errors='ignore')  # Don't normalize binary columns
        
        # Fit scalers on training data
        for col in numeric_cols:
            scaler = StandardScaler()
            scaler.fit(train_df[col].values.reshape(-1, 1))
            self.scalers[col] = scaler
            
            # Save scaler parameters
            scaler_path = scalers_dir / f"{col}_scaler.joblib"
            joblib.dump(scaler, scaler_path)
            
        # Transform all datasets
        transformed_train = self.transform(train_df)
        transformed_val = self.transform(val_df) if val_df is not None else None
        transformed_test = self.transform(test_df) if test_df is not None else None
        
        return transformed_train, transformed_val, transformed_test
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a dataset using fitted scalers.
        
        Args:
            df: Dataset to transform
            
        Returns:
            Transformed dataset
        """
        if df is None or df.empty:
            return df
            
        df = df.copy()
        
        # Apply normalization
        for col, scaler in self.scalers.items():
            if col in df.columns:
                df[col] = scaler.transform(df[col].values.reshape(-1, 1))
                
        return df
    
    def load_scalers(self) -> None:
        """Load saved scalers from disk."""
        if self.output_dir is None:
            raise ValueError("output_dir must be set to load normalization parameters")
            
        scalers_dir = self.output_dir / "scalers"
        if not scalers_dir.exists():
            raise ValueError(f"No scalers found in {scalers_dir}")
            
        for scaler_path in scalers_dir.glob("*_scaler.joblib"):
            col = scaler_path.stem.replace("_scaler", "")
            self.scalers[col] = joblib.load(scaler_path)
            
    def prepare_single_row(
        self,
        ticker: str,
        date: pd.Timestamp,
        features: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """Prepare a single row of data for inference.
        
        This function is optimized for production use and handles:
        1. Loading required features
        2. Applying consistent preprocessing
        3. Normalizing features
        
        Args:
            ticker: Ticker symbol
            date: Date for inference
            features: Optional pre-computed features
            
        Returns:
            DataFrame with a single row of normalized features
        """
        # Load ticker features
        if features is None:
            features_path = self.ticker_features_dir / f"{ticker}_features.parquet"
            if not features_path.exists():
                raise ValueError(f"No feature file found for {ticker}")
            df = pd.read_parquet(features_path)
            df.index = pd.to_datetime(df.index)
            row = df.loc[date:date]
        else:
            # Create DataFrame from provided features
            row = pd.DataFrame([features], index=[date])
            
        if row.empty:
            raise ValueError(f"No data available for {ticker} on {date}")
            
        # Add ticker column
        row['ticker'] = ticker
        
        # Get sector data
        sector = self.sector_mapping.get(ticker)
        if sector:
            sector_name = sector.lower().replace(' ', '_')
            sector_file = self.market_features_dir / 'sectors' / f"{sector_name}.parquet"
            if sector_file.exists():
                sector_df = pd.read_parquet(sector_file)
                sector_df.index = pd.to_datetime(sector_df.index)
                
                # Get most recent sector data
                available_dates = sector_df.index[sector_df.index <= date]
                if len(available_dates) > 0:
                    latest_date = available_dates.max()
                    sector_data = sector_df.loc[latest_date]
                    row.loc[date, sector_data.index] = sector_data.values
                    
        # Get S&P 500 data
        sp500_df = get_sp500_data(date.strftime('%Y-%m-%d'))
        if not sp500_df.empty:
            available_dates = sp500_df.index[sp500_df.index <= date]
            if len(available_dates) > 0:
                latest_date = available_dates.max()
                sp500_data = sp500_df.loc[latest_date]
                row.loc[date, sp500_data.index] = sp500_data.values
                
        # Handle missing values
        if 'stock_splits' in row.columns:
            row['stock_splits'] = row['stock_splits'].fillna(0)
        else:
            row['stock_splits'] = 0
            
        if 'adj_close' in row.columns:
            row['adj_close'] = row['adj_close'].fillna(row['close'])
        else:
            row['adj_close'] = row['close']
            
        if 'analyst_targets' in row.columns:
            row['analyst_targets'] = row['analyst_targets'].fillna({})
        else:
            row['analyst_targets'] = {}
            
        # Load scalers if not already loaded
        if not self.scalers:
            self.load_scalers()
            
        # Normalize features
        row = self.transform(row)
        
        return row 