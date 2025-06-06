"""
Data validation framework for LatentTrader.

This module provides validation utilities for ensuring data quality and consistency
across the LatentTrader project. It includes validators for:
- File naming conventions
- Column naming conventions
- Data types and formats
- Required columns
- Data quality metrics
"""

import os
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Base class for validation errors."""
    pass


class FileNameValidationError(ValidationError):
    """Raised when a file name doesn't follow conventions."""
    pass


class ColumnNameValidationError(ValidationError):
    """Raised when column names don't follow conventions."""
    pass


class DataTypeValidationError(ValidationError):
    """Raised when data types don't match expected types."""
    pass


class RequiredColumnError(ValidationError):
    """Raised when required columns are missing."""
    pass


class DataQualityError(ValidationError):
    """Raised when data quality checks fail."""
    pass


class DataValidator:
    """Base class for data validation."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self) -> bool:
        """Run all validations and return True if all pass."""
        raise NotImplementedError
    
    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
    
    def get_errors(self) -> List[str]:
        """Get all error messages."""
        return self.errors
    
    def get_warnings(self) -> List[str]:
        """Get all warning messages."""
        return self.warnings


class FileNameValidator(DataValidator):
    """Validates file names follow project conventions."""
    
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
    
    def validate(self) -> bool:
        """Validate file name follows conventions."""
        filename = os.path.basename(self.file_path)
        
        # Check for lowercase with underscores
        if filename != filename.lower():
            self.add_error(f"Filename should be lowercase: {filename}")
        
        if ' ' in filename:
            self.add_error(f"Filename should use underscores instead of spaces: {filename}")
        
        # Check for valid extension
        if not filename.endswith('.parquet'):
            self.add_error(f"File should have .parquet extension: {filename}")
        
        return len(self.errors) == 0


class ColumnNameValidator(DataValidator):
    """Validates column names follow project conventions."""
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df
    
    def validate(self) -> bool:
        """Validate column names follow conventions."""
        for col in self.df.columns:
            # Check for lowercase with underscores
            if col != col.lower():
                self.add_error(f"Column name should be lowercase: {col}")
            
            if ' ' in col:
                self.add_error(f"Column name should use underscores instead of spaces: {col}")
            
            # Check for valid characters
            if not all(c.isalnum() or c == '_' for c in col):
                self.add_error(f"Column name should only contain letters, numbers, and underscores: {col}")
        
        return len(self.errors) == 0


class DataTypeValidator(DataValidator):
    """Validates data types of columns."""
    
    def __init__(self, df: pd.DataFrame, expected_types: Dict[str, type]):
        super().__init__()
        self.df = df
        self.expected_types = expected_types
    
    def validate(self) -> bool:
        """Validate data types match expected types."""
        for col, expected_type in self.expected_types.items():
            if col not in self.df.columns:
                self.add_error(f"Column {col} not found in DataFrame")
                continue
            
            actual_type = self.df[col].dtype
            if expected_type == str:
                if not (np.issubdtype(actual_type, np.object_) or np.issubdtype(actual_type, np.str_)):
                    self.add_error(f"Column {col} has type {actual_type}, expected {expected_type}")
            elif not np.issubdtype(actual_type, expected_type):
                self.add_error(f"Column {col} has type {actual_type}, expected {expected_type}")
        
        return len(self.errors) == 0


class RequiredColumnValidator(DataValidator):
    """Validates presence of required columns."""
    
    def __init__(self, df: pd.DataFrame, required_columns: List[str]):
        super().__init__()
        self.df = df
        self.required_columns = required_columns
    
    def validate(self) -> bool:
        """Validate all required columns are present."""
        missing_columns = [col for col in self.required_columns if col not in self.df.columns]
        if missing_columns:
            self.add_error(f"Missing required columns: {', '.join(missing_columns)}")
        
        return len(self.errors) == 0


class DataQualityValidator(DataValidator):
    """Validates data quality metrics."""
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df
    
    def validate(self) -> bool:
        """Validate data quality metrics."""
        # Check for empty DataFrame
        if self.df.empty:
            self.add_error("DataFrame is empty")
            return False
        
        # Check for all-NaN in non-date columns
        data_cols = [col for col in self.df.columns if col != 'date']
        if data_cols and self.df[data_cols].isnull().all().all():
            self.add_error("All values in data columns are NaN")
            return False
        
        # Check for missing values
        null_counts = self.df.isnull().sum()
        if null_counts.any():
            for col in null_counts[null_counts > 0].index:
                self.add_warning(f"Column {col} has {null_counts[col]} missing values")
        
        # Check for infinite values
        inf_counts = np.isinf(self.df.select_dtypes(include=np.number)).sum()
        if inf_counts.any():
            for col in inf_counts[inf_counts > 0].index:
                self.add_error(f"Column {col} has {inf_counts[col]} infinite values")
        
        # Check for negative values in 'volume' columns
        for col in self.df.columns:
            if 'volume' in col.lower():
                if (self.df[col] < 0).any():
                    self.add_error(f"Column {col} has negative values")
        
        # Check for extremely large values in numeric columns (arbitrary threshold: 1e12)
        for col in self.df.select_dtypes(include=np.number).columns:
            if (self.df[col].abs() > 1e12).any():
                self.add_error(f"Column {col} has extremely large values (>|1e12|)")
        
        # Check for duplicate rows
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            self.add_warning(f"Found {duplicates} duplicate rows")
        
        return len(self.errors) == 0


class MarketSentimentValidator(DataValidator):
    """Validates market sentiment data files."""
    
    REQUIRED_COLUMNS = [
        'date',
        'sentiment_ma5',
        'sentiment_ma20',
        'sentiment_momentum',
        'sentiment_volatility',
        'sentiment_zscore'
    ]
    
    EXPECTED_TYPES = {
        'date': np.datetime64,
        'sentiment_ma5': np.float64,
        'sentiment_ma20': np.float64,
        'sentiment_momentum': np.float64,
        'sentiment_volatility': np.float64,
        'sentiment_zscore': np.float64
    }
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df
    
    def validate(self) -> bool:
        """Validate market sentiment data."""
        # Check required columns
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in self.df.columns]
        if missing_cols:
            self.add_error(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Check data types
        for col, expected_type in self.EXPECTED_TYPES.items():
            if col in self.df.columns:
                actual_type = self.df[col].dtype
                if not np.issubdtype(actual_type, expected_type):
                    self.add_error(f"Column {col} has type {actual_type}, expected {expected_type}")
        
        # Check for missing values
        null_counts = self.df.isnull().sum()
        if null_counts.any():
            for col in null_counts[null_counts > 0].index:
                self.add_warning(f"Column {col} has {null_counts[col]} missing values")
        
        return len(self.errors) == 0


class GDELTValidator(DataValidator):
    """Validates GDELT data files."""
    
    REQUIRED_COLUMNS = ['date', 'avg_tone']
    
    EXPECTED_TYPES = {
        'date': np.datetime64,
        'avg_tone': np.float64
    }
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df
    
    def validate(self) -> bool:
        """Validate GDELT data."""
        # Check required columns
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in self.df.columns]
        if missing_cols:
            self.add_error(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Check data types
        for col, expected_type in self.EXPECTED_TYPES.items():
            if col in self.df.columns:
                actual_type = self.df[col].dtype
                if not np.issubdtype(actual_type, expected_type):
                    self.add_error(f"Column {col} has type {actual_type}, expected {expected_type}")
        
        # Check for missing values
        null_counts = self.df.isnull().sum()
        if null_counts.any():
            for col in null_counts[null_counts > 0].index:
                self.add_warning(f"Column {col} has {null_counts[col]} missing values")
        
        return len(self.errors) == 0


class SectorDataValidator(DataValidator):
    """Validates sector data files."""
    
    REQUIRED_COLUMNS = [
        'date',
        '{sector}_price',
        '{sector}_volatility',
        '{sector}_volume',
        '{sector}_returns_1d',
        '{sector}_returns_5d',
        '{sector}_returns_20d',
        '{sector}_momentum_5d',
        '{sector}_momentum_20d',
        '{sector}_relative_strength'
    ]
    
    EXPECTED_TYPES = {
        'date': np.datetime64,
        '{sector}_price': np.float64,
        '{sector}_volatility': np.float64,
        '{sector}_volume': (np.float64, np.int64),  # Allow both float and int for volume
        '{sector}_returns_1d': np.float64,
        '{sector}_returns_5d': np.float64,
        '{sector}_returns_20d': np.float64,
        '{sector}_momentum_5d': np.float64,
        '{sector}_momentum_20d': np.float64,
        '{sector}_relative_strength': np.float64
    }
    
    # Expected missing values for time series calculations
    EXPECTED_MISSING = {
        '{sector}_returns_1d': 1,
        '{sector}_returns_5d': 5,
        '{sector}_returns_20d': 20,
        '{sector}_momentum_5d': 5,
        '{sector}_momentum_20d': 20,
        '{sector}_relative_strength': 39  # Based on validation results
    }
    
    def __init__(self, df: pd.DataFrame, sector: str):
        super().__init__()
        self.df = df
        self.sector = sector
    
    def validate(self) -> bool:
        """Validate sector data."""
        # Format required columns with sector name
        required_cols = [col.format(sector=self.sector) for col in self.REQUIRED_COLUMNS]
        expected_types = {col.format(sector=self.sector): dtype for col, dtype in self.EXPECTED_TYPES.items()}
        expected_missing = {col.format(sector=self.sector): count for col, count in self.EXPECTED_MISSING.items()}
        
        # Check required columns
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            self.add_error(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Check data types
        for col, expected_type in expected_types.items():
            if col in self.df.columns:
                actual_type = self.df[col].dtype
                if isinstance(expected_type, tuple):
                    if not any(np.issubdtype(actual_type, t) for t in expected_type):
                        self.add_error(f"Column {col} has type {actual_type}, expected one of {expected_type}")
                elif not np.issubdtype(actual_type, expected_type):
                    self.add_error(f"Column {col} has type {actual_type}, expected {expected_type}")
        
        # Check for missing values
        null_counts = self.df.isnull().sum()
        if null_counts.any():
            for col in null_counts[null_counts > 0].index:
                if col in expected_missing:
                    if null_counts[col] > expected_missing[col]:
                        self.add_warning(f"Column {col} has {null_counts[col]} missing values, expected at most {expected_missing[col]}")
                else:
                    self.add_warning(f"Column {col} has {null_counts[col]} missing values")
        
        return len(self.errors) == 0


def validate_parquet_file(file_path: str, expected_types: Dict[str, type] = None, required_columns: List[str] = None) -> bool:
    """Validate a Parquet file.
    
    Args:
        file_path: Path to the Parquet file
        expected_types: Dictionary mapping column names to expected types
        required_columns: List of required column names
        
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Validate file name
        file_validator = FileNameValidator(file_path)
        if not file_validator.validate():
            return False
        
        # Read the file
        df = pd.read_parquet(file_path)
        
        # Validate column names
        col_validator = ColumnNameValidator(df)
        if not col_validator.validate():
            return False
        
        # Validate data types if provided
        if expected_types:
            type_validator = DataTypeValidator(df, expected_types)
            if not type_validator.validate():
                return False
        
        # Validate required columns if provided
        if required_columns:
            req_validator = RequiredColumnValidator(df, required_columns)
            if not req_validator.validate():
                return False
        
        # Validate data quality
        quality_validator = DataQualityValidator(df)
        if not quality_validator.validate():
            return False
        
        return True
    except Exception as e:
        return False


# Define expected types for market features
MARKET_FEATURE_TYPES = {
    'date': np.datetime64,
    'market_volatility_daily_volatility': np.floating,
    'market_volatility_weekly_volatility': np.floating,
    'market_volatility_monthly_volatility': np.floating,
    'market_volatility_avg_correlation': np.floating,
    'market_volatility_ticker': str
}

# Define required columns for market features
MARKET_FEATURE_REQUIRED = ['date', 'market_volatility_daily_volatility']

# Define expected types for daily breadth
BREADTH_FEATURE_TYPES = {
    'date': np.datetime64,
    'daily_breadth_adv_dec_line': np.floating,
    'daily_breadth_new_highs': np.integer,
    'daily_breadth_new_lows': np.integer,
    'daily_breadth_above_ma20': np.floating,
    'daily_breadth_above_ma50': np.floating,
    'daily_breadth_rsi_bullish': np.floating,
    'daily_breadth_rsi_oversold': np.floating,
    'daily_breadth_rsi_overbought': np.floating,
    'daily_breadth_macd_bullish': np.floating
}

# Define required columns for daily breadth
BREADTH_FEATURE_REQUIRED = ['date', 'daily_breadth_adv_dec_line']

# Define expected types for market sentiment
SENTIMENT_FEATURE_TYPES = {
    'date': np.datetime64,
    'market_sentiment_ma5': np.floating,
    'market_sentiment_ma20': np.floating,
    'market_sentiment_momentum': np.floating,
    'market_sentiment_volatility': np.floating,
    'market_sentiment_zscore': np.floating
}

# Define required columns for market sentiment
SENTIMENT_FEATURE_REQUIRED = ['date', 'market_sentiment_ma5']

def validate_market_features(data_dir: str = 'data/market_features'):
    """Validate all market feature files."""
    data_dir = Path(data_dir)
    all_valid = True
    
    # Validate market volatility
    if not validate_parquet_file(
        data_dir / 'market_volatility.parquet',
        expected_types=MARKET_FEATURE_TYPES,
        required_columns=MARKET_FEATURE_REQUIRED
    ):
        all_valid = False
    
    # Validate daily breadth
    if not validate_parquet_file(
        data_dir / 'daily_breadth.parquet',
        expected_types=BREADTH_FEATURE_TYPES,
        required_columns=BREADTH_FEATURE_REQUIRED
    ):
        all_valid = False
    
    # Validate market sentiment
    if not validate_parquet_file(
        data_dir / 'market_sentiment.parquet',
        expected_types=SENTIMENT_FEATURE_TYPES,
        required_columns=SENTIMENT_FEATURE_REQUIRED
    ):
        all_valid = False
    
    # Validate GDELT raw data
    if not validate_parquet_file(
        data_dir / 'gdelt_raw.parquet',
        expected_types={'date': np.datetime64},
        required_columns=['date']
    ):
        all_valid = False
    
    # Validate sector files
    sectors_dir = data_dir / 'sectors'
    if sectors_dir.exists():
        for sector_file in sectors_dir.glob('*.parquet'):
            if not validate_parquet_file(
                str(sector_file),
                expected_types={
                    'date': np.datetime64,
                    'sector_price': np.floating,
                    'sector_volatility': np.floating,
                    'sector_volume': np.floating,
                    'sector_returns_1d': np.floating,
                    'sector_returns_5d': np.floating,
                    'sector_returns_20d': np.floating,
                    'sector_momentum_5d': np.floating,
                    'sector_momentum_20d': np.floating
                },
                required_columns=['date', 'sector_price', 'sector_volume']
            ):
                all_valid = False
    
    return all_valid 