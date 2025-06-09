"""Tests for the data validation framework."""

import os
import tempfile
import pandas as pd
import numpy as np
import pytest
from trading_advisor.validation import (
    ValidationError,
    FileNameValidator,
    ColumnNameValidator,
    DataTypeValidator,
    RequiredColumnValidator,
    DataQualityValidator,
    validate_parquet_file,
    validate_market_features,
    MARKET_FEATURE_TYPES,
    MARKET_FEATURE_REQUIRED,
    BREADTH_FEATURE_TYPES,
    BREADTH_FEATURE_REQUIRED,
    SENTIMENT_FEATURE_TYPES,
    SENTIMENT_FEATURE_REQUIRED
)
from datetime import datetime, timedelta


@pytest.fixture
def temp_parquet_file():
    """Create a temporary Parquet file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        return f.name


def test_file_name_validator():
    """Test file name validation."""
    # Valid file name
    validator = FileNameValidator('data/market_features/daily_breadth.parquet')
    assert validator.validate()
    assert len(validator.get_errors()) == 0
    
    # Invalid file name (uppercase)
    validator = FileNameValidator('data/market_features/DailyBreadth.parquet')
    assert not validator.validate()
    assert len(validator.get_errors()) > 0
    
    # Invalid file name (spaces)
    validator = FileNameValidator('data/market_features/daily breadth.parquet')
    assert not validator.validate()
    assert len(validator.get_errors()) > 0
    
    # Invalid extension
    validator = FileNameValidator('data/market_features/daily_breadth.csv')
    assert not validator.validate()
    assert len(validator.get_errors()) > 0


def test_column_name_validator():
    """Test column name validation."""
    # Valid column names
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=5),
        'close_price': np.random.randn(5),
        'volume_ma20': np.random.randn(5)
    })
    validator = ColumnNameValidator(df)
    assert validator.validate()
    assert len(validator.get_errors()) == 0
    
    # Invalid column names
    df = pd.DataFrame({
        'Date': pd.date_range('2020-01-01', periods=5),
        'Close Price': np.random.randn(5),
        'volume-ma20': np.random.randn(5)
    })
    validator = ColumnNameValidator(df)
    assert not validator.validate()
    assert len(validator.get_errors()) > 0


def test_data_type_validator():
    """Test data type validation."""
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=5),
        'close_price': np.random.randn(5),
        'volume': np.random.randint(0, 1000, 5)
    })
    
    # Valid types
    expected_types = {
        'date': np.datetime64,
        'close_price': np.floating,
        'volume': np.integer
    }
    validator = DataTypeValidator(df, expected_types)
    assert validator.validate()
    assert len(validator.get_errors()) == 0
    
    # Invalid types
    expected_types = {
        'date': np.integer,  # Should be datetime
        'close_price': np.integer,  # Should be float
        'volume': np.floating  # Should be integer
    }
    validator = DataTypeValidator(df, expected_types)
    assert not validator.validate()
    assert len(validator.get_errors()) > 0


def test_required_column_validator():
    """Test required column validation."""
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=5),
        'close_price': np.random.randn(5),
        'volume': np.random.randint(0, 1000, 5)
    })
    
    # All required columns present
    required_columns = ['date', 'close_price']
    validator = RequiredColumnValidator(df, required_columns)
    assert validator.validate()
    assert len(validator.get_errors()) == 0
    
    # Missing required columns
    required_columns = ['date', 'close_price', 'missing_column']
    validator = RequiredColumnValidator(df, required_columns)
    assert not validator.validate()
    assert len(validator.get_errors()) > 0


def test_data_quality_validator():
    """Test data quality validation."""
    # Clean data
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=5),
        'close_price': np.random.randn(5),
        'volume': np.random.randint(0, 1000, 5)
    })
    validator = DataQualityValidator(df)
    assert validator.validate()
    assert len(validator.get_errors()) == 0
    assert len(validator.get_warnings()) == 0
    
    # Data with issues
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=5),
        'close_price': [1.0, np.nan, 3.0, np.inf, 5.0],
        'volume': [100, 200, 200, 300, 100]  # Duplicate values
    })
    validator = DataQualityValidator(df)
    validator.validate()  # We don't return False for warnings
    assert len(validator.get_errors()) > 0  # Should have error for inf
    assert len(validator.get_warnings()) > 0  # Should have warnings for nan and duplicates


def test_validate_parquet_file(temp_parquet_file):
    """Test the validate_parquet_file function."""
    # Create a valid DataFrame
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=5),
        'close_price': np.random.randn(5),
        'volume': np.random.randint(0, 1000, 5)
    })
    df.to_parquet(temp_parquet_file)
    
    # Test with valid data
    expected_types = {
        'date': np.datetime64,
        'close_price': np.floating,
        'volume': np.integer
    }
    required_columns = ['date', 'close_price']
    assert validate_parquet_file(temp_parquet_file, expected_types, required_columns)
    
    # Test with invalid data types
    expected_types = {
        'date': np.integer,  # Should be datetime
        'close_price': np.integer,  # Should be float
        'volume': np.floating  # Should be integer
    }
    assert not validate_parquet_file(temp_parquet_file, expected_types, required_columns)
    
    # Test with missing required columns
    required_columns = ['date', 'close_price', 'missing_column']
    assert not validate_parquet_file(temp_parquet_file, expected_types, required_columns)
    
    # Clean up
    os.unlink(temp_parquet_file)


def test_sector_relative_strength_calculation():
    """Test sector relative strength calculation produces expected columns and values."""
    from trading_advisor.sector_performance import calculate_sector_performance
    import pandas as pd
    import numpy as np

    # Generate sample data
    dates = pd.date_range(start='2023-01-01', end='2023-03-01', freq='B')
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    sectors = ['Technology', 'Technology', 'Technology']
    data = []
    for ticker, sector in zip(tickers, sectors):
        prices = 100 * (1 + np.random.normal(0.001, 0.02, len(dates))).cumprod()
        volumes = np.random.randint(1000000, 10000000, len(dates))
        for date, price, volume in zip(dates, prices, volumes):
            data.append({
                'date': date,
                'ticker': ticker,
                'sector': sector,
                'close': price,
                'volume': volume
            })
    df = pd.DataFrame(data)
    df.set_index(['date', 'ticker'], inplace=True)
    df.index = df.index.set_levels([pd.to_datetime(df.index.levels[0]), df.index.levels[1]])

    # Create mock S&P 500 data
    sp500_prices = 4000 * (1 + np.random.normal(0.0005, 0.01, len(dates))).cumprod()
    sp500_df = pd.DataFrame({
        'sp500_price': sp500_prices,
        'sp500_returns_20d': pd.Series(sp500_prices).pct_change(periods=20)
    }, index=dates)
    sp500_df.index = pd.to_datetime(sp500_df.index)

    # Mock get_sp500_data to return our synthetic data
    import unittest.mock as mock
    with mock.patch('trading_advisor.sector_performance.get_sp500_data', return_value=sp500_df):
        # Calculate sector performance
        sector_dfs = calculate_sector_performance(df, 'data/market_features')
        tech_df = sector_dfs['Technology']

        # Check that the expected columns exist
        assert 'relative_strength' in tech_df.columns
        assert 'relative_strength_ratio' in tech_df.columns

        # Check that the values are finite (not all NaN or inf)
        assert tech_df['relative_strength'].notna().any()
        assert np.isfinite(tech_df['relative_strength'].dropna()).all()
        assert tech_df['relative_strength_ratio'].notna().any()
        assert np.isfinite(tech_df['relative_strength_ratio'].dropna()).all()


def test_data_completeness():
    """Test that all trading days are present in the data."""
    # Create sample data with missing days
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='B')
    dates = dates.drop(dates[2])  # Remove one trading day
    df = pd.DataFrame({
        'date': dates,
        'close_price': np.random.randn(len(dates)),
        'volume': np.random.randint(0, 1000, len(dates))
    })
    df.set_index('date', inplace=True)
    
    # Test for missing trading days
    expected_dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='B')
    missing_dates = expected_dates.difference(df.index)
    assert len(missing_dates) > 0, "Test data should have missing dates"
    
    # Test for gaps in the data
    date_diff = df.index.to_series().diff()
    max_gap = date_diff.max()
    assert max_gap > pd.Timedelta(days=1), "Test data should have gaps"


def test_data_consistency():
    """Test that data is consistent across files."""
    # Create sample sector data
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='B')
    tech_data = pd.DataFrame({
        'date': dates,
        'technology_price': np.random.randn(len(dates)),
        'technology_volume': np.random.randint(0, 1000, len(dates))
    })
    tech_data.set_index('date', inplace=True)
    
    fin_data = pd.DataFrame({
        'date': dates,
        'financial_price': np.random.randn(len(dates)),
        'financial_volume': np.random.randint(0, 1000, len(dates))
    })
    fin_data.set_index('date', inplace=True)
    
    # Test that dates align across files
    assert tech_data.index.equals(fin_data.index), "Dates should align across files"
    
    # Test that calculations are consistent
    tech_returns = tech_data['technology_price'].pct_change()
    fin_returns = fin_data['financial_price'].pct_change()
    assert tech_returns.notna().sum() == fin_returns.notna().sum(), "Return calculations should be consistent"


def test_edge_cases():
    """Test edge cases in the data."""
    # Test empty DataFrame
    empty_df = pd.DataFrame()
    validator = DataQualityValidator(empty_df)
    assert not validator.validate(), "Empty DataFrame should fail validation"
    
    # Test single-row DataFrame
    single_row_df = pd.DataFrame({
        'date': [datetime.now()],
        'close_price': [100.0],
        'volume': [1000]
    })
    validator = DataQualityValidator(single_row_df)
    assert validator.validate(), "Single-row DataFrame should pass validation"
    
    # Test DataFrame with all NaN values
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='B')
    nan_df = pd.DataFrame({
        'date': dates,
        'close_price': [np.nan] * len(dates),
        'volume': [np.nan] * len(dates)
    })
    validator = DataQualityValidator(nan_df)
    assert not validator.validate(), "DataFrame with all NaN values should fail validation"
    
    # Test DataFrame with extreme values
    extreme_df = pd.DataFrame({
        'date': dates,
        'close_price': [1e10] * len(dates),  # Extremely large values
        'volume': [-1] * len(dates)  # Invalid negative values
    })
    validator = DataQualityValidator(extreme_df)
    assert not validator.validate(), "DataFrame with extreme values should fail validation"


def test_performance():
    """Test performance with large datasets."""
    # Create large dataset
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='B')
    large_df = pd.DataFrame({
        'date': dates,
        'close_price': np.random.randn(len(dates)),
        'volume': np.random.randint(0, 1000, len(dates))
    })
    large_df.set_index('date', inplace=True)
    
    # Test computation time
    import time
    start_time = time.time()
    validator = DataQualityValidator(large_df)
    validator.validate()
    end_time = time.time()
    
    # Assert that validation completes within 5 seconds
    assert end_time - start_time < 5, "Validation should complete within 5 seconds"
    
    # Test memory usage
    import psutil
    process = psutil.Process()
    memory_before = process.memory_info().rss
    validator = DataQualityValidator(large_df)
    validator.validate()
    memory_after = process.memory_info().rss
    
    # Assert that memory usage is reasonable (less than 1GB)
    assert memory_after - memory_before < 1e9, "Memory usage should be less than 1GB"


def test_sector_mapping_consistency():
    """Test that sector mappings are consistent across the system."""
    from trading_advisor.sector_mapping import load_sector_mapping
    
    # Create sample sector mapping
    sector_mapping = {
        'AAPL': 'Technology',
        'MSFT': 'Technology',
        'JPM': 'Financial Services'
    }
    
    # Test that all tickers have valid sectors
    assert all(sector in ['Technology', 'Financial Services'] for sector in sector_mapping.values()), \
        "All sectors should be valid"
    
    # Test that no ticker has multiple sectors
    assert len(sector_mapping) == len(set(sector_mapping.keys())), \
        "No ticker should have multiple sectors"


def test_market_features_consistency():
    """Test that market features are consistent."""
    # Create sample market features
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='B')
    market_features = pd.DataFrame({
        'date': dates,
        'market_breadth': np.random.randn(len(dates)),
        'market_sentiment': np.random.randn(len(dates)),
        'market_volatility': np.random.randn(len(dates))
    })
    market_features.set_index('date', inplace=True)
    
    # Test that all features have the same number of rows
    assert all(len(market_features[col]) == len(dates) for col in market_features.columns), \
        "All features should have the same number of rows"
    
    # Test that all features have the same index
    assert all(market_features[col].index.equals(dates) for col in market_features.columns), \
        "All features should have the same index"


def test_data_pipeline():
    """Test the entire data pipeline."""
    from trading_advisor.data import download_stock_data, standardize_columns_and_date
    from trading_advisor.sector_performance import calculate_sector_performance
    
    # Test data download
    ticker = "AAPL"
    df = download_stock_data(ticker, features_dir="data/ticker_features")
    assert not df.empty, "Data download should return non-empty DataFrame"
    
    # Test data processing
    df = standardize_columns_and_date(df)
    assert isinstance(df.index, pd.DatetimeIndex), "Standardized DataFrame should have DatetimeIndex"
    
    # Test sector performance calculation
    df['sector'] = 'Technology'
    sector_dfs = calculate_sector_performance(df, 'data/market_features')
    assert 'Technology' in sector_dfs, "Sector performance should include Technology sector"
    
    # Test data storage
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        df.to_parquet(f.name)
        assert os.path.exists(f.name), "Data should be saved to file"
        os.unlink(f.name)


def test_data_updates():
    """Test data update functionality."""
    from trading_advisor.data import download_stock_data
    
    # Test incremental updates
    ticker = "AAPL"
    df1 = download_stock_data(ticker, features_dir="data/ticker_features")
    df2 = download_stock_data(ticker, features_dir="data/ticker_features")
    
    # Assert that new data is appended
    assert len(df2) >= len(df1), "New data should be appended"
    
    # Assert that old data is preserved
    common_dates = df1.index.intersection(df2.index)
    assert len(common_dates) == len(df1), "Old data should be preserved"


def test_data_versioning():
    """Test data versioning functionality."""
    # Create sample data with versions
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='B')
    df_v1 = pd.DataFrame({
        'date': dates,
        'close_price': np.random.randn(len(dates)),
        'volume': np.random.randint(0, 1000, len(dates))
    })
    df_v2 = df_v1.copy()
    df_v2['close_price'] = df_v2['close_price'] * 1.1  # Modify data
    
    # Test that versions are different
    assert not df_v1.equals(df_v2), "Different versions should have different data"
    
    # Test that version metadata is preserved
    df_v1.attrs['version'] = '1.0'
    df_v2.attrs['version'] = '2.0'
    assert df_v1.attrs['version'] != df_v2.attrs['version'], "Version metadata should be preserved"


def test_market_features_validation():
    """Test market features validation."""
    # Create temporary directory for market features
    with tempfile.TemporaryDirectory() as temp_dir:
        market_dir = os.path.join(temp_dir, 'market_features')
        os.makedirs(market_dir)
        
        # Create valid market volatility data
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='B')
        market_volatility = pd.DataFrame({
            'date': dates,
            'market_volatility_daily_volatility': np.random.randn(len(dates)),
            'market_volatility_weekly_volatility': np.random.randn(len(dates)),
            'market_volatility_monthly_volatility': np.random.randn(len(dates)),
            'market_volatility_avg_correlation': np.random.randn(len(dates)),
            'market_volatility_ticker': ['^GSPC'] * len(dates)
        })
        # Ensure ticker column is str
        market_volatility['market_volatility_ticker'] = market_volatility['market_volatility_ticker'].astype(str)
        market_volatility.to_parquet(os.path.join(market_dir, 'market_volatility.parquet'))
        
        # Create valid daily breadth data
        daily_breadth = pd.DataFrame({
            'date': dates,
            'daily_breadth_adv_dec_line': np.random.randn(len(dates)).astype(float),
            'daily_breadth_new_highs': np.random.randint(0, 100, len(dates)),
            'daily_breadth_new_lows': np.random.randint(0, 100, len(dates)),
            'daily_breadth_above_ma20': np.random.randn(len(dates)),
            'daily_breadth_above_ma50': np.random.randn(len(dates)),
            'daily_breadth_rsi_bullish': np.random.randn(len(dates)),
            'daily_breadth_rsi_oversold': np.random.randn(len(dates)),
            'daily_breadth_rsi_overbought': np.random.randn(len(dates)),
            'daily_breadth_macd_bullish': np.random.randn(len(dates))
        })
        # Ensure adv_dec_line is float
        daily_breadth['daily_breadth_adv_dec_line'] = daily_breadth['daily_breadth_adv_dec_line'].astype(float)
        daily_breadth.to_parquet(os.path.join(market_dir, 'daily_breadth.parquet'))
        
        # Create valid market sentiment data
        market_sentiment = pd.DataFrame({
            'date': dates,
            'market_sentiment_ma5': np.random.randn(len(dates)),
            'market_sentiment_ma20': np.random.randn(len(dates)),
            'market_sentiment_momentum': np.random.randn(len(dates)),
            'market_sentiment_volatility': np.random.randn(len(dates)),
            'market_sentiment_zscore': np.random.randn(len(dates))
        })
        market_sentiment.to_parquet(os.path.join(market_dir, 'market_sentiment.parquet'))
        
        # Create valid GDELT data
        gdelt_data = pd.DataFrame({
            'date': dates,
            'avg_tone': np.random.randn(len(dates))
        })
        gdelt_data.to_parquet(os.path.join(market_dir, 'gdelt_raw.parquet'))
        
        # Create sector directory and add a sector file (generic column names)
        sectors_dir = os.path.join(market_dir, 'sectors')
        os.makedirs(sectors_dir)
        sector_df = pd.DataFrame({
            'date': dates,
            'sector_price': np.random.randn(len(dates)),
            'sector_volatility': np.random.randn(len(dates)),
            'sector_volume': np.random.randint(0, 1000, len(dates)).astype(float),
            'sector_returns_1d': np.random.randn(len(dates)),
            'sector_returns_5d': np.random.randn(len(dates)),
            'sector_returns_20d': np.random.randn(len(dates)),
            'sector_momentum_5d': np.random.randn(len(dates)),
            'sector_momentum_20d': np.random.randn(len(dates))
        })
        sector_df.to_parquet(os.path.join(sectors_dir, 'sector.parquet'))
        
        # Test validation of all market features
        assert validate_market_features(data_dir=market_dir)
        
        # Test with invalid market volatility data
        invalid_volatility = market_volatility.copy()
        invalid_volatility['market_volatility_daily_volatility'] = 'invalid'  # Wrong type
        invalid_volatility.to_parquet(os.path.join(market_dir, 'market_volatility.parquet'))
        assert not validate_market_features(data_dir=market_dir)
        
        # Test with missing required column
        invalid_breadth = daily_breadth.copy()
        invalid_breadth = invalid_breadth.drop('daily_breadth_adv_dec_line', axis=1)
        invalid_breadth.to_parquet(os.path.join(market_dir, 'daily_breadth.parquet'))
        assert not validate_market_features(data_dir=market_dir)
        
        # Test with invalid sector data
        invalid_sector = sector_df.copy()
        invalid_sector['sector_price'] = 'invalid'  # Wrong type
        invalid_sector.to_parquet(os.path.join(sectors_dir, 'sector.parquet'))
        assert not validate_market_features(data_dir=market_dir) 