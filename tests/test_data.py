"""Tests for the data module."""

import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import os

from trading_advisor.data import (
    download_stock_data,
    ensure_data_dir,
    read_csv_with_dates,
    is_csv_format_valid,
    handle_multiindex_columns,
    parse_brokerage_csv,
    load_positions,
    load_tickers,
    get_yf_ticker
)
from trading_advisor.analysis import calculate_technical_indicators, get_analyst_targets

@pytest.fixture
def sample_csv_data(tmp_path):
    """Create a sample CSV file for testing."""
    df = pd.DataFrame({
        'Open': [100.0] * 10,
        'High': [105.0] * 10,
        'Low': [95.0] * 10,
        'Close': [102.0] * 10,
        'Volume': [1000000] * 10
    }, index=pd.date_range(start='2024-01-01', end='2024-01-10', freq='D'))
    
    file_path = tmp_path / "test.csv"
    df.to_csv(file_path)
    return file_path

@pytest.fixture
def sample_brokerage_csv(tmp_path):
    """Create a sample brokerage CSV file for testing."""
    content = """Header
Empty Row
Symbol,Security Type,Qty (Quantity),Price,Mkt Val (Market Value),Cost Basis,Gain % (Gain/Loss %),% of Acct (% of Account)
AAPL,Equity,100,$150.00,$15000.00,$14000.00,7.14%,25.0%
CASH,Cash,10000,$1.00,$10000.00,$10000.00,0.00%,75.0%"""
    
    file_path = tmp_path / "positions.csv"
    file_path.write_text(content)
    return file_path

def test_get_yf_ticker():
    """Test getting a cached Ticker object."""
    ticker = get_yf_ticker("AAPL")
    assert ticker is not None
    # Test caching
    ticker2 = get_yf_ticker("AAPL")
    assert ticker is ticker2

def test_ensure_data_dir(tmp_path):
    """Test ensuring data directory exists."""
    with patch('trading_advisor.data.DATA_DIR', tmp_path):
        ensure_data_dir()
        assert tmp_path.exists()
        assert tmp_path.is_dir()

def test_read_csv_with_dates(sample_csv_data):
    """Test reading CSV with dates."""
    df = read_csv_with_dates(sample_csv_data, pd.Timestamp('2024-01-10'))
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert isinstance(df.index, pd.DatetimeIndex)

def test_is_csv_format_valid(sample_csv_data):
    """Test CSV format validation."""
    assert is_csv_format_valid(sample_csv_data)
    assert not is_csv_format_valid(sample_csv_data, ['InvalidColumn'])

def test_handle_multiindex_columns():
    """Test handling of MultiIndex columns."""
    # Create DataFrame with MultiIndex columns
    df = pd.DataFrame(
        [[1, 2], [3, 4]],
        columns=pd.MultiIndex.from_tuples([('A', 'AAPL'), ('B', 'AAPL')])
    )
    
    result = handle_multiindex_columns(df, 'AAPL')
    assert isinstance(result.columns, pd.Index)
    assert 'A' in result.columns
    assert 'B' in result.columns

def test_parse_brokerage_csv(sample_brokerage_csv):
    """Test parsing brokerage CSV file."""
    positions = parse_brokerage_csv(sample_brokerage_csv)
    
    assert 'AAPL' in positions
    assert 'CASH' not in positions  # Should exclude non-equity positions
    
    aapl_position = positions['AAPL']
    assert aapl_position['quantity'] == 100
    assert aapl_position['price'] == 150.00
    assert aapl_position['market_value'] == 15000.00
    assert aapl_position['cost_basis'] == 14000.00
    assert aapl_position['gain_pct'] == 7.14
    assert aapl_position['account_pct'] == 25.0

def test_load_positions(sample_brokerage_csv):
    """Test loading positions from file."""
    positions = load_positions(sample_brokerage_csv)
    assert isinstance(positions, dict)
    assert 'AAPL' in positions

def test_load_positions_none():
    """Test loading positions with no file."""
    positions = load_positions(None)
    assert isinstance(positions, dict)
    assert len(positions) == 0

def test_load_tickers(tmp_path):
    """Test loading tickers from file."""
    # Create a tickers file
    tickers_file = tmp_path / "tickers.txt"
    tickers_file.write_text("AAPL\nMSFT\nGOOGL")
    
    tickers = load_tickers(str(tickers_file))
    assert isinstance(tickers, list)
    assert len(tickers) == 3
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    assert "GOOGL" in tickers

def test_load_tickers_default():
    """Test loading default ticker."""
    tickers = load_tickers(None)
    assert isinstance(tickers, list)
    assert len(tickers) == 1
    assert tickers[0] == "AAPL"

def test_load_tickers_all():
    """Test loading all tickers."""
    tickers = load_tickers("all")
    assert isinstance(tickers, list)
    assert len(tickers) > 0
    assert "AAPL" in tickers

def test_download_stock_data(tmp_path):
    """Test download_stock_data with mocks and temp features dir."""
    today = pd.Timestamp.today().normalize()
    sample_dates = pd.bdate_range(end=today, periods=50)
    sample_df = pd.DataFrame({
        'Open': [100 + i for i in range(50)],
        'High': [101 + i for i in range(50)],
        'Low': [99 + i for i in range(50)],
        'Close': [100.5 + i for i in range(50)],
        'Volume': [1000 + i * 100 for i in range(50)]
    }, index=sample_dates)
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    features_path = features_dir / "AAPL_features_test.parquet"

    def mock_calculate_technical_indicators(df, *args, **kwargs):
        # Add all expected indicator columns with dummy values
        for col in ["RSI", "MACD", "MACD_Signal", "MACD_Hist", "BB_Upper", "BB_Lower", "BB_Middle", "BB_Pband", "SMA_20", "SMA_50", "SMA_100", "SMA_200", "EMA_100", "EMA_200"]:
            df[col] = 0.0
        return df

    with patch('trading_advisor.data.get_yf_ticker') as mock_get_ticker, \
         patch('trading_advisor.data.calculate_technical_indicators', side_effect=mock_calculate_technical_indicators), \
         patch('trading_advisor.data.get_analyst_targets', return_value={"target": 200}):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_df.copy()
        mock_get_ticker.return_value = mock_ticker

        # First call: no file exists, should download and write
        df = download_stock_data('AAPL', history_days=50, features_dir=str(features_dir), features_filename="AAPL_features_test.parquet")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50
        assert features_path.exists()

        # After running download_stock_data, check that 'score' and 'score_details' are NOT in the columns
        assert 'score' not in df.columns
        assert 'score_details' not in df.columns
        # Check that technical indicators and price/volume columns are present
        for col in ["Close", "High", "Low", "Open", "Volume", "RSI", "MACD", "MACD_Signal", "MACD_Hist", "BB_Upper", "BB_Lower", "BB_Middle", "BB_Pband", "SMA_20", "SMA_50", "SMA_100", "SMA_200", "EMA_100", "EMA_200"]:
            assert col in df.columns

        # Overwrite Parquet with up-to-date data (last date is today)
        sample_df.to_parquet(features_path)

        # Second call: file exists and is up-to-date, should not call history
        mock_ticker.history.reset_mock()
        df2 = download_stock_data('AAPL', history_days=50, features_dir=str(features_dir), features_filename="AAPL_features_test.parquet")
        assert isinstance(df2, pd.DataFrame)
        assert len(df2) == 50
        mock_ticker.history.assert_not_called()

        # Manually delete the last 5 rows from the Parquet file
        df_truncated = df.iloc[:-5]
        df_truncated.to_parquet(features_path)

        # Third call: file exists but is missing the last 5 rows, should redownload and fill in
        mock_ticker.history.reset_mock()
        df3 = download_stock_data('AAPL', history_days=50, features_dir=str(features_dir), features_filename="AAPL_features_test.parquet")
        assert isinstance(df3, pd.DataFrame)
        assert len(df3) == 50
        mock_ticker.history.assert_called_once()

def test_init_features(tmp_path):
    """Test init_features function to ensure it initializes features correctly."""
    from trading_advisor.cli import init_features
    from trading_advisor.data import load_tickers
    from unittest.mock import patch, MagicMock

    # Mock load_tickers to return a test list of tickers
    with patch('trading_advisor.data.load_tickers', return_value=['AAPL', 'MSFT']), \
         patch('trading_advisor.data.download_stock_data') as mock_download:
        # Mock download_stock_data to return a sample DataFrame
        sample_df = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [98, 99],
            'Close': [101, 102],
            'Volume': [1000, 1100]
        }, index=pd.date_range(start='2024-01-01', periods=2))
        mock_download.return_value = sample_df

        # Call init_features with a temporary directory
        init_features(tickers=None, all_tickers=True, years=1, features_dir=tmp_path)

        # Verify that download_stock_data was called for each ticker
        assert mock_download.call_count == 2
        mock_download.assert_any_call('AAPL', history_days=365, features_dir=str(tmp_path))
        mock_download.assert_any_call('MSFT', history_days=365, features_dir=str(tmp_path)) 