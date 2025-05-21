"""Tests for the data module."""

import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data import (
    get_yf_ticker,
    ensure_data_dir,
    read_csv_with_dates,
    is_csv_format_valid,
    handle_multiindex_columns,
    parse_brokerage_csv,
    load_positions,
    load_tickers
)

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
    assert get_yf_ticker("AAPL") is ticker  # Should return cached object

def test_ensure_data_dir(tmp_path):
    """Test data directory creation."""
    with patch('weekly_trading_advisor.data.DATA_DIR', tmp_path):
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