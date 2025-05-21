"""Tests for the analysis module."""

import pandas as pd
import pytest

from src.analysis import (
    calculate_technical_indicators,
    calculate_score,
    analyze_stock
)

@pytest.fixture
def sample_data():
    """Create sample stock data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    # Simulate a price trend and some volatility
    close = [100 + i + (i % 3) * 2 for i in range(30)]
    open_ = [c + 1 for c in close]
    high = [c + 2 for c in close]
    low = [c - 2 for c in close]
    volume = [1000000 + (i * 10000) for i in range(30)]
    data = {
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }
    return pd.DataFrame(data, index=dates)

def test_calculate_technical_indicators(sample_data):
    """Test calculation of technical indicators."""
    df = calculate_technical_indicators(sample_data)
    
    # Indicators that should be computable with 30 rows
    computable_indicators = [
        'RSI', 'BB_Upper', 'BB_Lower', 'BB_Middle', 'SMA_20'
    ]
    # Indicators that may require more data
    long_window_indicators = [
        'MACD', 'MACD_Signal', 'MACD_Hist', 'SMA_50', 'SMA_200'
    ]
    
    for indicator in computable_indicators:
        assert indicator in df.columns
        assert df[indicator].notna().any(), f"{indicator} should have at least one non-NaN value"
    for indicator in long_window_indicators:
        assert indicator in df.columns
        # Just check the column exists; don't assert non-NaN due to window length

def test_calculate_score(sample_data):
    """Test score calculation."""
    df = calculate_technical_indicators(sample_data)
    score, details = calculate_score(df)
    
    # Check score is between 0 and 10
    assert 0 <= score <= 10
    
    # Check score details
    assert isinstance(details, dict)
    assert all(0 <= value <= 2 for value in details.values())

def test_analyze_stock(sample_data):
    """Test stock analysis."""
    score, details, targets = analyze_stock('AAPL', sample_data)
    
    # Check score
    assert 0 <= score <= 10
    
    # Check details: all values should be numeric (float or int)
    assert isinstance(details, dict)
    for value in details.values():
        assert isinstance(value, (float, int))
    
    # Check targets (may be None if no data available)
    assert targets is None or isinstance(targets, dict) 