"""Tests for the analysis module."""

import pandas as pd
import pytest

from weekly_trading_advisor.analysis import (
    calculate_technical_indicators,
    calculate_score,
    analyze_stock
)

@pytest.fixture
def sample_data():
    """Create sample stock data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    data = {
        'Open': [100.0] * 10,
        'High': [105.0] * 10,
        'Low': [95.0] * 10,
        'Close': [102.0] * 10,
        'Volume': [1000000] * 10
    }
    return pd.DataFrame(data, index=dates)

def test_calculate_technical_indicators(sample_data):
    """Test calculation of technical indicators."""
    df = calculate_technical_indicators(sample_data)
    
    # Check that all required indicators are present
    required_indicators = [
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Upper', 'BB_Lower', 'BB_Middle',
        'SMA_20', 'SMA_50', 'SMA_200'
    ]
    
    for indicator in required_indicators:
        assert indicator in df.columns
        assert not df[indicator].isna().all()

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
    
    # Check details
    assert isinstance(details, dict)
    assert all(0 <= value <= 2 for value in details.values())
    
    # Check targets (may be None if no data available)
    assert targets is None or isinstance(targets, dict) 