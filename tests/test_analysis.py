"""Tests for the analysis module."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from trading_advisor.analysis import (
    calculate_score,
    calculate_technical_indicators,
    analyze_stock,
    get_analyst_targets
)
from trading_advisor.config import SCORE_WEIGHTS

@pytest.fixture
def sample_data():
    """Create sample stock data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=60, freq='D')  # Changed to 60 days
    data = {
        'Open': [100.0] * 60,
        'High': [105.0] * 60,
        'Low': [95.0] * 60,
        'Close': [102.0] * 60,
        'Volume': [1000000] * 60,
        'RSI': [65.0] * 60,
        'MACD': [2.0] * 60,
        'MACD_Signal': [1.5] * 60,
        'MACD_Hist': [0.5] * 60,
        'BB_Upper': [105.0] * 60,
        'BB_Lower': [95.0] * 60,
        'BB_Middle': [100.0] * 60,
        'SMA_20': [101.0] * 60,
        'SMA_50': [100.0] * 60
    }
    return pd.DataFrame(data, index=dates)

def test_calculate_technical_indicators(sample_data):
    """Test calculation of technical indicators."""
    df = calculate_technical_indicators(sample_data)
    
    # Check that all required indicators are present
    required_columns = [
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Pband',
        'SMA_20'
    ]
    for col in required_columns:
        assert col in df.columns

def test_calculate_score(sample_data):
    """Test score calculation."""
    df = calculate_technical_indicators(sample_data)
    score, details = calculate_score(df)
    
    assert isinstance(score, float)
    assert 0 <= score <= 10
    assert isinstance(details, dict)
    assert all(key in details for key in ['macd', 'rsi', 'bollinger', 'moving_averages', 'volume', 'analyst_targets'])

def test_analyze_stock(sample_data):
    """Test stock analysis."""
    score, details, targets = analyze_stock('AAPL', sample_data)
    
    assert isinstance(score, float)
    assert 0 <= score <= 10
    assert isinstance(details, dict)
    assert all(key in details for key in ['macd', 'rsi', 'bollinger', 'moving_averages', 'volume', 'analyst_targets'])
    assert targets is None or isinstance(targets, dict)

def test_get_analyst_targets():
    """Test getting analyst targets."""
    with patch('yfinance.Ticker') as mock_ticker:
        mock_ticker.return_value.info = {
            'targetMeanPrice': 200.0,
            'currentPrice': 150.0,
            'targetLowPrice': 180.0,
            'targetHighPrice': 220.0
        }
        
        targets = get_analyst_targets('AAPL')
        assert isinstance(targets, dict)
        assert 'current_price' in targets
        assert 'median_target' in targets
        assert 'low_target' in targets
        assert 'high_target' in targets

def test_score_components():
    """Test individual score components."""
    # Create data with specific conditions
    dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
    data = {
        'Open': [100.0] * 60,
        'High': [105.0] * 60,
        'Low': [95.0] * 60,
        'Close': [102.0] * 60,
        'Volume': [1000000] * 60,
        'RSI': [50.0] * 60,
        'MACD': [1.0] * 60,
        'MACD_Signal': [0.5] * 60,
        'MACD_Hist': [0.5] * 60,
        'BB_Upper': [110.0] * 60,
        'BB_Middle': [105.0] * 60,
        'BB_Lower': [100.0] * 60,
        'BB_Pband': [0.5] * 60,
        'SMA_20': [104.0] * 60,
        'SMA_50': [103.0] * 60
    }
    df = pd.DataFrame(data, index=dates)
    
    # Test RSI component
    df['RSI'] = 29  # Oversold condition (must be < 30)
    score, details = calculate_score(df)
    assert isinstance(score, float)
    assert details['rsi'] > 0  # Should be positive for oversold
    
    # Test MACD component
    df['MACD'] = 1.0
    df['MACD_Signal'] = 0.5
    df['MACD_Hist'] = 0.6  # > MACD_WEAK_DIVERGENCE
    score, details = calculate_score(df)
    assert isinstance(score, float)
    assert details['macd'] > 0  # Should be positive for bullish MACD
    
    # Test Bollinger Bands component
    df['BB_Pband'] = 0.02  # Oversold condition
    score, details = calculate_score(df)
    assert isinstance(score, float)
    assert details['bollinger'] == SCORE_WEIGHTS['bollinger_low']  # Should be positive for oversold

def test_score_normalization():
    """Test score normalization."""
    # Create sample data with 60 days
    dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
    data = {
        'Open': [100.0] * 60,
        'High': [105.0] * 60,
        'Low': [95.0] * 60,
        'Close': [102.0] * 60,
        'Volume': [1000000] * 60
    }
    df = pd.DataFrame(data, index=dates)
    df = calculate_technical_indicators(df)
    
    # Test minimum score
    df['RSI'] = 80  # Overbought
    df['MACD'] = -1.0
    df['MACD_Signal'] = 0.0
    df['MACD_Hist'] = -1.0
    df['BB_Pband'] = 0.98  # Overbought
    score, _ = calculate_score(df)
    assert score >= 0
    
    # Test maximum score
    df['RSI'] = 20  # Oversold
    df['MACD'] = 1.0
    df['MACD_Signal'] = 0.0
    df['MACD_Hist'] = 1.0
    df['BB_Pband'] = 0.02  # Oversold
    score, _ = calculate_score(df)
    assert score <= 10 