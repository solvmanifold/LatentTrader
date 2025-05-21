"""Tests for the analysis module."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from analysis import (
    calculate_score,
    calculate_technical_indicators,
    analyze_stock,
    get_analyst_targets
)
from config import SCORE_WEIGHTS

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
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
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    data = {
        'Open': [100.0] * 10,
        'High': [105.0] * 10,
        'Low': [95.0] * 10,
        'Close': [102.0] * 10,
        'Volume': [1000000] * 10
    }
    df = pd.DataFrame(data, index=dates)
    df = calculate_technical_indicators(df)
    
    # Test RSI component
    df['RSI'] = 29  # Oversold condition (must be < 30)
    score, details = calculate_score(df)
    assert details['rsi'] > 0  # Should be positive for oversold
    
    # Test MACD component
    df['MACD'] = 1.0
    df['MACD_Signal'] = 0.5
    df['MACD_Hist'] = 0.6  # > MACD_WEAK_DIVERGENCE
    score, details = calculate_score(df)
    assert details['macd'] > 0  # Should be positive for bullish MACD
    
    # Test Bollinger Bands component
    df['BB_Pband'] = 0.02  # Oversold condition
    score, details = calculate_score(df)
    assert details['bollinger'] == SCORE_WEIGHTS['bollinger_low']  # Should be positive for oversold

def test_score_normalization():
    """Test score normalization to [0, 10] range."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    data = {
        'Open': [100.0] * 10,
        'High': [105.0] * 10,
        'Low': [95.0] * 10,
        'Close': [102.0] * 10,
        'Volume': [1000000] * 10
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