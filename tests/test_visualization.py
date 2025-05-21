"""Tests for the visualization module."""

import pytest
import pandas as pd
import os
from pathlib import Path
from trading_advisor.visualization import create_stock_chart, create_score_breakdown

@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    data = {
        'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'High': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'Low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
        'Close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        'MA20': [100, 100.5, 101, 101.5, 102, 102.5, 103, 103.5, 104, 104.5],
        'MA50': [99, 99.5, 100, 100.5, 101, 101.5, 102, 102.5, 103, 103.5],
        'RSI': [45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
        'bb_upper': [105, 105.5, 106, 106.5, 107, 107.5, 108, 108.5, 109, 109.5],
        'bb_lower': [95, 95.5, 96, 96.5, 97, 97.5, 98, 98.5, 99, 99.5]
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def sample_indicators():
    """Create sample technical indicators for testing."""
    return {
        'rsi_score': 2.0,
        'bb_score': 1.5,
        'macd_score': 2.0,
        'ma_score': 1.0,
        'volume_score': 0.5,
        'score': 7.0
    }

@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    return str(tmp_path / "charts")

def test_create_stock_chart(sample_data, sample_indicators, output_dir):
    """Test creating a stock chart."""
    # Create chart
    output_path = create_stock_chart(sample_data, "AAPL", sample_indicators, output_dir)
    
    # Check that file was created
    assert os.path.exists(output_path)
    assert output_path.endswith("AAPL_chart.html")
    
    # Check file size (should be non-zero)
    assert os.path.getsize(output_path) > 0

def test_create_score_breakdown(sample_indicators, output_dir):
    """Test creating a score breakdown chart."""
    # Create score breakdown
    output_path = create_score_breakdown("AAPL", 7.0, sample_indicators, output_dir)
    
    # Check that file was created
    assert os.path.exists(output_path)
    assert output_path.endswith("AAPL_score.html")
    
    # Check file size (should be non-zero)
    assert os.path.getsize(output_path) > 0

def test_create_stock_chart_missing_indicators(sample_data, output_dir):
    """Test creating a stock chart with missing indicators."""
    # Create minimal indicators
    indicators = {'score': 5.0}
    
    # Create chart
    output_path = create_stock_chart(sample_data, "AAPL", indicators, output_dir)
    
    # Check that file was created
    assert os.path.exists(output_path)
    assert output_path.endswith("AAPL_chart.html")
    
    # Check file size (should be non-zero)
    assert os.path.getsize(output_path) > 0

def test_create_score_breakdown_missing_scores(sample_indicators, output_dir):
    """Test creating a score breakdown with missing score components."""
    # Create minimal indicators
    indicators = {'score': 5.0}
    
    # Create score breakdown
    output_path = create_score_breakdown("AAPL", 5.0, indicators, output_dir)
    
    # Check that file was created
    assert os.path.exists(output_path)
    assert output_path.endswith("AAPL_score.html")
    
    # Check file size (should be non-zero)
    assert os.path.getsize(output_path) > 0 