"""Tests for the output module."""

import json
import pytest
from pathlib import Path
from datetime import datetime

import pandas as pd

from src.output import (
    generate_technical_summary,
    generate_structured_data,
    save_json_report,
    generate_report
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
        'Volume': [1000000] * 10,
        'RSI': [65.0] * 10,
        'MACD': [2.0] * 10,
        'MACD_Signal': [1.5] * 10,
        'MACD_Hist': [0.5] * 10,
        'BB_Upper': [105.0] * 10,
        'BB_Lower': [95.0] * 10,
        'BB_Middle': [100.0] * 10,
        'SMA_20': [101.0] * 10,
        'SMA_50': [100.0] * 10,
        'SMA_200': [99.0] * 10
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def sample_position():
    """Create sample position data."""
    return {
        'quantity': 100,
        'price': 150.00,
        'market_value': 15000.00,
        'cost_basis': 14000.00,
        'gain_pct': 7.14,
        'account_pct': 25.0
    }

@pytest.fixture
def sample_analyst_targets():
    """Create sample analyst targets."""
    return {
        'current_price': 150.00,
        'median_target': 165.00,
        'low_target': 140.00,
        'high_target': 180.00
    }

def test_generate_technical_summary(sample_data, sample_position, sample_analyst_targets):
    """Test technical summary generation."""
    summary = generate_technical_summary(
        'AAPL',
        sample_data,
        7.5,
        {'rsi': 1.0, 'bollinger': 1.0, 'macd': 2.0, 'moving_averages': 2.0, 'analyst_targets': 1.5},
        sample_analyst_targets,
        sample_position
    )
    
    assert isinstance(summary, str)
    assert 'AAPL' in summary
    assert 'Current Position' in summary
    assert 'Price:' in summary
    assert 'Technical Score:' in summary

def test_generate_structured_data(sample_data, sample_position, sample_analyst_targets):
    """Test structured data generation."""
    data = generate_structured_data(
        'AAPL',
        sample_data,
        7.5,
        {'rsi': 1.0, 'bollinger': 1.0, 'macd': 2.0, 'moving_averages': 2.0, 'analyst_targets': 1.5},
        sample_analyst_targets,
        sample_position
    )
    
    assert isinstance(data, dict)
    assert data['ticker'] == 'AAPL'
    assert 'price_data' in data
    assert 'technical_indicators' in data
    assert 'score' in data
    assert data['score']['total'] == 7.5
    assert 'position' in data
    assert 'analyst_targets' in data

def test_save_json_report(tmp_path):
    """Test saving JSON report."""
    data = {
        'timestamp': datetime.now().isoformat(),
        'positions': [],
        'new_picks': []
    }
    
    output_path = tmp_path / "test.json"
    save_json_report(data, output_path)
    
    assert output_path.exists()
    with open(output_path) as f:
        saved_data = json.load(f)
    assert saved_data == data

def test_generate_report(tmp_path):
    """Test report generation."""
    positions = [
        ('AAPL', 7.5, '## AAPL\nTest summary'),
        ('MSFT', 6.5, '## MSFT\nTest summary')
    ]
    new_picks = [
        ('GOOGL', 8.5, '## GOOGL\nTest summary'),
        ('AMZN', 7.0, '## AMZN\nTest summary')
    ]
    structured_data = {
        'timestamp': datetime.now().isoformat(),
        'positions': [],
        'new_picks': []
    }
    
    # Test console output
    report = generate_report(positions, new_picks, structured_data)
    assert isinstance(report, str)
    assert 'Weekly Trading Advisor Report' in report
    assert 'Current Positions' in report
    assert 'New Technical Picks' in report
    
    # Test file output
    output_path = tmp_path / "report.md"
    report = generate_report(positions, new_picks, structured_data, output_path)
    assert output_path.exists()
    content = output_path.read_text()
    assert 'Weekly Trading Advisor Report' in content
    
    # Test JSON output
    json_path = tmp_path / "analysis.json"
    report = generate_report(positions, new_picks, structured_data, output_path, json_path)
    assert json_path.exists()
    with open(json_path) as f:
        saved_data = json.load(f)
    assert saved_data == structured_data 