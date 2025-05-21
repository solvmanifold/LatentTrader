"""Tests for the output module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime
import pandas as pd

from output import (
    generate_technical_summary,
    generate_structured_data,
    save_json_report,
    generate_report
)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    data = {
        'Open': [100.0] * 10,
        'High': [105.0] * 10,
        'Low': [95.0] * 10,
        'Close': [102.0] * 10,
        'Volume': [1000000] * 10,
        'RSI': [50.0] * 10,
        'MACD': [0.0] * 10,
        'MACD_Signal': [0.0] * 10,
        'MACD_Hist': [0.0] * 10,
        'BB_Upper': [110.0] * 10,
        'BB_Middle': [100.0] * 10,
        'BB_Lower': [90.0] * 10,
        'SMA_20': [100.0] * 10
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def sample_analyst_targets():
    """Create sample analyst targets."""
    return {
        'current_price': 150.0,
        'median_target': 200.0,
        'low_target': 180.0,
        'high_target': 220.0
    }

def test_generate_technical_summary(sample_data, sample_analyst_targets):
    """Test technical summary generation."""
    summary = generate_technical_summary(
        ticker='AAPL',
        df=sample_data,
        score=7.5,
        score_details={
            'macd': 1.5,
            'rsi': 2.0,
            'bollinger': 1.0,
            'moving_averages': 1.5,
            'volume': 1.0,
            'analyst_targets': 0.5
        },
        analyst_targets=sample_analyst_targets
    )
    
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert "AAPL" in summary
    assert "RSI" in summary
    assert "MACD" in summary
    assert "BB:" in summary
    assert "7.5/10" in summary

def test_generate_structured_data(sample_data, sample_analyst_targets):
    """Test structured data generation."""
    data = generate_structured_data(
        ticker='AAPL',
        df=sample_data,
        score=7.5,
        score_details={
            'macd': 1.5,
            'rsi': 2.0,
            'bollinger': 1.0,
            'moving_averages': 1.5,
            'volume': 1.0,
            'analyst_targets': 0.5
        },
        analyst_targets=sample_analyst_targets
    )
    
    assert isinstance(data, dict)
    assert data['ticker'] == 'AAPL'
    assert 'timestamp' in data
    assert 'price_data' in data
    assert 'technical_indicators' in data
    assert 'score' in data
    assert 'analyst_targets' in data

def test_save_json_report(tmp_path):
    """Test saving JSON report."""
    data = {
        'ticker': 'AAPL',
        'timestamp': datetime.now().isoformat(),
        'score': 7.5
    }
    
    output_path = tmp_path / "test_report.json"
    save_json_report(data, output_path)
    
    assert output_path.exists()
    content = output_path.read_text()
    assert '"ticker": "AAPL"' in content
    assert '"score": 7.5' in content

def test_generate_report(tmp_path):
    """Test complete report generation."""
    positions = [('AAPL', 7.5, 'Sample summary for AAPL')]
    new_picks = [('MSFT', 8.0, 'Sample summary for MSFT')]
    structured_data = {
        'AAPL': {
            'ticker': 'AAPL',
            'score': 7.5
        },
        'MSFT': {
            'ticker': 'MSFT',
            'score': 8.0
        }
    }
    
    output_path = tmp_path / "test_report.md"
    json_path = tmp_path / "test_report.json"
    
    report = generate_report(
        positions=positions,
        new_picks=new_picks,
        structured_data=structured_data,
        output_path=output_path,
        save_json=json_path
    )
    
    assert isinstance(report, str)
    assert "Weekly Trading Advisor Report" in report
    assert "AAPL" in report
    assert "MSFT" in report
    
    if output_path:
        assert output_path.exists()
        content = output_path.read_text()
        assert "Weekly Trading Advisor Report" in content
    
    if json_path:
        assert json_path.exists()
        content = json_path.read_text()
        assert '"ticker": "AAPL"' in content
        assert '"ticker": "MSFT"' in content 