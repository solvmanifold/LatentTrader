"""Tests for the output module."""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

import pandas as pd

from trading_advisor.output import (
    generate_technical_summary,
    generate_structured_data,
    save_json_report,
    generate_report
)

@pytest.fixture
def sample_data():
    """Create sample stock data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
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

def test_generate_report():
    positions = [
        {
            "ticker": "AAPL",
            "score": {"total": 7.5, "details": {"rsi": 1.0}},
            "price_data": {"current": 150.0},
            "technical_indicators": {"rsi": 65.0},
            "summary": "Test summary",
            "position": {"quantity": 100},
            "analyst_targets": None
        }
    ]
    new_picks = [
        {
            "ticker": "MSFT",
            "score": {"total": 8.0, "details": {"rsi": 1.0}},
            "price_data": {"current": 250.0},
            "technical_indicators": {"rsi": 60.0},
            "summary": "Test summary",
            "position": None,
            "analyst_targets": None
        }
    ]
    structured_data = {
        "timestamp": "2024-01-01T00:00:00",
        "positions": positions,
        "new_picks": new_picks
    }
    report = generate_report(structured_data)
    assert isinstance(report, str)
    assert "AAPL" in report
    assert "MSFT" in report

def test_generate_report_with_benchmarking():
    positions = [
        {
            "ticker": "AAPL",
            "score": {"total": 7.5, "details": {"rsi": 1.0}},
            "price_data": {"current": 150.0},
            "technical_indicators": {"rsi": 65.0},
            "summary": "Test summary",
            "position": {"quantity": 100},
            "analyst_targets": None
        }
    ]
    new_picks = [
        {
            "ticker": "MSFT",
            "score": {"total": 8.0, "details": {"rsi": 1.0}},
            "price_data": {"current": 250.0},
            "technical_indicators": {"rsi": 60.0},
            "summary": "Test summary",
            "position": None,
            "analyst_targets": None
        },
        {
            "ticker": "GOOGL",
            "score": {"total": 6.5, "details": {"rsi": 1.0}},
            "price_data": {"current": 2800.0},
            "technical_indicators": {"rsi": 55.0},
            "summary": "Test summary",
            "position": None,
            "analyst_targets": None
        }
    ]
    structured_data = {
        "timestamp": "2024-01-01T00:00:00",
        "positions": positions,
        "new_picks": new_picks
    }
    report = generate_report(structured_data)
    assert isinstance(report, str)
    assert "Top 2 Setups by Confidence x Upside" in report
    assert "MSFT" in report
    assert "GOOGL" in report 