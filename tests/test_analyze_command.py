"""Tests for the refactored analyze command."""

import json
import pytest
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
import pandas as pd

from trading_advisor.cli import app

runner = CliRunner()

@pytest.fixture
def mock_data_dir(tmp_path):
    """Create a mock data directory."""
    return tmp_path

@pytest.fixture
def mock_tickers_file(tmp_path):
    """Create a mock tickers file."""
    file_path = tmp_path / "tickers.txt"
    file_path.write_text("AAPL\nMSFT\nGOOGL")
    return file_path

@pytest.fixture
def mock_positions_file(tmp_path):
    """Create a mock positions file."""
    file_path = tmp_path / "positions.csv"
    content = """Header
Empty Row
Symbol,Security Type,Qty (Quantity),Price,Mkt Val (Market Value),Cost Basis,Gain % (Gain/Loss %),% of Acct (% of Account)
AAPL,Equity,100,$150.00,$15000.00,$14000.00,7.14%,25.0%"""
    file_path.write_text(content)
    return file_path

@patch('trading_advisor.cli.download_stock_data')
@patch('trading_advisor.cli.analyze_stock')
def test_analyze_command_json_output(mock_analyze_stock, mock_download_stock_data, mock_tickers_file, mock_positions_file, tmp_path):
    # Setup mocks
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    data = {
        'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'High': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'Low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
        'Close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        'RSI': [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
        'MACD': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'MACD_Signal': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        'MACD_Hist': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        'BB_Upper': [110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
        'BB_Middle': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
        'BB_Lower': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'SMA_20': [104, 105, 106, 107, 108, 109, 110, 111, 112, 113],
        'SMA_50': [103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
        'SMA_200': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
    }
    mock_download_stock_data.return_value = pd.DataFrame(data, index=dates)
    mock_analyze_stock.return_value = (7.0, {
        "rsi": 2.0,
        "bollinger": 1.5,
        "macd": 2.0,
        "moving_averages": 1.0,
        "volume": 0.5
    }, {
        "current_price": 150.0,
        "median_target": 160.0,
        "low_target": 140.0,
        "high_target": 180.0
    })
    output_file = tmp_path / "analysis.json"
    result = runner.invoke(app, [
        "analyze",
        "--tickers", str(mock_tickers_file),
        "--positions", str(mock_positions_file),
        "--output", str(output_file)
    ])
    if result.exit_code != 0:
        print("CLI OUTPUT:\n", result.stdout)
    assert result.exit_code == 0
    assert output_file.exists()
    
    # Load and validate JSON structure
    with open(output_file) as f:
        data = json.load(f)
    
    # Check top-level structure
    assert "timestamp" in data
    assert "positions" in data
    assert "new_picks" in data
    
    # Check position data structure
    if data["positions"]:
        position = data["positions"][0]
        assert "ticker" in position
        assert "price_data" in position
        assert "technical_indicators" in position
        assert "score" in position
        assert "position" in position
        
        # Check technical indicators
        indicators = position["technical_indicators"]
        assert "rsi" in indicators
        assert "macd" in indicators
        assert "bollinger_bands" in indicators
        assert "moving_averages" in indicators
        
        # Check score structure
        score = position["score"]
        assert "total" in score
        assert "details" in score

@patch('trading_advisor.cli.download_stock_data')
@patch('trading_advisor.cli.analyze_stock')
def test_analyze_command_no_positions(mock_analyze_stock, mock_download_stock_data, mock_tickers_file, tmp_path):
    # Setup mocks
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    data = {
        'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'High': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'Low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
        'Close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        'RSI': [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
        'MACD': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'MACD_Signal': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        'MACD_Hist': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        'BB_Upper': [110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
        'BB_Middle': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
        'BB_Lower': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'SMA_20': [104, 105, 106, 107, 108, 109, 110, 111, 112, 113],
        'SMA_50': [103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
        'SMA_200': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
    }
    mock_download_stock_data.return_value = pd.DataFrame(data, index=dates)
    mock_analyze_stock.return_value = (7.0, {
        "rsi": 2.0,
        "bollinger": 1.5,
        "macd": 2.0,
        "moving_averages": 1.0,
        "volume": 0.5
    }, {
        "current_price": 150.0,
        "median_target": 160.0,
        "low_target": 140.0,
        "high_target": 180.0
    })
    output_file = tmp_path / "analysis.json"
    result = runner.invoke(app, [
        "analyze",
        "--tickers", str(mock_tickers_file),
        "--output", str(output_file)
    ])
    assert result.exit_code == 0
    assert output_file.exists()
    
    with open(output_file) as f:
        data = json.load(f)
    
    assert "positions" in data
    assert len(data["positions"]) == 0
    assert "new_picks" in data
    assert len(data["new_picks"]) > 0

@patch('trading_advisor.cli.download_stock_data')
@patch('trading_advisor.cli.analyze_stock')
def test_analyze_command_positions_only(mock_analyze_stock, mock_download_stock_data, mock_tickers_file, mock_positions_file, tmp_path):
    # Setup mocks
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    data = {
        'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'High': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'Low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
        'Close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        'RSI': [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
        'MACD': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'MACD_Signal': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        'MACD_Hist': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        'BB_Upper': [110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
        'BB_Middle': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
        'BB_Lower': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'SMA_20': [104, 105, 106, 107, 108, 109, 110, 111, 112, 113],
        'SMA_50': [103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
        'SMA_200': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
    }
    mock_download_stock_data.return_value = pd.DataFrame(data, index=dates)
    mock_analyze_stock.return_value = (7.0, {
        "rsi": 2.0,
        "bollinger": 1.5,
        "macd": 2.0,
        "moving_averages": 1.0,
        "volume": 0.5
    }, {
        "current_price": 150.0,
        "median_target": 160.0,
        "low_target": 140.0,
        "high_target": 180.0
    })
    output_file = tmp_path / "analysis.json"
    result = runner.invoke(app, [
        "analyze",
        "--tickers", str(mock_tickers_file),
        "--positions", str(mock_positions_file),
        "--positions-only",
        "--output", str(output_file)
    ])
    assert result.exit_code == 0
    assert output_file.exists()
    
    with open(output_file) as f:
        data = json.load(f)
    
    assert "positions" in data
    assert len(data["positions"]) > 0
    assert "new_picks" in data
    assert len(data["new_picks"]) == 0

@patch('trading_advisor.cli.download_stock_data')
def test_analyze_command_error_handling(mock_download_stock_data, mock_tickers_file, tmp_path):
    """Test error handling in analyze command."""
    output_file = tmp_path / "analysis.json"
    # Make download_stock_data raise an exception
    mock_download_stock_data.side_effect = Exception("Test error")
    result = runner.invoke(app, [
        "analyze",
        "--tickers", str(mock_tickers_file),
        "--output", str(output_file)
    ])
    assert result.exit_code == 1
    assert "Error during analysis: Test error" in result.stdout 