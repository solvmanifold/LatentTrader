"""Tests for the CLI module."""

import pytest
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
import pandas as pd

from weekly_trading_advisor.cli import app

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

def test_version():
    """Test version command."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "Weekly Trading Advisor v" in result.stdout

def test_analyze_help():
    """Test analyze command help."""
    result = runner.invoke(app, ["analyze", "--help"])
    assert result.exit_code == 0
    assert "Analyze stocks and generate trading advice" in result.stdout

@patch('weekly_trading_advisor.cli.ensure_data_dir')
@patch('weekly_trading_advisor.cli.load_tickers')
@patch('weekly_trading_advisor.cli.load_positions')
@patch('weekly_trading_advisor.cli.download_stock_data')
@patch('weekly_trading_advisor.cli.analyze_stock')
@patch('weekly_trading_advisor.cli.generate_technical_summary')
@patch('weekly_trading_advisor.cli.generate_structured_data')
@patch('weekly_trading_advisor.cli.generate_report')
def test_analyze_command(
    mock_generate_report,
    mock_generate_structured_data,
    mock_generate_technical_summary,
    mock_analyze_stock,
    mock_download_stock_data,
    mock_load_positions,
    mock_load_tickers,
    mock_ensure_data_dir,
    mock_tickers_file,
    mock_positions_file
):
    """Test analyze command with all options."""
    # Setup mocks
    mock_load_tickers.return_value = ["AAPL", "MSFT", "GOOGL"]
    mock_load_positions.return_value = {"AAPL": {"quantity": 100}}
    # Return a non-empty DataFrame for download_stock_data
    df = pd.DataFrame({
        'Open': [100.0] * 20,
        'High': [105.0] * 20,
        'Low': [95.0] * 20,
        'Close': [102.0] * 20,
        'Volume': [1000000] * 20
    }, index=pd.date_range(start='2024-01-01', periods=20, freq='D'))
    mock_download_stock_data.return_value = df
    mock_analyze_stock.return_value = (7.5, {"rsi": 1.0}, None)
    mock_generate_technical_summary.return_value = "Test summary"
    mock_generate_structured_data.return_value = {"ticker": "AAPL"}
    mock_generate_report.return_value = "Test report"
    
    # Test with all options
    result = runner.invoke(app, [
        "analyze",
        "--tickers", str(mock_tickers_file),
        "--positions", str(mock_positions_file),
        "--top-n", "3",
        "--output", "report.md",
        "--save-json", "analysis.json",
        "--history-days", "50"
    ])
    
    assert result.exit_code == 0
    assert "Test report" in result.stdout
    
    # Verify mock calls
    mock_ensure_data_dir.assert_called_once()
    mock_load_tickers.assert_called_once_with(str(mock_tickers_file))
    mock_load_positions.assert_called_once_with(Path(mock_positions_file))
    assert mock_download_stock_data.call_count == 3  # One for each ticker
    assert mock_analyze_stock.call_count == 3
    assert mock_generate_technical_summary.call_count == 3
    assert mock_generate_structured_data.call_count == 3
    mock_generate_report.assert_called_once()

@patch('weekly_trading_advisor.cli.ensure_data_dir')
@patch('weekly_trading_advisor.cli.load_tickers')
@patch('weekly_trading_advisor.cli.load_positions')
@patch('weekly_trading_advisor.cli.download_stock_data')
@patch('weekly_trading_advisor.cli.analyze_stock')
@patch('weekly_trading_advisor.cli.generate_technical_summary')
@patch('weekly_trading_advisor.cli.generate_structured_data')
@patch('weekly_trading_advisor.cli.generate_report')
def test_analyze_positions_only(
    mock_generate_report,
    mock_generate_structured_data,
    mock_generate_technical_summary,
    mock_analyze_stock,
    mock_download_stock_data,
    mock_load_positions,
    mock_load_tickers,
    mock_ensure_data_dir,
    mock_tickers_file,
    mock_positions_file
):
    """Test analyze command with positions-only option."""
    # Setup mocks
    mock_load_tickers.return_value = ["AAPL", "MSFT", "GOOGL"]
    mock_load_positions.return_value = {"AAPL": {"quantity": 100}}
    df = pd.DataFrame({
        'Open': [100.0] * 20,
        'High': [105.0] * 20,
        'Low': [95.0] * 20,
        'Close': [102.0] * 20,
        'Volume': [1000000] * 20
    }, index=pd.date_range(start='2024-01-01', periods=20, freq='D'))
    mock_download_stock_data.return_value = df
    mock_analyze_stock.return_value = (7.5, {"rsi": 1.0}, None)
    mock_generate_technical_summary.return_value = "Test summary"
    mock_generate_structured_data.return_value = {"ticker": "AAPL"}
    mock_generate_report.return_value = "Test report"
    
    # Test with positions-only
    result = runner.invoke(app, [
        "analyze",
        "--tickers", str(mock_tickers_file),
        "--positions", str(mock_positions_file),
        "--positions-only"
    ])
    
    assert result.exit_code == 0
    assert "Test report" in result.stdout
    
    # Verify mock calls
    mock_ensure_data_dir.assert_called_once()
    mock_load_tickers.assert_called_once_with(str(mock_tickers_file))
    mock_load_positions.assert_called_once_with(Path(mock_positions_file))
    assert mock_download_stock_data.call_count == 3
    assert mock_analyze_stock.call_count == 3
    assert mock_generate_technical_summary.call_count == 3
    assert mock_generate_structured_data.call_count == 3
    mock_generate_report.assert_called_once()

@patch('weekly_trading_advisor.cli.ensure_data_dir')
@patch('weekly_trading_advisor.cli.load_tickers')
@patch('weekly_trading_advisor.cli.load_positions')
@patch('weekly_trading_advisor.cli.download_stock_data')
@patch('weekly_trading_advisor.cli.analyze_stock')
@patch('weekly_trading_advisor.cli.generate_technical_summary')
@patch('weekly_trading_advisor.cli.generate_structured_data')
@patch('weekly_trading_advisor.cli.generate_report')
def test_analyze_no_positions(
    mock_generate_report,
    mock_generate_structured_data,
    mock_generate_technical_summary,
    mock_analyze_stock,
    mock_download_stock_data,
    mock_load_positions,
    mock_load_tickers,
    mock_ensure_data_dir,
    mock_tickers_file
):
    """Test analyze command with no-positions option."""
    # Setup mocks
    mock_load_tickers.return_value = ["AAPL", "MSFT", "GOOGL"]
    df = pd.DataFrame({
        'Open': [100.0] * 20,
        'High': [105.0] * 20,
        'Low': [95.0] * 20,
        'Close': [102.0] * 20,
        'Volume': [1000000] * 20
    }, index=pd.date_range(start='2024-01-01', periods=20, freq='D'))
    mock_download_stock_data.return_value = df
    mock_analyze_stock.return_value = (7.5, {"rsi": 1.0}, None)
    mock_generate_technical_summary.return_value = "Test summary"
    mock_generate_structured_data.return_value = {"ticker": "AAPL"}
    mock_generate_report.return_value = "Test report"
    
    # Test with no-positions
    result = runner.invoke(app, [
        "analyze",
        "--tickers", str(mock_tickers_file),
        "--no-positions"
    ])
    
    assert result.exit_code == 0
    assert "Test report" in result.stdout
    
    # Verify mock calls
    mock_ensure_data_dir.assert_called_once()
    mock_load_tickers.assert_called_once_with(str(mock_tickers_file))
    assert mock_load_positions.call_count == 0
    assert mock_download_stock_data.call_count == 3  # For all tickers
    assert mock_analyze_stock.call_count == 3
    assert mock_generate_technical_summary.call_count == 3
    assert mock_generate_structured_data.call_count == 3
    mock_generate_report.assert_called_once() 