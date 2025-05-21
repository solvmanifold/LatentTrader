"""Tests for the CLI module."""

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

@pytest.fixture
def mock_download_stock_data():
    """Mock the download_stock_data function."""
    with patch('trading_advisor.cli.download_stock_data') as mock:
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
        data = {
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'Low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            'Close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        }
        mock.return_value = pd.DataFrame(data, index=dates)
        yield mock

@pytest.fixture
def mock_analyze_stock():
    """Mock the analyze_stock function."""
    with patch('trading_advisor.cli.analyze_stock') as mock:
        mock.return_value = {
            'score': 7.0,
            'rsi_score': 2.0,
            'bb_score': 1.5,
            'macd_score': 2.0,
            'ma_score': 1.0,
            'volume_score': 0.5
        }
        yield mock

@pytest.fixture
def mock_create_stock_chart():
    """Mock the create_stock_chart function."""
    with patch('trading_advisor.cli.create_stock_chart') as mock:
        mock.return_value = "output/charts/AAPL_chart.html"
        yield mock

@pytest.fixture
def mock_create_score_breakdown():
    """Mock the create_score_breakdown function."""
    with patch('trading_advisor.cli.create_score_breakdown') as mock:
        mock.return_value = "output/charts/AAPL_score.html"
        yield mock

def test_version():
    """Test version command."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "Trading Advisor v" in result.stdout

def test_analyze_help():
    """Test analyze command help."""
    result = runner.invoke(app, ["analyze", "--help"])
    assert result.exit_code == 0
    assert "Analyze stocks and generate trading advice" in result.stdout

@patch('trading_advisor.cli.ensure_data_dir')
@patch('trading_advisor.cli.load_tickers')
@patch('trading_advisor.cli.load_positions')
@patch('trading_advisor.cli.download_stock_data')
@patch('trading_advisor.cli.analyze_stock')
@patch('trading_advisor.cli.generate_technical_summary')
@patch('trading_advisor.cli.generate_structured_data')
@patch('trading_advisor.cli.generate_report')
@patch('trading_advisor.cli.console')
@patch('trading_advisor.cli.Progress')
def test_analyze_command(
    mock_progress,
    mock_console,
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
    
    # Mock the progress bar
    mock_progress_instance = MagicMock()
    mock_progress.return_value.__enter__.return_value = mock_progress_instance
    
    # Return a non-empty DataFrame for download_stock_data
    df = pd.DataFrame({
        'Open': [100.0] * 20,
        'High': [105.0] * 20,
        'Low': [95.0] * 20,
        'Close': [102.0] * 20,
        'Volume': [1000000] * 20,
        'RSI': [65.0] * 20,
        'MACD': [2.0] * 20,
        'MACD_Signal': [1.5] * 20,
        'MACD_Hist': [0.5] * 20,
        'BB_Upper': [105.0] * 20,
        'BB_Lower': [95.0] * 20,
        'BB_Middle': [100.0] * 20,
        'SMA_20': [101.0] * 20,
        'SMA_50': [100.0] * 20,
        'SMA_200': [99.0] * 20
    }, index=pd.date_range(start='2024-01-01', periods=20, freq='D'))
    mock_download_stock_data.return_value = df

    mock_analyze_stock.side_effect = lambda *args, **kwargs: (7.5, {"rsi": 1.0}, None)
    mock_generate_technical_summary.side_effect = lambda *args, **kwargs: "Test summary"
    def structured_data_side_effect(ticker, *args, **kwargs):
        scores = {"AAPL": 7.5, "MSFT": 8.0, "GOOGL": 6.5}
        return {
            "ticker": ticker,
            "score": {"total": scores[ticker]},
            "price_data": {"current": 150.0},
            "technical_indicators": {"rsi": 65.0},
            "summary": "Test summary",
            "position": {"quantity": 100} if ticker == "AAPL" else None
        }
    mock_generate_structured_data.side_effect = structured_data_side_effect

    mock_generate_report.return_value = "Test report"
    mock_console.print = MagicMock()
    
    # Test with all options
    result = runner.invoke(app, [
        "analyze",
        "--tickers", str(mock_tickers_file),
        "--positions", str(mock_positions_file),
        "--top-n", "3",
        "--output", "report.md",
        "--save-json", "analysis.json",
        "--history-days", "50"
    ], catch_exceptions=False)
    
    assert result.exit_code == 0
    try:
        mock_console.print.assert_called_once_with("Test report")
    except AssertionError as e:
        print("\nCLI output for test_analyze_command:\n", result.output)
        raise
    
    # Verify mock calls
    mock_ensure_data_dir.assert_called_once()
    mock_load_tickers.assert_called_once_with(str(mock_tickers_file))
    mock_load_positions.assert_called_once_with(Path(mock_positions_file))
    assert mock_download_stock_data.call_count == 3  # One for each ticker
    assert mock_analyze_stock.call_count == 3
    assert mock_generate_technical_summary.call_count == 3
    assert mock_generate_structured_data.call_count == 3
    mock_generate_report.assert_called_once()

@patch('trading_advisor.cli.ensure_data_dir')
@patch('trading_advisor.cli.load_tickers')
@patch('trading_advisor.cli.load_positions')
@patch('trading_advisor.cli.download_stock_data')
@patch('trading_advisor.cli.analyze_stock')
@patch('trading_advisor.cli.generate_technical_summary')
@patch('trading_advisor.cli.generate_structured_data')
@patch('trading_advisor.cli.generate_report')
@patch('trading_advisor.cli.console')
@patch('trading_advisor.cli.Progress')
def test_analyze_positions_only(
    mock_progress,
    mock_console,
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
    
    # Mock the progress bar
    mock_progress_instance = MagicMock()
    mock_progress.return_value.__enter__.return_value = mock_progress_instance
    
    df = pd.DataFrame({
        'Open': [100.0] * 20,
        'High': [105.0] * 20,
        'Low': [95.0] * 20,
        'Close': [102.0] * 20,
        'Volume': [1000000] * 20,
        'RSI': [65.0] * 20,
        'MACD': [2.0] * 20,
        'MACD_Signal': [1.5] * 20,
        'MACD_Hist': [0.5] * 20,
        'BB_Upper': [105.0] * 20,
        'BB_Lower': [95.0] * 20,
        'BB_Middle': [100.0] * 20,
        'SMA_20': [101.0] * 20,
        'SMA_50': [100.0] * 20,
        'SMA_200': [99.0] * 20
    }, index=pd.date_range(start='2024-01-01', periods=20, freq='D'))
    mock_download_stock_data.return_value = df

    mock_analyze_stock.side_effect = lambda *args, **kwargs: (7.5, {"rsi": 1.0}, None)
    mock_generate_technical_summary.side_effect = lambda *args, **kwargs: "Test summary"
    def structured_data_side_effect(ticker, *args, **kwargs):
        scores = {"AAPL": 7.5, "MSFT": 8.0, "GOOGL": 6.5}
        return {
            "ticker": ticker,
            "score": {"total": scores[ticker]},
            "price_data": {"current": 150.0},
            "technical_indicators": {"rsi": 65.0},
            "summary": "Test summary",
            "position": {"quantity": 100} if ticker == "AAPL" else None
        }
    mock_generate_structured_data.side_effect = structured_data_side_effect

    mock_generate_report.return_value = "Test report"
    mock_console.print = MagicMock()
    
    # Test with positions-only
    result = runner.invoke(app, [
        "analyze",
        "--tickers", str(mock_tickers_file),
        "--positions", str(mock_positions_file),
        "--positions-only"
    ], catch_exceptions=False)
    
    assert result.exit_code == 0
    try:
        mock_console.print.assert_called_once_with("Test report")
    except AssertionError as e:
        print("\nCLI output for test_analyze_positions_only:\n", result.output)
        raise
    
    # Verify mock calls
    mock_ensure_data_dir.assert_called_once()
    mock_load_tickers.assert_called_once_with(str(mock_tickers_file))
    mock_load_positions.assert_called_once_with(Path(mock_positions_file))
    assert mock_download_stock_data.call_count == 3
    assert mock_analyze_stock.call_count == 3
    assert mock_generate_technical_summary.call_count == 3
    assert mock_generate_structured_data.call_count == 3
    mock_generate_report.assert_called_once()

def test_chart_command(
    mock_download_stock_data,
    mock_analyze_stock,
    mock_create_stock_chart,
    mock_create_score_breakdown
):
    """Test the chart command."""
    result = runner.invoke(app, ["chart", "AAPL"])
    
    # Check command execution
    assert result.exit_code == 0
    
    # Check that functions were called with correct arguments
    mock_download_stock_data.assert_called_once_with("AAPL", days=100)
    mock_analyze_stock.assert_called_once()
    mock_create_stock_chart.assert_called_once()
    mock_create_score_breakdown.assert_called_once()
    
    # Check output messages
    assert "Downloading data for AAPL" in result.stdout
    assert "Analyzing stock" in result.stdout
    assert "Generating charts" in result.stdout
    assert "Charts generated successfully" in result.stdout
    assert "Technical Analysis Chart: output/charts/AAPL_chart.html" in result.stdout
    assert "Score Breakdown Chart: output/charts/AAPL_score.html" in result.stdout

def test_chart_command_custom_days(
    mock_download_stock_data,
    mock_analyze_stock,
    mock_create_stock_chart,
    mock_create_score_breakdown
):
    """Test the chart command with custom days parameter."""
    result = runner.invoke(app, ["chart", "AAPL", "--days", "50"])
    
    # Check command execution
    assert result.exit_code == 0
    
    # Check that download_stock_data was called with correct days
    mock_download_stock_data.assert_called_once_with("AAPL", days=50)

def test_chart_command_custom_output_dir(
    mock_download_stock_data,
    mock_analyze_stock,
    mock_create_stock_chart,
    mock_create_score_breakdown
):
    """Test the chart command with custom output directory."""
    result = runner.invoke(app, ["chart", "AAPL", "--output-dir", "custom/charts"])
    
    # Check command execution
    assert result.exit_code == 0
    
    # Check that chart functions were called with correct output directory
    mock_create_stock_chart.assert_called_once()
    mock_create_score_breakdown.assert_called_once()
    
    # Get the actual arguments used in the calls
    stock_chart_args = mock_create_stock_chart.call_args[0]  # positional args
    score_breakdown_args = mock_create_score_breakdown.call_args[0]  # positional args
    
    # Check that output_dir was passed correctly as the last positional argument
    assert stock_chart_args[-1] == "custom/charts"
    assert score_breakdown_args[-1] == "custom/charts"

def test_chart_command_error_handling(mock_download_stock_data):
    """Test error handling in the chart command."""
    # Make download_stock_data raise an exception
    mock_download_stock_data.side_effect = Exception("Test error")
    
    result = runner.invoke(app, ["chart", "AAPL"])
    
    # Check that command failed with error message
    assert result.exit_code == 1
    assert "Error generating charts: Test error" in result.stdout 