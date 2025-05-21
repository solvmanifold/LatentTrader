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
        dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
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
        mock.return_value = pd.DataFrame(data, index=dates)
        yield mock

@pytest.fixture
def mock_analyze_stock():
    """Mock the analyze_stock function."""
    with patch('trading_advisor.cli.analyze_stock') as mock:
        mock.return_value = (7.0, {"rsi": 2.0, "bb": 1.5, "macd": 2.0, "ma": 1.0, "volume": 0.5}, None)
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

@pytest.fixture
def mock_generate_structured_data():
    """Mock the generate_structured_data function."""
    with patch('trading_advisor.cli.generate_structured_data') as mock:
        def side_effect(ticker, df, score, score_details, analyst_targets=None, position=None):
            return {
                "ticker": ticker,
                "score": {"total": score, "details": score_details},
                "price_data": {"current": 150.0},
                "technical_indicators": {"rsi": 65.0},
                "summary": "Test summary",
                "position": position,
                "analyst_targets": analyst_targets
            }
        mock.side_effect = side_effect
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
    assert "Analyze stocks and output structured JSON data." in result.stdout

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
    }, index=pd.date_range(start='2024-01-01', periods=60, freq='D'))
    mock_download_stock_data.return_value = df

    mock_analyze_stock.side_effect = lambda *args, **kwargs: (7.5, {"rsi": 1.0}, None)
    mock_generate_technical_summary.side_effect = lambda *args, **kwargs: "Test summary"
    mock_generate_report.return_value = "Test report"
    mock_console.print = MagicMock()
    
    # Make generate_structured_data return JSON serializable data
    mock_generate_structured_data.return_value = {
        "ticker": "AAPL",
        "score": 7.5,
        "price_data": {
            "current_price": 102.0,
            "price_change": 2.0,
            "volume_change": 0.0
        },
        "technical_indicators": {
            "rsi": 65.0,
            "macd": 2.0,
            "bb_upper": 105.0,
            "bb_lower": 95.0
        },
        "summary": "Test summary",
        "position": {"quantity": 100},
        "analyst_targets": None
    }
    
    # Test with all options
    command_args = [
        "analyze",
        "--tickers", str(mock_tickers_file),
        "--positions", str(mock_positions_file)
    ]
    result = runner.invoke(app, command_args, catch_exceptions=False)
    
    if "Usage: trading-advisor" in result.stdout:
        print(f"DEBUG: Command args: {command_args}")
        print("DEBUG: CLI output (help menu):\n", result.stdout)
    
    assert result.exit_code == 0
    assert "Analysis complete. Results written to output/analysis.json" in result.stdout
    
    # Verify mock calls
    mock_load_tickers.assert_called_once()
    assert mock_load_tickers.call_args[0][0] == mock_tickers_file
    mock_load_positions.assert_called_once_with(Path(mock_positions_file))
    assert mock_download_stock_data.call_count == 3  # One for each ticker
    assert mock_analyze_stock.call_count == 3
    assert mock_generate_structured_data.call_count == 3  # One for each ticker

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
    }, index=pd.date_range(start='2024-01-01', periods=60, freq='D'))
    mock_download_stock_data.return_value = df

    mock_analyze_stock.side_effect = lambda *args, **kwargs: (7.5, {"rsi": 1.0}, None)
    mock_generate_technical_summary.side_effect = lambda *args, **kwargs: "Test summary"
    mock_generate_report.return_value = "Test report"
    mock_console.print = MagicMock()
    
    # Make generate_structured_data return JSON serializable data
    mock_generate_structured_data.return_value = {
        "ticker": "AAPL",
        "score": 7.5,
        "price_data": {
            "current_price": 102.0,
            "price_change": 2.0,
            "volume_change": 0.0
        },
        "technical_indicators": {
            "rsi": 65.0,
            "macd": 2.0,
            "bb_upper": 105.0,
            "bb_lower": 95.0
        },
        "summary": "Test summary",
        "position": {"quantity": 100},
        "analyst_targets": None
    }
    
    # Test with positions-only
    command_args = [
        "analyze",
        "--tickers", str(mock_tickers_file),
        "--positions", str(mock_positions_file),
        "--positions-only"
    ]
    result = runner.invoke(app, command_args, catch_exceptions=False)
    
    if "Usage: trading-advisor" in result.stdout:
        print(f"DEBUG: Command args: {command_args}")
        print("DEBUG: CLI output (help menu):\n", result.stdout)
    
    assert result.exit_code == 0
    assert "Analysis complete. Results written to output/analysis.json" in result.stdout
    
    # Verify mock calls
    mock_load_tickers.assert_called_once()
    assert mock_load_tickers.call_args[0][0] == mock_tickers_file
    mock_load_positions.assert_called_once_with(Path(mock_positions_file))
    assert mock_download_stock_data.call_count == 1  # Only one position
    assert mock_analyze_stock.call_count == 1
    assert mock_generate_structured_data.call_count == 1  # Only one position

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
    mock_download_stock_data.assert_called_once_with("AAPL", history_days=100)
    mock_analyze_stock.assert_called_once()
    
    # Check create_stock_chart was called with correct parameter order
    mock_create_stock_chart.assert_called_once()
    stock_chart_args = mock_create_stock_chart.call_args[1]  # keyword args
    assert 'df' in stock_chart_args
    assert 'ticker' in stock_chart_args
    assert 'indicators' in stock_chart_args
    assert 'output_dir' in stock_chart_args
    
    # Check create_score_breakdown was called with correct parameter order
    mock_create_score_breakdown.assert_called_once()
    score_breakdown_args = mock_create_score_breakdown.call_args[1]  # keyword args
    assert 'ticker' in score_breakdown_args
    assert 'score' in score_breakdown_args
    assert 'indicators' in score_breakdown_args
    assert 'output_dir' in score_breakdown_args
    
    # Check output messages
    assert "Charts generated successfully" in result.stdout
    assert "output/charts/AAPL_chart.html" in result.stdout
    assert "output/charts/AAPL_score.html" in result.stdout

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
    mock_download_stock_data.assert_called_once_with("AAPL", history_days=50)
    
    # Check that chart functions were called with correct parameter order
    mock_create_stock_chart.assert_called_once()
    stock_chart_args = mock_create_stock_chart.call_args[1]  # keyword args
    assert 'df' in stock_chart_args
    assert 'ticker' in stock_chart_args
    assert 'indicators' in stock_chart_args
    assert 'output_dir' in stock_chart_args

def test_chart_command_custom_output_dir(
    mock_download_stock_data,
    mock_analyze_stock,
    mock_create_stock_chart,
    mock_create_score_breakdown
):
    """Test the chart command with custom output directory."""
    result = runner.invoke(app, ["chart", "AAPL", "--output-dir", "custom/charts"])
    assert result.exit_code == 0
    
    # Check that chart functions were called with correct output directory and parameter order
    mock_create_stock_chart.assert_called_once()
    mock_create_score_breakdown.assert_called_once()
    
    # Get the actual arguments used in the calls
    stock_chart_args = mock_create_stock_chart.call_args[1]  # keyword args
    score_breakdown_args = mock_create_score_breakdown.call_args[1]  # keyword args
    
    # Check parameter order and values
    assert 'df' in stock_chart_args
    assert 'ticker' in stock_chart_args
    assert 'indicators' in stock_chart_args
    assert stock_chart_args['output_dir'] == Path("custom/charts")
    
    assert 'ticker' in score_breakdown_args
    assert 'score' in score_breakdown_args
    assert 'indicators' in score_breakdown_args
    assert score_breakdown_args['output_dir'] == Path("custom/charts")

def test_chart_command_error_handling(mock_download_stock_data):
    """Test error handling in the chart command."""
    # Make download_stock_data raise an exception
    mock_download_stock_data.side_effect = Exception("Test error")
    
    result = runner.invoke(app, ["chart", "AAPL"])
    
    # Check that command failed with error message
    assert result.exit_code == 1
    assert "Error generating charts: Test error" in result.stdout 