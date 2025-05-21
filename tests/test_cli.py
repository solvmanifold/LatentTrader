"""Tests for the CLI module."""

import pytest
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
import pandas as pd
import json
import tempfile
import os

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
        "score": {"total": 7.5},
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
        "score": {"total": 7.5},
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

@patch('trading_advisor.cli.load_positions', return_value={"AAPL": {"quantity": 100}})
def test_chart_command(
    mock_load_positions,
    mock_download_stock_data,
    mock_analyze_stock,
    mock_create_stock_chart,
    mock_create_score_breakdown
):
    mock_analyze_stock.return_value = (7.0, {"rsi": 2.0, "bb": 1.5, "macd": 2.0, "ma": 1.0, "volume": 0.5}, None)
    result = runner.invoke(app, ["chart", "AAPL"])
    if result.exit_code != 0:
        print("STDOUT:", result.stdout)
    assert result.exit_code == 0
    # Check that functions were called with correct arguments
    mock_download_stock_data.assert_called_once_with("AAPL", history_days=100)
    mock_analyze_stock.assert_called_once()
    mock_create_stock_chart.assert_called_once()
    stock_chart_args = mock_create_stock_chart.call_args[1]  # keyword args
    assert 'df' in stock_chart_args
    assert 'ticker' in stock_chart_args
    assert 'indicators' in stock_chart_args
    assert 'output_dir' in stock_chart_args
    mock_create_score_breakdown.assert_called_once()
    score_breakdown_args = mock_create_score_breakdown.call_args[1]  # keyword args
    assert 'ticker' in score_breakdown_args
    assert 'score' in score_breakdown_args
    assert 'indicators' in score_breakdown_args
    assert 'output_dir' in score_breakdown_args
    assert "Charts generated successfully" in result.stdout
    assert "output/charts/AAPL_chart.html" in result.stdout
    assert "output/charts/AAPL_score.html" in result.stdout

@patch('trading_advisor.cli.load_positions', return_value={"AAPL": {"quantity": 100}})
def test_chart_command_custom_days(
    mock_load_positions,
    mock_download_stock_data,
    mock_analyze_stock,
    mock_create_stock_chart,
    mock_create_score_breakdown
):
    mock_analyze_stock.return_value = (7.0, {"rsi": 2.0, "bb": 1.5, "macd": 2.0, "ma": 1.0, "volume": 0.5}, None)
    result = runner.invoke(app, ["chart", "AAPL", "--days", "50"])
    if result.exit_code != 0:
        print("STDOUT:", result.stdout)
    assert result.exit_code == 0
    mock_download_stock_data.assert_called_once_with("AAPL", history_days=50)
    mock_create_stock_chart.assert_called_once()
    stock_chart_args = mock_create_stock_chart.call_args[1]  # keyword args
    assert 'df' in stock_chart_args
    assert 'ticker' in stock_chart_args
    assert 'indicators' in stock_chart_args
    assert 'output_dir' in stock_chart_args

@patch('trading_advisor.cli.load_positions', return_value={"AAPL": {"quantity": 100}})
def test_chart_command_custom_output_dir(
    mock_load_positions,
    mock_download_stock_data,
    mock_analyze_stock,
    mock_create_stock_chart,
    mock_create_score_breakdown
):
    mock_analyze_stock.return_value = (7.0, {"rsi": 2.0, "bb": 1.5, "macd": 2.0, "ma": 1.0, "volume": 0.5}, None)
    result = runner.invoke(app, ["chart", "AAPL", "--output-dir", "custom/charts"])
    if result.exit_code != 0:
        print("STDOUT:", result.stdout)
    assert result.exit_code == 0
    mock_create_stock_chart.assert_called_once()
    mock_create_score_breakdown.assert_called_once()
    stock_chart_args = mock_create_stock_chart.call_args[1]  # keyword args
    score_breakdown_args = mock_create_score_breakdown.call_args[1]  # keyword args
    assert 'df' in stock_chart_args
    assert 'ticker' in stock_chart_args
    assert 'indicators' in stock_chart_args
    assert stock_chart_args['output_dir'] == Path("custom/charts")
    assert 'ticker' in score_breakdown_args
    assert 'score' in score_breakdown_args
    assert 'indicators' in score_breakdown_args
    assert score_breakdown_args['output_dir'] == Path("custom/charts")

def test_prompt_command():
    """Test the prompt command."""
    # Create a temporary JSON file with test data
    test_data = {
        "timestamp": "2024-01-01T00:00:00",
        "positions": [{
            "ticker": "AAPL",
            "score": {"total": 7.5, "details": {"rsi": 2.0, "bb": 1.5, "macd": 2.0, "ma": 1.0, "volume": 0.5}},
            "technical_indicators": {
                "rsi": 65.0,
                "macd": {"value": 2.0, "signal": 1.5, "histogram": 0.5},
                "bollinger_bands": {"upper": 105.0, "lower": 95.0, "middle": 100.0},
                "moving_averages": {"sma_20": 101.0}
            },
            "position": {"quantity": 100, "cost_basis": 150.0, "gain_pct": 7.14, "account_pct": 25.0},
            "price_data": {
                "current_price": 100.0,
                "price_change": 1.0,
                "price_change_pct": 1.0,
                "volume": 1000000,
                "volume_change": 100000,
                "volume_change_pct": 10.0
            },
            "analyst_targets": {
                "current_price": 100.0,
                "median_target": 120.0,
                "low_target": 100.0,
                "high_target": 140.0
            }
        }],
        "new_picks": [
            {
                "ticker": "MSFT",
                "score": {"total": 8.0, "details": {"rsi": 2.5, "bb": 1.5, "macd": 2.5, "ma": 1.0, "volume": 0.5}},
                "technical_indicators": {
                    "rsi": 70.0,
                    "macd": {"value": 2.5, "signal": 2.0, "histogram": 0.5},
                    "bollinger_bands": {"upper": 110.0, "lower": 90.0, "middle": 100.0},
                    "moving_averages": {"sma_20": 102.0}
                },
                "price_data": {
                    "current_price": 200.0,
                    "price_change": 2.0,
                    "price_change_pct": 1.0,
                    "volume": 2000000,
                    "volume_change": 200000,
                    "volume_change_pct": 10.0
                },
                "analyst_targets": {
                    "current_price": 200.0,
                    "median_target": 220.0,
                    "low_target": 200.0,
                    "high_target": 240.0
                }
            },
            {
                "ticker": "GOOGL",
                "score": {"total": 7.0, "details": {"rsi": 2.0, "bb": 1.5, "macd": 2.0, "ma": 1.0, "volume": 0.5}},
                "technical_indicators": {
                    "rsi": 65.0,
                    "macd": {"value": 2.0, "signal": 1.5, "histogram": 0.5},
                    "bollinger_bands": {"upper": 105.0, "lower": 95.0, "middle": 100.0},
                    "moving_averages": {"sma_20": 101.0}
                },
                "price_data": {
                    "current_price": 150.0,
                    "price_change": 1.5,
                    "price_change_pct": 1.0,
                    "volume": 1500000,
                    "volume_change": 150000,
                    "volume_change_pct": 10.0
                },
                "analyst_targets": {
                    "current_price": 150.0,
                    "median_target": 170.0,
                    "low_target": 150.0,
                    "high_target": 190.0
                }
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        json_path = f.name
    
    try:
        # Test standard prompt with default top_n
        result = runner.invoke(app, ["prompt", "--json-file", json_path, "--output", "output/prompt.md"])
        assert result.exit_code == 0
        assert "Prompt written to output/prompt.md" in result.stdout
        
        # Test standard prompt with custom top_n
        result = runner.invoke(app, ["prompt", "--json-file", json_path, "--output", "output/prompt_top1.md", "--top-n", "1"])
        assert result.exit_code == 0
        assert "Prompt written to output/prompt_top1.md" in result.stdout
        
        # Test deep research prompt with default top_n
        result = runner.invoke(app, ["prompt", "--json-file", json_path, "--output", "output/deep_prompt.md", "--deep-research"])
        assert result.exit_code == 0
        assert "Deep research prompt written to output/deep_prompt.md" in result.stdout
        
        # Test deep research prompt with custom top_n
        result = runner.invoke(app, ["prompt", "--json-file", json_path, "--output", "output/deep_prompt_top1.md", "--deep-research", "--top-n", "1"])
        assert result.exit_code == 0
        assert "Deep research prompt written to output/deep_prompt_top1.md" in result.stdout
        
        # Check that all output files were created
        assert Path("output/prompt.md").exists()
        assert Path("output/prompt_top1.md").exists()
        assert Path("output/deep_prompt.md").exists()
        assert Path("output/deep_prompt_top1.md").exists()
        
        # Check the content of the output files
        with open("output/prompt.md") as f:
            content = f.read()
            assert "You are a tactical swing trader" in content
            assert "Current Positions" in content
            assert "New Technical Picks" in content
            assert "MSFT" in content
            assert "GOOGL" in content
            
        with open("output/prompt_top1.md") as f:
            content = f.read()
            assert "You are a tactical swing trader" in content
            assert "Current Positions" in content
            assert "New Technical Picks" in content
            assert "MSFT" in content
            assert "GOOGL" not in content  # Should only include top 1 pick
            
        with open("output/deep_prompt.md") as f:
            content = f.read()
            assert "You are a tactical swing trader" in content
            assert "Current Positions" in content
            assert "New Technical Picks" in content
            assert "MSFT" in content
            assert "GOOGL" in content
            
        with open("output/deep_prompt_top1.md") as f:
            content = f.read()
            assert "You are a tactical swing trader" in content
            assert "Current Positions" in content
            assert "New Technical Picks" in content
            assert "MSFT" in content
            assert "GOOGL" not in content  # Should only include top 1 pick
            
    finally:
        # Clean up
        os.unlink(json_path)
        for file in ["prompt.md", "prompt_top1.md", "deep_prompt.md", "deep_prompt_top1.md"]:
            if Path(f"output/{file}").exists():
                Path(f"output/{file}").unlink()

def test_prompt_command_error_handling():
    """Test error handling in the prompt command."""
    # Test with non-existent JSON file
    result = runner.invoke(app, ["prompt", "--json-file", "nonexistent.json", "--output", "output/prompt.md"])
    assert result.exit_code == 1
    assert "Error generating research prompt" in result.stdout
    
    # Test with invalid JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("invalid json")
        json_path = f.name
    
    try:
        result = runner.invoke(app, ["prompt", "--json-file", json_path, "--output", "output/prompt.md"])
        assert result.exit_code == 1
        assert "Error generating research prompt" in result.stdout
    finally:
        os.unlink(json_path)

def test_analyze_positions_file_missing():
    """Test error if positions file does not exist."""
    result = runner.invoke(app, [
        "analyze",
        "--positions", "nonexistent.csv",
        "--positions-only",
        "--output", "output/should_not_exist.json"
    ])
    assert result.exit_code == 1
    assert "does not exist" in result.stdout or result.stderr

def test_analyze_positions_file_empty(tmp_path):
    """Test warning if positions file exists but is empty or has no valid positions."""
    empty_file = tmp_path / "empty_positions.csv"
    empty_file.write_text("Symbol,Security Type,Qty (Quantity),Price\n")
    result = runner.invoke(app, [
        "analyze",
        "--positions", str(empty_file),
        "--positions-only",
        "--output", "output/should_not_exist.json"
    ])
    assert result.exit_code == 1
    assert "No positions loaded" in result.stdout or result.stderr 