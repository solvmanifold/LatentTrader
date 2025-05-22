"""Tests for CLI commands."""

import json
import pytest
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta

from trading_advisor.cli import app

runner = CliRunner()

@pytest.fixture
def mock_stock_data():
    """Create mock stock data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
    data = {
        'Open': [100.0] * 60,
        'High': [105.0] * 60,
        'Low': [95.0] * 60,
        'Close': [102.0] * 60,
        'Volume': [1000000] * 60
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def mock_analysis_data():
    """Create mock analysis data for testing."""
    return {
        "timestamp": datetime.now().isoformat(),
        "positions": [
            {
                "ticker": "AAPL",
                "score": {"total": 7.5, "details": {"rsi": 1.0, "macd": 1.0}},
                "price_data": {
                    "current_price": 150.0,
                    "price_change": 2.0,
                    "price_change_pct": 1.3,
                    "volume": 1000000,
                    "volume_change": 10000,
                    "volume_change_pct": 1.0
                },
                "technical_indicators": {
                    "rsi": 65.0,
                    "macd": {"value": 2.0, "signal": 1.5, "histogram": 0.5},
                    "bollinger_bands": {"upper": 155.0, "middle": 150.0, "lower": 145.0},
                    "moving_averages": {"sma_20": 148.0}
                },
                "analyst_targets": {"median_target": 160.0, "low_target": 150.0, "high_target": 170.0, "current_price": 150.0},
                "position": {"quantity": 100, "price": 150.0, "market_value": 15000.0, "cost_basis": 14000.0, "gain_pct": 7.14, "account_pct": 25.0}
            }
        ],
        "new_picks": [
            {
                "ticker": "MSFT",
                "score": {"total": 8.0, "details": {"rsi": 1.0, "macd": 1.0}},
                "price_data": {
                    "current_price": 300.0,
                    "price_change": 5.0,
                    "price_change_pct": 1.7,
                    "volume": 2000000,
                    "volume_change": 20000,
                    "volume_change_pct": 1.0
                },
                "technical_indicators": {
                    "rsi": 70.0,
                    "macd": {"value": 2.5, "signal": 2.0, "histogram": 0.7},
                    "bollinger_bands": {"upper": 310.0, "middle": 300.0, "lower": 290.0},
                    "moving_averages": {"sma_20": 295.0}
                },
                "analyst_targets": {"median_target": 320.0, "low_target": 310.0, "high_target": 330.0, "current_price": 300.0}
            }
        ]
    }

def test_version():
    """Test version command."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "Trading Advisor v" in result.stdout

def test_analyze_command(mock_stock_data, tmp_path):
    """Test analyze command."""
    # Create test files
    tickers_file = tmp_path / "tickers.txt"
    tickers_file.write_text("AAPL\nMSFT")
    
    positions_file = tmp_path / "positions.csv"
    positions_file.write_text("""Header
Empty Row
Symbol,Security Type,Qty (Quantity),Price,Mkt Val (Market Value),Cost Basis,Gain % (Gain/Loss %),% of Acct (% of Account)
AAPL,Equity,100,$150.00,$15000.00,$14000.00,7.14%,25.0%""")
    
    output_file = tmp_path / "analysis.json"
    
    with patch('trading_advisor.data.download_stock_data', return_value=mock_stock_data), \
         patch('trading_advisor.analysis.analyze_stock', return_value=(7.5, {"rsi": 1.0, "macd": 1.0}, None)):
        result = runner.invoke(app, [
            "analyze",
            "--tickers", str(tickers_file),
            "--positions", str(positions_file),
            "--output", str(output_file)
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        
        # Verify JSON output
        with open(output_file) as f:
            data = json.load(f)
            assert "positions" in data
            assert "new_picks" in data
            assert len(data["positions"]) > 0
            assert len(data["new_picks"]) > 0

def test_chart_command(mock_stock_data, tmp_path):
    """Test chart command."""
    output_dir = tmp_path / "charts"
    
    with patch('trading_advisor.data.download_stock_data', return_value=mock_stock_data):
        result = runner.invoke(app, [
            "chart",
            "AAPL",
            "MSFT",
            "--output-dir", str(output_dir)
        ])
        
        assert result.exit_code == 0
        assert output_dir.exists()
        assert len(list(output_dir.glob("*.html"))) > 0

def test_chart_command_json(mock_stock_data, mock_analysis_data, tmp_path):
    """Test chart command with JSON input."""
    output_dir = tmp_path / "charts"
    json_file = tmp_path / "analysis.json"
    
    # Write mock analysis data
    with open(json_file, 'w') as f:
        json.dump(mock_analysis_data, f)
    
    with patch('trading_advisor.data.download_stock_data', return_value=mock_stock_data):
        result = runner.invoke(app, [
            "chart",
            "--json", str(json_file),
            "--output-dir", str(output_dir)
        ])
        
        assert result.exit_code == 0
        assert output_dir.exists()
        assert len(list(output_dir.glob("*.html"))) > 0

def test_report_command(mock_analysis_data, tmp_path):
    """Test report command."""
    json_file = tmp_path / "analysis.json"
    output_file = tmp_path / "report.md"
    
    # Write mock analysis data
    with open(json_file, 'w') as f:
        json.dump(mock_analysis_data, f)
    
    with patch('trading_advisor.output.generate_report', return_value="# Test Report\n\nThis is a test report."):
        result = runner.invoke(app, [
            "report",
            "--json", str(json_file),
            "--output", str(output_file)
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        assert output_file.read_text().strip() != ""

def test_prompt_command(mock_analysis_data, tmp_path):
    """Test prompt command."""
    json_file = tmp_path / "analysis.json"
    output_file = tmp_path / "prompt.txt"
    
    # Write mock analysis data
    with open(json_file, 'w') as f:
        json.dump(mock_analysis_data, f)
    
    with patch('trading_advisor.output.generate_research_prompt', return_value="Test prompt content"), \
         patch('trading_advisor.output.generate_deep_research_prompt', return_value="Test deep prompt content"):
        result = runner.invoke(app, [
            "prompt",
            "--json-file", str(json_file),
            "--output", str(output_file)
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        assert output_file.read_text().strip() != ""

def test_backtest_command(mock_stock_data, tmp_path):
    """Test backtest command."""
    with patch('trading_advisor.data.download_stock_data', return_value=mock_stock_data):
        result = runner.invoke(app, [
            "backtest",
            "AAPL", "MSFT",
            "--start-date", "2024-01-01",
            "--end-date", "2024-02-01",
            "--top-n", "2",
            "--hold-days", "5",
            "--stop-loss", "-0.05",
            "--profit-target", "0.05"
        ])
        
        assert result.exit_code == 0
        assert "Backtest complete" in result.stdout
        assert "Total return" in result.stdout

def test_analyze_command_error_handling(tmp_path):
    """Test analyze command error handling."""
    # Test with non-existent tickers file
    result = runner.invoke(app, [
        "analyze",
        "--tickers", "nonexistent.txt"
    ])
    assert result.exit_code != 0
    
    # Test with invalid positions file
    positions_file = tmp_path / "positions.csv"
    positions_file.write_text("invalid,csv,format")
    
    result = runner.invoke(app, [
        "analyze",
        "--positions", str(positions_file)
    ])
    assert result.exit_code != 0

def test_chart_command_error_handling(tmp_path):
    """Test chart command error handling."""
    # Test with non-existent JSON file
    result = runner.invoke(app, [
        "chart",
        "--json", "nonexistent.json"
    ])
    assert result.exit_code != 0
    
    # Test with invalid JSON file
    json_file = tmp_path / "invalid.json"
    json_file.write_text("invalid json")
    
    result = runner.invoke(app, [
        "chart",
        "--json", str(json_file)
    ])
    assert result.exit_code != 0

def test_report_command_error_handling(tmp_path):
    """Test report command error handling."""
    # Test with non-existent JSON file
    result = runner.invoke(app, [
        "report",
        "--json", "nonexistent.json"
    ])
    assert result.exit_code != 0
    
    # Test with invalid JSON file
    json_file = tmp_path / "invalid.json"
    json_file.write_text("invalid json")
    
    result = runner.invoke(app, [
        "report",
        "--json", str(json_file)
    ])
    assert result.exit_code != 0 