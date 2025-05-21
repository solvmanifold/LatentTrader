import json
import pytest
from typer.testing import CliRunner
from trading_advisor.cli import app

@pytest.fixture
def mock_json_file(tmp_path):
    json_data = {
        "timestamp": "2023-10-01T12:00:00",
        "positions": [
            {
                "ticker": "AAPL",
                "score": {"total": 8, "details": {"rsi": 70, "macd": 1.5}},
                "price_data": [{"Close": 150.0}]
            }
        ],
        "new_picks": [
            {
                "ticker": "MSFT",
                "score": {"total": 6, "details": {"rsi": 60, "macd": 0.5}},
                "price_data": [{"Close": 200.0}]
            }
        ]
    }
    json_file = tmp_path / "analysis.json"
    json_file.write_text(json.dumps(json_data))
    return json_file

@pytest.fixture
def mock_empty_json_file(tmp_path):
    json_data = {"timestamp": "2023-10-01T12:00:00", "positions": [], "new_picks": []}
    json_file = tmp_path / "empty_analysis.json"
    json_file.write_text(json.dumps(json_data))
    return json_file

def test_report_command_generates_markdown(tmp_path):
    # Prepare a sample analysis.json file
    data = {
        "timestamp": "2024-01-01T00:00:00",
        "positions": [
            {
                "ticker": "AAPL",
                "score": {"total": 7.5, "details": {"rsi": 1.0}},
                "price_data": {"current": 150.0},
                "technical_indicators": {"rsi": 65.0},
                "summary": "Test summary",
                "position": {"quantity": 100},
                "analyst_targets": None
            }
        ],
        "new_picks": [
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
    }
    json_file = tmp_path / "analysis.json"
    with open(json_file, "w") as f:
        json.dump(data, f)
    output_file = tmp_path / "report.md"
    runner = CliRunner()
    result = runner.invoke(app, ["report", "--json", str(json_file), "--output", str(output_file)])
    assert result.exit_code == 0
    assert "Trading Advisor Report" in result.stdout
    assert "AAPL" in result.stdout
    assert "MSFT" in result.stdout
    assert output_file.exists()
    content = output_file.read_text()
    assert "Trading Advisor Report" in content
    assert "AAPL" in content
    assert "MSFT" in content

def test_report_command_handles_empty_data(tmp_path):
    # Prepare an empty analysis.json file
    data = {"timestamp": "2024-01-01T00:00:00", "positions": [], "new_picks": []}
    json_file = tmp_path / "analysis.json"
    with open(json_file, "w") as f:
        json.dump(data, f)
    output_file = tmp_path / "report.md"
    runner = CliRunner()
    result = runner.invoke(app, ["report", "--json", str(json_file), "--output", str(output_file)])
    assert result.exit_code == 0
    assert "Trading Advisor Report" in result.stdout
    assert output_file.exists()
    content = output_file.read_text()
    assert "Trading Advisor Report" in content

def test_report_command_invalid_json(tmp_path):
    # Prepare an invalid JSON file
    json_file = tmp_path / "invalid.json"
    json_file.write_text("not a json")
    output_file = tmp_path / "report.md"
    runner = CliRunner()
    result = runner.invoke(app, ["report", "--json", str(json_file), "--output", str(output_file)])
    assert result.exit_code == 1
    assert "Error loading JSON data" in result.stdout 