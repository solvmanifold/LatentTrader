# Trading Advisor

A Python CLI tool for generating tactical swing trading advice based on technical indicators and analyst targets for S&P 500 stocks.

## Features

- Pulls historical stock data using `yfinance`
- Calculates various technical indicators:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Moving Averages (20, 50, 200-day)
- Incorporates analyst price targets
- Generates markdown-formatted reports with actionable trading playbooks
- Supports current position analysis
- Caches Ticker objects for performance
- Saves structured JSON data for programmatic analysis
- Normalized technical scores (0-10) for easy comparison
- Ranks top setups by confidence × upside

## Installation

### From Source

1. Clone the repository:
   ```bash
   git clone https://github.com/solvmanifold/LatentTrader.git
   cd LatentTrader
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

### From PyPI (Coming Soon)

```bash
pip install trading-advisor
```

## Recent Improvements

- Table values in score breakdown are now robustly sanitized and explicitly converted to strings, preventing HTML tags or formatting issues in both HTML and static image exports.
- Debug print statements have been removed for clean CLI output.
- If you see unexpected <br> tags in PNG/PDF exports, this is a limitation of the Plotly/Kaleido rendering stack; the HTML output will always be correct.
- For best results, always check the HTML output for the most accurate rendering.

## Usage

### Update Data

```bash
trading-advisor update-data --tickers tickers.txt --days 60
```

*Note: If you omit the `--tickers` argument, only tickers with existing feature files will be updated. To update all S&P 500 tickers, use `--tickers all`.*

Options:
- `--tickers`, `-t`: Path to a file containing ticker symbols (required unless you want to update only existing tickers). Use `all` for S&P 500. If omitted, only tickers with existing feature files will be updated.
- `--days`, `-d`: Number of days of historical data to download (default: 60)
- `--features-dir`: Directory to store feature files (default: data/ticker_features)
- `--start-date`: Start date for data collection (optional)
- `--update-sector-mapping`: Force update sector mapping (default: false)
- `--update-tickers/--no-update-tickers`: Update individual ticker features (default: true)
- `--update-market/--no-update-market`: Update market-wide features (default: true)

### Analyze Stocks

```bash
trading-advisor analyze --tickers tickers.txt --positions positions.csv
```

Options:
- `--tickers`, `-t`: Path to a file containing ticker symbols (required unless --positions-only)
- `--positions`, `-p`: Path to a CSV file containing positions (optional)
- `--positions-only`: Analyze only positions (optional)
- `--output`, `-o`: Path to the output JSON file (default: `output/analysis.json`)
- `--days`, `-d`: Number of days of historical data to analyze (default: 100)

### Generate Daily Report

```bash
trading-advisor report-daily --model-name TechnicalScorer --date 2025-05-23 --top-n 6
```

### Generate Daily Research Prompt

```bash
trading-advisor prompt-daily --model-name TechnicalScorer --date 2025-05-23 --top-n 6
```

### Generate Interactive Charts

```bash
trading-advisor chart AAPL MSFT --output-dir output/charts --days 100
```

Options:
- `TICKER ...`: One or more stock ticker symbols (e.g., AAPL MSFT)
- `--output-dir`, `-o`: Directory to save the charts (default: `output/charts`)
- `--days`, `-d`: Number of days of historical data to include (default: 100)
- `--pdf`: Export charts as images and combine into a single PDF (optional)
- `--json`, `-j`: Path to analysis JSON file (optional, for charting analyzed tickers)

### Backtest Strategy

```bash
trading-advisor backtest AAPL MSFT --start-date 2024-01-01 --end-date 2024-02-01 --top-n 2 --hold-days 5 --stop-loss -0.05 --profit-target 0.05
```

Options:
- `TICKER ...`: One or more stock ticker symbols to backtest (e.g., AAPL MSFT)
- `--start-date`: Backtest start date (YYYY-MM-DD, required)
- `--end-date`: Backtest end date (YYYY-MM-DD, required)
- `--top-n`: Number of top picks to buy each week (default: 3)
- `--hold-days`: Max holding period in trading days (default: 10)
- `--stop-loss`: Stop-loss threshold (e.g., -0.10 for -10%, default: -0.10)
- `--profit-target`: Profit target threshold (e.g., 0.10 for +10%, default: 0.10)

The backtest command simulates a weekly top-N strategy with fixed holding periods and stop/profit exits, reporting total return and trade log.

---

## Logging

Trading Advisor logs all activity to both the terminal and a log file:

- **Terminal:** Pretty, colorized logs using Rich for easy reading.
- **File:** All logs are saved to `logs/trading_advisor.log` with automatic rotation (max 10MB per file, up to 5 backup files).
- The `logs/` folder is created automatically if it does not exist.

This ensures you always have a persistent record of all analysis, errors, and activity for debugging or audit purposes.

---

## Test Coverage

- The CLI is fully tested with robust coverage for all commands and error cases.
- See `tests/test_cli.py` for comprehensive CLI tests, including analyze, chart, report-daily, prompt-daily, backtest, and error handling scenarios.

## Requirements

- Python 3.9 or higher
- pandas >= 2.0.0
- numpy >= 1.24.0
- yfinance >= 0.2.36
- ta >= 0.10.0
- typer >= 0.9.0
- rich >= 13.0.0
- plotly >= 5.13.0

## License

This project is licensed under the MIT License - see the LICENSE file for details. 