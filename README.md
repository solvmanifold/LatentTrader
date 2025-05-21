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

## Usage

Basic usage:
```bash
trading-advisor analyze --tickers tickers.txt
```

Analyze current positions and generate recommendations:
```bash
trading-advisor analyze --tickers tickers.txt --positions positions.csv
```

Generate both markdown and JSON output:
```bash
trading-advisor analyze --tickers tickers.txt --output report.md --save-json analysis.json
```

### Generate Interactive Charts

To generate technical analysis charts for a specific stock:

```bash
trading-advisor chart AAPL
```

**Options:**

- `--output-dir`, `-o`: Directory to save the charts (default: `output/charts`)
- `--days`, `-d`: Number of days of historical data to include (default: 100)

This command produces two interactive HTML files:
- `{ticker}_chart.html`: A candlestick chart with Bollinger Bands, moving averages, volume, and RSI.
- `{ticker}_score.html`: A breakdown of the technical score components.

### Command Line Options

- `--tickers`, `-t`: Path to file containing ticker symbols, or 'all' for S&P 500
- `--positions`, `-p`: Path to brokerage CSV file containing current positions
- `--output`, `-o`: Path to save the JSON output (default: output/analysis.json)
- `--history-days`: Number of days of historical data to analyze (default: 100)
- `--positions-only`: Only analyze current positions
- `--version`, `-v`: Show version and exit

### New Workflow

The CLI now follows a modular workflow:
1. Run `analyze` to produce a structured JSON output.
2. Use additional commands (to be implemented) to generate markdown reports, prompts, or interactive charts from the JSON.

### JSON Output Structure

```json
{
  "timestamp": "2024-03-20T14:30:00",
  "positions": [
    {
      "ticker": "AAPL",
      "price_data": [
        {
          "Date": "2024-03-20",
          "Open": 175.0,
          "High": 176.0,
          "Low": 174.0,
          "Close": 175.5,
          "Volume": 50000000
        }
      ],
      "technical_indicators": {
        "rsi": [65.5],
        "macd": {
          "macd": [2.5],
          "signal": [1.8],
          "histogram": [0.7]
        },
        "bollinger_bands": {
          "upper": [180.0],
          "middle": [175.0],
          "lower": [170.0]
        },
        "moving_averages": {
          "sma_20": [174.0],
          "sma_50": [172.0],
          "sma_200": [170.0]
        }
      },
      "score": {
        "total": 7.5,
        "details": {
          "rsi": 1.0,
          "bollinger_low": 1.0,
          "bollinger_high": 1.0,
          "macd": 2.0,
          "moving_averages": 2.0,
          "analyst_targets": 1.5
        }
      },
      "position": {
        "quantity": 100,
        "cost_basis": 150.00,
        "market_value": 17550.00,
        "gain_pct": 17.0,
        "account_pct": 25.0
      }
    }
  ],
  "new_picks": [
    {
      "ticker": "NVDA",
      "price_data": [
        {
          "Date": "2024-03-20",
          "Open": 890.0,
          "High": 895.0,
          "Low": 885.0,
          "Close": 890.0,
          "Volume": 30000000
        }
      ],
      "technical_indicators": {
        "rsi": [70.5],
        "macd": {
          "macd": [15.5],
          "signal": [10.8],
          "histogram": [4.7]
        },
        "bollinger_bands": {
          "upper": [900.0],
          "middle": [890.0],
          "lower": [880.0]
        },
        "moving_averages": {
          "sma_20": [888.0],
          "sma_50": [885.0],
          "sma_200": [880.0]
        }
      },
      "score": {
        "total": 8.0,
        "details": {
          "rsi": 1.5,
          "bollinger_low": 1.0,
          "bollinger_high": 1.0,
          "macd": 2.0,
          "moving_averages": 2.0,
          "analyst_targets": 0.5
        }
      }
    }
  ]
}
```

### Test Coverage

The CLI is now fully tested. See `tests/test_analyze_command.py` for details.

## Requirements

- Python 3.8 or higher
- pandas >= 2.0.0
- numpy >= 1.24.0
- yfinance >= 0.2.36
- ta >= 0.10.0
- typer >= 0.9.0
- rich >= 13.0.0

## License

This project is licensed under the MIT License - see the LICENSE file for details. 