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

### Command Line Options

- `--tickers`, `-t`: Path to file containing ticker symbols, or 'all' for S&P 500
- `--positions`, `-p`: Path to brokerage CSV file containing current positions
- `--top-n`: Number of top stocks to recommend (default: 5)
- `--output`, `-o`: Path to save the markdown report
- `--save-json`: Path to save structured JSON data
- `--history-days`: Number of days of historical data to analyze (default: 100)
- `--positions-only`: Only analyze current positions
- `--no-positions`: Skip analysis of current positions
- `--version`, `-v`: Show version and exit

## Example Output

### Markdown Report

```markdown
# Trading Advisor Report

Generated on: 2024-03-20 14:30:00

## Current Positions

### AAPL
✅ Action: Hold
🎯 Entry strategy: Maintain current position
🛑 Stop-loss level: $170.00 (3% below current)
💰 Profit-taking strategy: $180.00 target
🔍 Confidence level: High
🧠 Rationale: Strong technicals with bullish MACD

## New Technical Picks

### NVDA
✅ Action: Buy Now
🎯 Entry strategy: Limit order at $880.00
🛑 Stop-loss level: $850.00 (3.4% below entry)
💰 Profit-taking strategy: $950.00 target
🔍 Confidence level: High
🧠 Rationale: Bullish setup with strong volume

## Top Setups
1. NVDA: High confidence × 8% upside = Strong momentum with clear entry
2. AAPL: High confidence × 5% upside = Solid technicals with analyst support
```

### JSON Output Structure

```json
{
  "timestamp": "2024-03-20T14:30:00",
  "positions": [
    {
      "ticker": "AAPL",
      "price_data": {
        "current_price": 175.50,
        "price_change": 4.25,
        "price_change_pct": 2.5,
        "volume": 50000000,
        "volume_change": 1000000,
        "volume_change_pct": 2.0
      },
      "technical_indicators": {
        "rsi": 65.5,
        "macd": {
          "value": 2.5,
          "signal": 1.8,
          "histogram": 0.7
        },
        "bollinger_bands": {
          "upper": 180.0,
          "middle": 175.0,
          "lower": 170.0
        },
        "moving_averages": {
          "sma_20": 174.0,
          "sma_50": 172.0,
          "sma_200": 170.0
        }
      },
      "score": {
        "total": 7.5,
        "details": {
          "rsi": 1.0,
          "bollinger": 1.0,
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
      "price_data": {
        "current_price": 890.00,
        "price_change": 45.00,
        "price_change_pct": 5.2,
        "volume": 30000000,
        "volume_change": 2000000,
        "volume_change_pct": 7.1
      },
      "technical_indicators": {
        "rsi": 70.5,
        "macd": {
          "value": 15.5,
          "signal": 10.8,
          "histogram": 4.7
        },
        "bollinger_bands": {
          "upper": 900.0,
          "middle": 880.0,
          "lower": 860.0
        },
        "moving_averages": {
          "sma_20": 875.0,
          "sma_50": 850.0,
          "sma_200": 800.0
        }
      },
      "score": {
        "total": 8.3,
        "details": {
          "rsi": 1.5,
          "bollinger": 1.5,
          "macd": 2.0,
          "moving_averages": 2.0,
          "analyst_targets": 1.3
        }
      }
    }
  ]
}
```

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