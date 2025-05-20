# Weekly Trading Advisor

A Python CLI tool that generates trading advice based on technical indicators and analyst targets for S&P 500 stocks.

## Features

- Pulls historical stock data using `yfinance`
- Calculates technical indicators (RSI, MACD, Bollinger Bands)
- Incorporates analyst price targets
- Generates markdown-formatted reports
- Supports current position analysis
- Caches Ticker objects for performance
- Saves structured JSON data for programmatic analysis
- Normalized technical scores (0-10 scale)

## Installation

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

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage with tickers file:
```bash
python weekly_trading_advisor.py --tickers tickers.txt
```

Analyze current positions:
```bash
python weekly_trading_advisor.py --positions positions.csv
```

Generate both markdown and JSON output:
```bash
python weekly_trading_advisor.py --tickers tickers.txt --positions positions.csv --output report.md --save-json analysis.json
```

### Command Line Options

- `--top_n`: Number of new tickers to include (default: 5)
- `--output`: Path to write the markdown file (default: stdout)
- `--positions`: Path to brokerage positions CSV
- `--tickers`: Path to tickers.txt file (one ticker per line), or 'all' for all S&P 500 tickers
- `--history_days`: Days of historical data to fetch (default: 100)
- `--include_charts`: Include plots in the summary output (default: False)
- `--positions-only`: Only analyze current positions
- `--no-positions`: Exclude current positions from analysis
- `--save-json`: Path to save structured JSON data

## Example Output

### Markdown Report
```markdown
You are an equity analyst. Here is the technical summary for several S&P 500 stocks.

The first section contains current positions. Please advise whether to hold, sell, or adjust (e.g. trailing stop).
The second section contains new picks flagged by our model this week. Please assess each for trade viability.

---

### 📊 Current Positions (Hold/Sell Guidance)

---

**$AAPL** (Score: 7.5/10)
- Current price: $150.25
- Analyst target: $165.00 median (range: $140.00–$180.00)
- Implied upside: +9.8%
- RSI: 45.5 (neutral)
- MACD: bullish with moderate divergence
- Volume: normal
- MA trend: bullish
- 5d change: +2.5%
- Position: 100 shares @ $145.00, +3.6%, 15.0% of account

---

### 🚀 New Technical Picks (Trade Candidates)

---

**$NVDA** (Score: 8.3/10)
- Current price: $950.25
- Analyst target: $1000.00 median (range: $900.00–$1100.00)
- Implied upside: +5.2%
- RSI: 75.5 (overbought)
- MACD: bullish with strong divergence accelerating
- Volume: high
- MA trend: bullish
- 5d change: +5.2%
```

### JSON Output Structure
```json
{
  "timestamp": "2025-05-20T13:49:01.123456",
  "positions": [
    {
      "ticker": "AAPL",
      "timestamp": "2025-05-20T13:49:01.123456",
      "score": 7.5,
      "price_data": {
        "current": 150.25,
        "change_5d": 2.5,
        "volume": 1000000,
        "volume_ratio": 1.2
      },
      "technical_indicators": {
        "rsi": {
          "value": 45.5,
          "status": "neutral"
        },
        "macd": {
          "value": 0.5,
          "signal": 0.3,
          "diff": 0.2
        },
        "bollinger_bands": {
          "upper": 155.0,
          "lower": 145.0,
          "position": "within"
        },
        "moving_averages": {
          "ma20": 148.0,
          "ma50": 146.0,
          "trend": "bullish"
        }
      },
      "analyst_targets": {
        "current_price": 150.25,
        "median_target": 165.0,
        "low_target": 140.0,
        "high_target": 180.0,
        "implied_upside": 9.82
      },
      "position": {
        "quantity": 100,
        "cost_basis": 145.0,
        "market_value": 15025.0,
        "gain_pct": 3.62,
        "account_pct": 15.0
      }
    }
  ],
  "new_picks": [
    {
      "ticker": "NVDA",
      "timestamp": "2025-05-20T13:49:01.123456",
      "score": 8.3,
      "price_data": {
        "current": 950.25,
        "change_5d": 5.2,
        "volume": 2000000,
        "volume_ratio": 1.5
      },
      "technical_indicators": {
        "rsi": {
          "value": 75.5,
          "status": "overbought"
        },
        "macd": {
          "value": 2.5,
          "signal": 1.5,
          "diff": 1.0
        },
        "bollinger_bands": {
          "upper": 960.0,
          "lower": 940.0,
          "position": "within"
        },
        "moving_averages": {
          "ma20": 945.0,
          "ma50": 940.0,
          "trend": "bullish"
        }
      },
      "analyst_targets": {
        "current_price": 950.25,
        "median_target": 1000.0,
        "low_target": 900.0,
        "high_target": 1100.0,
        "implied_upside": 5.24
      }
    }
  ]
}
```

## Score Calculation

The technical score is normalized to a 0-10 scale based on the following factors:

1. RSI Conditions:
   - Oversold (< 30): +2 points
   - Overbought (> 70): +1 point

2. Bollinger Bands:
   - Price above upper band: +2 points
   - Price below lower band: +2 points

3. MACD Analysis:
   - Bullish/bearish crossover: +2 points
   - Strong divergence: +2 points
   - Moderate divergence: +1 point
   - Trend acceleration: +1 point

4. Volume Analysis:
   - Volume spike (> 2x average): +1 point

5. Analyst Targets:
   - High upside potential (≥ 20%): +2 points
   - Moderate upside potential (≥ 10%): +1 point

The raw score (maximum 12 points) is then normalized to a 0-10 scale.

## Requirements

- Python 3.8+
- pandas>=2.0.0
- numpy>=1.24.0
- yfinance>=0.2.36
- ta>=0.10.0
- typer>=0.9.0

## License

MIT License 