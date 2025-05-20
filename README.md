# Weekly Trading Advisor

A Python CLI tool that generates trading advice based on technical indicators and analyst targets for S&P 500 stocks.

## Features

- Pulls historical stock data using yfinance
- Calculates technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Incorporates analyst price targets
- Generates markdown-formatted reports
- Supports current position analysis
- Caches Ticker objects for better performance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/weekly-trading-advisor.git
cd weekly-trading-advisor
```

2. Create and activate a virtual environment:
```bash
python -m venv trading-advisor
source trading-advisor/bin/activate  # On Windows: trading-advisor\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python weekly_trading_advisor.py --tickers tickers.txt
```

With positions file:
```bash
python weekly_trading_advisor.py --positions positions.csv --tickers tickers.txt
```

Options:
- `--top_n`: Number of new tickers to include (default: 5)
- `--output`: Path to write the markdown file
- `--positions`: Path to brokerage positions CSV
- `--tickers`: Path to tickers.txt file or 'all' for all S&P 500 tickers
- `--history_days`: Days of historical data to fetch (default: 100)
- `--include_charts`: Include plots in the summary output
- `--positions-only`: Only analyze current positions
- `--no-positions`: Exclude current positions from analysis

## Example Output

```markdown
You are an equity analyst. Here is the technical summary for several S&P 500 stocks.

---

### 📊 Current Positions (Hold/Sell Guidance)

---

**$AAPL**
- Current price: $176.86
- Analyst target: $195.00 median (range: $160.00–$230.00)
- Implied upside: +10.3%
- RSI: 45.2 (neutral)
- MACD: bullish with moderate divergence
- Volume: normal
- MA trend: bullish
- 5d change: +2.1%

---

### 🚀 New Technical Picks (Trade Candidates)

---

**$NVDA**
- Current price: $924.79
- Analyst target: $950.00 median (range: $800.00–$1100.00)
- Implied upside: +2.7%
- RSI: 68.5 (neutral)
- MACD: bullish with strong divergence accelerating
- Volume: high
- MA trend: bullish
- 5d change: +5.2%
```

## Requirements

- Python 3.8+
- pandas
- numpy
- yfinance
- ta (technical analysis library)
- typer

## License

MIT License 