You're a Python engineer tasked with building a lightweight CLI tool called `weekly_trading_advisor.py`.

The goal is to help a user generate a prompt for ChatGPT Deep Research, based on technical indicators for S&P 500 stocks.

---

**Functionality:**

1. **Pull historical stock data** for all S&P 500 tickers.
   - Use `yfinance` to download OHLCV data.
   - Only download missing or new data:
     - If a local CSV file already exists for a ticker, load it, find the most recent date, and only append new data (`yfinance.download(start=last_date + 1)`).
     - Otherwise, download ~100 trading days of history.
   - Store data in a local folder like `data/`.
   - Use a hardcoded list of S&P 500 tickers, or load from `tickers.csv`.

2. **Include current user-held positions** from a separate input CSV (e.g. `positions.csv`):
   - Format: one stock ticker per row (or include optional quantity/cost basis).
   - These stocks should always be included in the final Deep Research output, even if their technical score is low.
   - Mark them clearly as "Current Position – Seeking Advice (Buy/Hold/Sell/Trailing Stop/etc.)"

3. **Compute technical indicators** for each stock:
   - Bollinger Bands (20-day window, 2 std dev)
   - RSI (14-day window)
   - MACD and signal line
   - 20-day and 50-day moving averages
   - Detect MACD crossovers and moving average crossovers
   - Flag volume spikes (e.g., 2× average volume)

4. **Score and rank each stock**:
   - Assign scores based on indicator combinations:
     - RSI < 30 (oversold): +2
     - RSI > 70 (overbought): +1
     - Price > upper BB: +2
     - Price < lower BB: +2
     - MACD crossover (bullish or bearish): +2
     - Volume spike: +1

5. **Select the top 5–10 new candidates** based on the highest score (excluding already-held positions).

6. **Output a markdown-formatted summary** of all selected stocks:
   - Always include all stocks from `positions.csv` under a separate "Current Holdings" section.
   - Include the top 5–10 newly flagged stocks under "New Technical Picks".
   - Each stock entry should include symbol, key indicator values, and a plain-English interpretation.

7. **Generate a ChatGPT-ready Deep Research prompt**, e.g.:

'''
You are an equity analyst. Here is the technical summary for several S&P 500 stocks.

The first section contains current positions. Please advise whether to hold, sell, or adjust (e.g. trailing stop).
The second section contains new picks flagged by our model this week. Please assess each for trade viability.
CURRENT POSITIONS:
$AAPL: RSI = 32, MACD neutral, volume surge. Recent pullback after earnings miss.

NEW TECHNICAL PICKS:
$NVDA: Price above upper BB, RSI = 74, MACD bullish 2d ago, high volume.
$DIS: RSI = 28, touching lower BB, MACD bearish crossover, but diverging volume.
...
'''

8. **Optional CLI args**:
   - `--top_n`: Number of new tickers to include (default = 5)
   - `--output`: Path to write the markdown file (default = stdout)
   - `--positions`: Path to `positions.csv` (optional but recommended)

---

**Requirements**:
- Python 3.8+
- Use `pandas`, `numpy`, `yfinance`, and optionally `ta` (technical analysis library)
- Use `argparse` or `typer` for CLI
- Output must be modular, readable, and suitable for pasting into ChatGPT manually

This tool should not query any LLM itself—it prepares structured prompts and summaries for the user to paste into ChatGPT with Deep Research enabled.
