# Ticker Features

This document provides detailed descriptions of the features computed for individual tickers in the LatentTrader project.

## Technical Indicators

- **Moving Averages:**
  - `sma_20`: 20-day Simple Moving Average
    - Formula: SMA(n) = (P₁ + P₂ + ... + Pₙ) / n
  - `sma_50`: 50-day Simple Moving Average
  - `sma_200`: 200-day Simple Moving Average
  - `ema_20`: 20-day Exponential Moving Average
    - Formula: EMA(today) = α × Price(today) + (1-α) × EMA(yesterday)
    - where α = 2/(n+1), n = 20
  - `ema_50`: 50-day Exponential Moving Average
  - `ema_200`: 200-day Exponential Moving Average

- **Relative Strength Index (RSI):**
  - `rsi_14`: 14-day RSI value
    - Formula: RSI = 100 - (100 / (1 + RS))
    - where RS = Average Gain / Average Loss over 14 days
  - `rsi_signal`: Binary signal indicating if RSI is bullish (>50) or bearish (<50)

- **MACD (Moving Average Convergence Divergence):**
  - `macd`: MACD line (12-day EMA - 26-day EMA)
    - Formula: MACD = EMA(12) - EMA(26)
  - `macd_signal`: Signal line (9-day EMA of MACD)
    - Formula: Signal = EMA(9) of MACD
  - `macd_hist`: MACD histogram (MACD - Signal)
  - `macd_signal_binary`: Binary signal indicating if MACD is bullish (histogram > 0) or bearish (histogram < 0)

## Price/Volume Metrics

- **OHLC (Open, High, Low, Close):**
  - `open`: Opening price
  - `high`: Highest price
  - `low`: Lowest price
  - `close`: Closing price
  - `adj_close`: Adjusted closing price (adjusted for splits and dividends)

- **Volume:**
  - `volume`: Trading volume
  - `volume_ma20`: 20-day moving average of volume
  - `volume_ratio`: Current volume / 20-day average volume
    - Formula: Volume Ratio = Volume(today) / SMA(20) of Volume

- **Returns:**
  - `returns_1d`: Daily returns
    - Formula: r = (P₁ - P₀) / P₀
  - `returns_5d`: 5-day returns
  - `returns_20d`: 20-day returns
  - `volatility_20d`: 20-day rolling volatility (annualized)
    - Formula: σ = √(252 × Variance of daily returns)
    - where 252 is the number of trading days in a year

## Additional Features

- **Analyst Targets:**
  - `price_target_mean`: Mean analyst price target
    - Formula: Mean = Σ(targets) / n
  - `price_target_median`: Median analyst price target
  - `price_target_std`: Standard deviation of price targets
    - Formula: σ = √(Σ(x - μ)² / n)
  - `recommendation_mean`: Mean recommendation score (1=Strong Buy, 5=Strong Sell)

- **Short Interest:**
  - `short_interest`: Number of shares sold short
  - `short_interest_ratio`: Short interest / Average daily volume
    - Formula: Ratio = Short Interest / (20-day Average Volume)
  - `short_interest_change`: Change in short interest from previous period
    - Formula: Change = (Current - Previous) / Previous

The data is stored in `data/ticker_features/{ticker}.parquet` for individual tickers and `data/ticker_features/all_tickers.parquet` for the combined dataset.

Feel free to explore each feature for more detailed information and their significance in trading strategies. 