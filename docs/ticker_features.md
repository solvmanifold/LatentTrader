# Ticker Features

This document provides detailed descriptions of the features computed for individual tickers in the LatentTrader project. These ticker-level features are used as building blocks for market-wide features, which are documented in `market_features.md`.

## Directory Structure

All ticker features are stored in the `data/ticker_features/` directory:

```
data/ticker_features/
├── {ticker}.parquet              # Individual ticker features
└── all_tickers.parquet           # Combined dataset
```

## Technical Indicators

All technical indicators are calculated using the `ta` (Technical Analysis) library, which provides efficient and well-tested implementations of common technical analysis functions.

- **Moving Averages:**
  - `sma_20`: 20-day Simple Moving Average
    - Formula: SMA(n) = (P₁ + P₂ + ... + Pₙ) / n
    - where Pₙ is the price at day n
  - `sma_50`: 50-day Simple Moving Average
  - `sma_100`: 100-day Simple Moving Average
  - `sma_200`: 200-day Simple Moving Average
  - `ema_100`: 100-day Exponential Moving Average
    - Formula: EMA(today) = α × Price(today) + (1-α) × EMA(yesterday)
    - where α = 2/(n+1), n = 100
    - Initial EMA = SMA(n)
  - `ema_200`: 200-day Exponential Moving Average
    - Same formula as EMA(100) with n = 200

- **Relative Strength Index (RSI):**
  - `rsi`: 14-day RSI value
    - Formula: RSI = 100 - (100 / (1 + RS))
    - where RS = Average Gain / Average Loss over 14 days
    - Average Gain = Sum of gains over 14 days / 14
    - Average Loss = Sum of losses over 14 days / 14
    - Gains and losses are calculated as positive price changes

- **MACD (Moving Average Convergence Divergence):**
  - `macd`: MACD line (12-day EMA - 26-day EMA)
    - Formula: MACD = EMA(12) - EMA(26)
    - where EMA(12) uses α = 2/(12+1) = 0.1538
    - and EMA(26) uses α = 2/(26+1) = 0.0741
  - `macd_signal`: Signal line (9-day EMA of MACD)
    - Formula: Signal = EMA(9) of MACD
    - where α = 2/(9+1) = 0.2
  - `macd_hist`: MACD histogram (MACD - Signal)
    - Formula: Histogram = MACD - Signal
    - Positive values indicate bullish momentum
    - Negative values indicate bearish momentum

- **Bollinger Bands:**
  - `bb_upper`: Upper Bollinger Band
    - Formula: BB_upper = SMA(20) + (2 × σ)
    - where σ is the 20-day standard deviation of prices
  - `bb_middle`: Middle Bollinger Band
    - Formula: BB_middle = SMA(20)
  - `bb_lower`: Lower Bollinger Band
    - Formula: BB_lower = SMA(20) - (2 × σ)
  - `bb_pband`: Percentage B (position within the bands)
    - Formula: %B = (Price - BB_lower) / (BB_upper - BB_lower)
    - Values > 1 indicate price above upper band
    - Values < 0 indicate price below lower band
    - Values between 0 and 1 indicate position within bands

## Price/Volume Metrics

- **OHLC (Open, High, Low, Close):**
  - `open`: Opening price
  - `high`: Highest price
  - `low`: Lowest price
  - `close`: Closing price
  - `adj_close`: Adjusted closing price (adjusted for splits and dividends)
    - Formula: Adj_Close = Close × Split Factor × Dividend Adjustment Factor

- **Volume:**
  - `volume`: Trading volume
  - `volume_prev`: Previous day's trading volume
    - Formula: Volume_prev = Volume(t-1)
    - where t is the current day

- **Corporate Actions:**
  - `dividends`: Dividend payments
    - Raw dividend amounts paid per share
    - Used internally for calculating adjusted close prices
  - `stock_splits`: Stock split events
    - Format: "N:M" where N is the new number of shares and M is the old number
    - Used internally for calculating adjusted close prices

## Analyst Targets

- **Analyst Information:**
  - `analyst_targets`: JSON string containing analyst target information
    - Includes current price and median target price
    - Format: `{"current_price": float, "median_target": float}`
    - Upside potential = (median_target - current_price) / current_price × 100%

## Date Handling

All ticker feature files follow these structural requirements:

1. **Date Indexing:**
   - All data must be indexed by date using a DatetimeIndex
   - Dates are normalized (time set to midnight)
   - Missing trading days are filled with NaN values to preserve data integrity

2. **Date Format:**
   - All dates are stored in YYYY-MM-DD format
   - No timezone information is included

For detailed information about data validation and processing, see `validation.md` and `data_processing.md`. 