# Ticker Features

This document provides detailed descriptions of the features computed for individual tickers in the LatentTrader project.

## Directory Structure

All ticker features are stored in the `data/ticker_features/` directory:

```
data/ticker_features/
├── {ticker}.parquet              # Individual ticker features
└── all_tickers.parquet           # Combined dataset
```

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

## Data Structure

All ticker feature files follow these structural requirements:

1. **Date Indexing:**
   - All data must be indexed by date using a DatetimeIndex
   - No duplicate date columns are allowed
   - Dates are normalized (time set to midnight)
   - Missing trading days are filled with NaN values to preserve data integrity

2. **Column Naming:**
   - All column names are lowercase with underscores
   - No spaces or special characters allowed
   - Feature-specific prefixes are used where appropriate

3. **Handling Missing Trading Days:**
   - Missing trading days are preserved as NaN values in the raw data
   - When forward filling is needed for analysis:
     - Use `df.ffill()` for price data
     - Use `df.ffill(limit=1)` for returns data to prevent artificial smoothing
     - Use `df.ffill(limit=5)` for sentiment data to maintain recent context
   - Always document when forward filling is applied
   - Consider the impact of forward filling on your analysis:
     - Price data: Forward filling is generally acceptable
     - Returns data: Forward filling can create false patterns
     - Volume data: Forward filling is not recommended
     - Sentiment data: Forward filling should be limited to recent context

## Data Validation

All ticker feature files are subject to rigorous validation:

1. **File Validation:**
   - Consistent naming conventions (lowercase with underscores)
   - Proper file extensions (.parquet)
   - Correct directory structure
   - Proper date indexing (DatetimeIndex, no duplicate date columns)

2. **Column Validation:**
   - Required columns present
   - Column naming conventions
   - Data type consistency
   - Feature-specific validations
   - No duplicate date columns

3. **Data Quality Checks:**
   - Missing value detection
   - Outlier detection
   - Data range validation
   - Consistency across related features

4. **Feature-Specific Validation:**
   - Technical indicators
   - Price/volume metrics
   - Returns calculations
   - Analyst targets
   - Short interest data

For detailed information about the validation framework, see `validation.md`.

## Data Update Process

Ticker features are updated using the `update-data` command, which takes a `days` parameter to specify how many days of historical data to download. The system will:

1. Check for existing data and only download new data points
2. Calculate technical indicators
3. Compute price/volume metrics
4. Generate returns and volatility measures
5. Update analyst targets and short interest data
6. Run data validation checks:
   - File naming conventions
   - Column naming standards
   - Data type validation
   - Required column checks
   - Data quality metrics
   - Feature consistency validation

The update process is incremental, meaning it will only process new dates that aren't already in the feature files. This ensures efficient updates while maintaining historical data consistency.

Feel free to explore each feature for more detailed information and their significance in trading strategies. 