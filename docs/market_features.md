# Market Features

This document provides detailed descriptions of the market-wide features computed in the LatentTrader project. These market features are derived from individual ticker features, which are documented in `ticker_features.md`.

## Directory Structure

All market features are stored in the `data/market_features/` directory:

```
data/market_features/
├── metadata/
│   └── sector_mapping.json       # Ticker -> Sector mapping
├── daily_breadth.parquet         # Market breadth indicators
├── market_volatility.parquet     # Market volatility measures
├── market_sentiment.parquet      # Market sentiment indicators
├── gdelt_raw.parquet             # Raw GDELT sentiment data
├── sp500.parquet                 # S&P 500 index data
└── sectors/
    └── {sector_name}.parquet     # Sector-level metrics
```

## Market Breadth

Market breadth indicators are calculated daily and include:

- **Advance/Decline Line:** Net difference between advancing and declining stocks (`daily_breadth_adv_dec_line`)
  - Formula: ADL = Σ(Advances - Declines) for each day
  - Cumulative sum of daily net advances/declines
  - Mathematical representation: ADL(t) = ADL(t-1) + (Advances(t) - Declines(t))

- **New Highs/Lows:** Number of stocks making new 20-day highs (`daily_breadth_new_highs`) and lows (`daily_breadth_new_lows`)
  - Formula: New Highs = Count of stocks at 20-day high
    - Mathematical representation: New Highs(t) = Σ[Price(t) > max(Price(t-20:t-1))]
  - Formula: New Lows = Count of stocks at 20-day low
    - Mathematical representation: New Lows(t) = Σ[Price(t) < min(Price(t-20:t-1))]

- **Moving Average Indicators:** Percentage of stocks above 20-day (`daily_breadth_above_ma20`) and 50-day (`daily_breadth_above_ma50`) moving averages
  - Formula: % Above MA = (Count of stocks above MA / Total stocks) × 100
  - Mathematical representation: 
    - MA20(t) = SMA(20) of price
    - % Above MA20(t) = (Σ[Price(t) > MA20(t)] / N) × 100
    - MA50(t) = SMA(50) of price
    - % Above MA50(t) = (Σ[Price(t) > MA50(t)] / N) × 100

- **RSI Indicators:** 
  - Percentage of stocks with bullish RSI (`daily_breadth_rsi_bullish`)
    - Formula: % Bullish = (Count of stocks with RSI > 50 / Total stocks) × 100
    - Mathematical representation: RSI = 100 - (100 / (1 + RS))
      where RS = Average Gain / Average Loss over 14 days
  - Percentage of stocks in oversold condition (`daily_breadth_rsi_oversold`)
    - Formula: % Oversold = (Count of stocks with RSI < 30 / Total stocks) × 100
  - Percentage of stocks in overbought condition (`daily_breadth_rsi_overbought`)
    - Formula: % Overbought = (Count of stocks with RSI > 70 / Total stocks) × 100

- **MACD Indicators:** Percentage of stocks with bullish MACD signals (`daily_breadth_macd_bullish`)
  - Formula: % Bullish = (Count of stocks with positive MACD histogram / Total stocks) × 100
  - Mathematical representation:
    - MACD Line = EMA(12) - EMA(26)
    - Signal Line = EMA(9) of MACD Line
    - Histogram = MACD Line - Signal Line

The data is stored in `data/market_features/daily_breadth.parquet`

## Market Volatility

Market volatility features are calculated daily and include:

- **VIX:** The CBOE Volatility Index (`market_volatility_vix`)
  - Raw VIX value from CBOE
  - Measures market's expectation of 30-day forward-looking volatility

- **Market-wide Volatility Measures:**
  - Daily Volatility (`market_volatility_daily_volatility`)
    - Formula: Standard deviation of daily returns across all stocks
    - Mathematical representation: σ = √(Σ(r - μ)² / (n-1))
      where r = daily returns, μ = mean return, n = number of stocks
  - Weekly Volatility (`market_volatility_weekly_volatility`)
    - Formula: Standard deviation of weekly returns across all stocks
    - Uses 5-day rolling window
  - Monthly Volatility (`market_volatility_monthly_volatility`)
    - Formula: Standard deviation of monthly returns across all stocks
    - Uses 21-day rolling window

- **Average Correlation:** (`market_volatility_avg_correlation`)
  - Formula: Average pairwise correlation between stock returns
  - Mathematical representation: ρ = Σ(ρᵢⱼ) / (n(n-1)/2)
    where ρᵢⱼ = correlation between stocks i and j
  - Uses 20-day rolling window for correlation calculation

- **Ticker-specific Volatility:** (`market_volatility_ticker`)
  - Individual stock volatility relative to market
  - Formula: σᵢ / σₘ
    where σᵢ = stock volatility, σₘ = market volatility
  - Uses 20-day rolling window

The data is stored in `data/market_features/market_volatility.parquet`

## Market Sentiment

Market sentiment features are calculated daily and include:

- **Moving Averages:**
  - 5-day Moving Average (`market_sentiment_ma5`)
    - Formula: MA5(t) = (Price(t) + Price(t-1) + Price(t-2) + Price(t-3) + Price(t-4)) / 5
    - Mathematical representation: MA5(t) = Σ(P(t-i)) / 5 for i = 0 to 4
  - 20-day Moving Average (`market_sentiment_ma20`)
    - Formula: MA20(t) = Σ(Price(t-i)) / 20 for i = 0 to 19
    - Mathematical representation: MA20(t) = Σ(P(t-i)) / 20 for i = 0 to 19

- **Momentum:** (`market_sentiment_momentum`)
  - Price change over 5 days
  - Formula: (Price(t) - Price(t-5)) / Price(t-5)
  - Mathematical representation: M(t) = (P(t) - P(t-5)) / P(t-5)

- **Volatility:** (`market_sentiment_volatility`)
  - 20-day rolling standard deviation of returns
  - Formula: σ = √(Σ(r - μ)² / (n-1))
  - Mathematical representation: σ(t) = √(Σ(r(t-i) - μ)² / 19) for i = 0 to 19
    where r = daily returns, μ = mean return over 20 days

- **Z-Score:** (`market_sentiment_zscore`)
  - Standardized measure of current price relative to its moving average
  - Formula: (Price(t) - MA20(t)) / σ(t)
  - Mathematical representation: Z(t) = (P(t) - MA20(t)) / σ(t)
    where σ(t) is the 20-day standard deviation

The data is stored in `data/market_features/market_sentiment.parquet`

## Sector Performance

Sector performance features are calculated daily and stored in individual files under `data/market_features/sectors/`:

- **Price:** (`sector_performance_price`)
  - Sector index price level
  - Weighted average of constituent stock prices

- **Volatility:** (`sector_performance_volatility`)
  - 20-day rolling volatility of sector returns
  - Formula: Standard deviation of daily returns

- **Volume:** (`sector_performance_volume`)
  - Total trading volume across sector
  - Normalized by market cap

- **Returns:**
  - 1-day returns (`sector_performance_returns_1d`)
  - 5-day returns (`sector_performance_returns_5d`)
  - 20-day returns (`sector_performance_returns_20d`)

- **Momentum:**
  - 5-day momentum (`sector_performance_momentum_5d`)
    - Price change over 5 days
  - 20-day momentum (`sector_performance_momentum_20d`)
    - Price change over 20 days

- **Relative Strength:**
  - Raw relative strength (`sector_performance_relative_strength`)
    - Sector return vs S&P 500 return
  - Relative strength ratio (`sector_performance_relative_strength_ratio`)
    - Sector return / S&P 500 return

Each sector's data is stored in its own file: `data/market_features/sectors/{sector_name}.parquet`

## S&P 500 Index Data

The S&P 500 index data is stored in `data/market_features/sp500.parquet` and includes:

- **Price:** (`sp500_price`) Daily closing price of the S&P 500 index
- **Returns:** (`sp500_returns_20d`) 20-day returns of the index
  - Formula: (Price(t) - Price(t-20)) / Price(t-20)
  - Mathematical representation: r₂₀(t) = (P(t) - P(t-20)) / P(t-20)
  - Where P(t) is the closing price at time t

This data serves as a benchmark for relative performance calculations and market regime analysis.

## Date Handling

All market feature files follow a consistent date handling approach:

1. **Date Indexing:**
   - All data must be indexed by date using a DatetimeIndex
   - Dates are normalized (time set to midnight)
   - Missing trading days are filled with NaN values to preserve data integrity

2. **Date Format:**
   - All dates are stored in YYYY-MM-DD format
   - No timezone information is included

For detailed information about data validation and processing, see `validation.md` and `data_processing.md`. 