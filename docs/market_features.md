# Market Features

This document provides detailed descriptions of the market-wide features computed in the LatentTrader project.

## Market Breadth

Market breadth indicators are calculated daily and include:

- **Advance/Decline Line:** Net difference between advancing and declining stocks (`adv_dec_line`)
  - Formula: ADL = Σ(Advances - Declines) for each day
  - Cumulative sum of daily net advances/declines

- **New Highs/Lows:** Number of stocks making new 20-day highs (`new_highs`) and lows (`new_lows`)
  - Formula: New Highs = Count of stocks at 20-day high
  - Formula: New Lows = Count of stocks at 20-day low

- **Moving Average Indicators:** Percentage of stocks above 20-day (`above_ma20`) and 50-day (`above_ma50`) moving averages
  - Formula: % Above MA = (Count of stocks above MA / Total stocks) × 100

- **RSI Indicators:** 
  - Percentage of stocks with bullish RSI (`rsi_bullish`)
    - Formula: % Bullish = (Count of stocks with RSI > 50 / Total stocks) × 100
  - Percentage of stocks in oversold condition (`rsi_oversold`)
    - Formula: % Oversold = (Count of stocks with RSI < 30 / Total stocks) × 100
  - Percentage of stocks in overbought condition (`rsi_overbought`)
    - Formula: % Overbought = (Count of stocks with RSI > 70 / Total stocks) × 100

- **MACD Indicators:** Percentage of stocks with bullish MACD signals (`macd_bullish`)
  - Formula: % Bullish = (Count of stocks with positive MACD histogram / Total stocks) × 100

The data is stored in `data/market_features/daily_breadth.parquet`

## Sector Performance

Each sector's performance is tracked through the following metrics:

- **Price Metrics:**
  - `price`: Mean price of stocks in the sector
    - Formula: Mean Price = Σ(Stock Prices) / Number of Stocks
  - `volatility`: Standard deviation of prices within the sector
    - Formula: σ = √(Σ(x - μ)² / n)
  - `volume`: Total trading volume of stocks in the sector
    - Formula: Total Volume = Σ(Individual Stock Volumes)

- **Returns:**
  - `returns_1d`: Daily returns
    - Formula: r = (P₁ - P₀) / P₀
  - `returns_5d`: 5-day returns
  - `returns_20d`: 20-day returns

- **Momentum Indicators:**
  - `momentum_5d`: 5-day rolling average of returns
    - Formula: Momentum = SMA(5) of daily returns
  - `momentum_20d`: 20-day rolling average of returns
    - Formula: Momentum = SMA(20) of daily returns

The data is stored in two formats:
1. Individual sector files: `data/market_features/sectors/{sector_name}.parquet`
2. Combined wide-format table: `data/market_features/all_sectors.parquet`

## Market Volatility

Market volatility features include:

- **VIX Indicators:**
  - `vix`: Daily VIX closing price
  - `vix_ma20`: 20-day moving average of VIX
    - Formula: VIX MA20 = SMA(20) of VIX
  - `vix_std20`: 20-day standard deviation of VIX
    - Formula: σ = √(Σ(x - μ)² / n)

- **Market-Wide Volatility:**
  - `market_volatility`: 20-day rolling volatility of S&P 500 returns (annualized)
    - Formula: σ = √(252 × Variance of daily returns)
  - `vol_of_vol`: Volatility of the market volatility (20-day standard deviation)
    - Formula: Vol of Vol = σ of 20-day rolling volatility

- **Cross-Sectional Measures:**
  - `cross_sectional_vol`: Daily dispersion of returns across all stocks (annualized)
    - Formula: σ = √(252 × Variance of cross-sectional returns)
  - `avg_correlation`: Average correlation between stocks (20-day rolling window)
    - Formula: ρ = Average of all pairwise correlations

The data is stored in `data/market_features/market_volatility.parquet`

## Market Sentiment

Market sentiment is derived from GDELT news data and includes the following metrics:

- **Moving Averages:**
  - `sentiment_ma5`: 5-day moving average of sentiment
    - Formula: Sentiment MA5 = SMA(5) of daily sentiment
  - `sentiment_ma20`: 20-day moving average of sentiment
    - Formula: Sentiment MA20 = SMA(20) of daily sentiment

- **Momentum and Volatility:**
  - `sentiment_momentum`: 5-day change in sentiment
    - Formula: Momentum = Sentiment(today) - Sentiment(5 days ago)
  - `sentiment_volatility`: 20-day standard deviation of sentiment
    - Formula: σ = √(Σ(x - μ)² / n)
  - `sentiment_zscore`: Standardized sentiment score relative to 20-day mean
    - Formula: z = (x - μ) / σ

The data is stored in two formats:
1. Raw GDELT data: `data/market_features/gdelt_raw.parquet`
2. Processed sentiment features: `data/market_features/market_sentiment.parquet`

Note: Put/Call ratios, short interest trends, and analyst sentiment aggregation are planned for future implementation.

Feel free to explore each feature for more detailed information and their significance in market analysis. 