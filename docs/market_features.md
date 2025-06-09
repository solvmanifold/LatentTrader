# Market Features

This document provides detailed descriptions of the market-wide features computed in the LatentTrader project. These market features are derived from individual ticker features, which are documented in `ticker_features.md`.

## Directory Structure

All market features are stored in the `data/market_features/` directory:

```
data/market_features/
├── sector_mapping.json           # Ticker -> Sector mapping
├── daily_breadth.parquet         # Market breadth indicators
├── market_volatility.parquet     # Market volatility measures
├── market_sentiment.parquet      # Market sentiment indicators
├── gdelt_raw.parquet             # Raw GDELT sentiment data
├── sp500.parquet                 # S&P 500 index data
└── sectors/                      # Sector-level metrics
    ├── basic_materials.parquet
    ├── communication_services.parquet
    ├── consumer_cyclical.parquet
    ├── consumer_defensive.parquet
    ├── energy.parquet
    ├── financial_services.parquet
    ├── healthcare.parquet
    ├── industrials.parquet
    ├── real_estate.parquet
    ├── technology.parquet
    └── utilities.parquet
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

## S&P 500 Data

The S&P 500 data provides key market benchmark metrics:

- **Price Data:**
  - `sp500_price`: Daily closing price of the S&P 500 index
  - `sp500_returns_20d`: 20-day returns of the S&P 500 index
    - Formula: Returns = (Price(today) - Price(20 days ago)) / Price(20 days ago)

The data is stored in `data/market_features/sp500.parquet`

## Sector Performance

Each sector's performance is tracked through the following metrics (columns prefixed with sector name, e.g. `technology_price` for Technology sector):

- **Price Metrics:**
  - `{sector}_price`: Mean price of stocks in the sector
    - Formula: Mean Price = Σ(Stock Prices) / Number of Stocks
  - `{sector}_volatility`: Standard deviation of prices within the sector
    - Formula: σ = √(Σ(x - μ)² / n)
  - `{sector}_volume`: Total trading volume of stocks in the sector
    - Formula: Total Volume = Σ(Individual Stock Volumes)

- **Returns:**
  - `{sector}_returns_1d`: Daily returns
    - Formula: r = (P₁ - P₀) / P₀
  - `{sector}_returns_5d`: 5-day returns
  - `{sector}_returns_20d`: 20-day returns

- **Momentum Indicators:**
  - `{sector}_momentum_5d`: 5-day rolling average of returns
    - Formula: Momentum = SMA(5) of daily returns
  - `{sector}_momentum_20d`: 20-day rolling average of returns
    - Formula: Momentum = SMA(20) of daily returns

- **Relative Strength:**
  - `{sector}_relative_strength`: Ratio of cumulative returns vs S&P 500
    - Formula: RS = (1 + r_sector).cumprod() / (1 + r_sp500).cumprod()
    - Where r_sector and r_sp500 are daily returns
  - `{sector}_relative_strength_ratio`: Difference in 5-day returns vs S&P 500
    - Formula: RSR = r_sector_5d - r_sp500_5d
    - Where r_sector_5d and r_sp500_5d are 5-day returns

The data is stored in the following format:
- Individual sector files: `data/market_features/sectors/{sector_name}.parquet`

## Market Volatility

Market volatility features include:

- **VIX Indicator:**
  - `market_volatility_vix`: Daily VIX closing price
    - Formula: Raw VIX index value from CBOE
    - Mathematical representation: VIX = σ × √(252) × 100
      where σ is the implied volatility of S&P 500 options

- **Short-term Volatility Measures:**
  - `market_volatility_daily_volatility`: 2-day rolling standard deviation of returns
    - Formula: σ = √(Σ(x - μ)² / n) over 2-day window
    - Mathematical representation: σ(t) = √(Σ(r(t-i) - μ)² / 2) for i = 0,1
      where r is the daily return and μ is the 2-day mean return
  - `market_volatility_weekly_volatility`: 5-day rolling standard deviation of returns
    - Formula: σ = √(Σ(x - μ)² / n) over 5-day window
    - Mathematical representation: σ(t) = √(Σ(r(t-i) - μ)² / 5) for i = 0,1,2,3,4
  - `market_volatility_monthly_volatility`: 20-day rolling standard deviation of returns
    - Formula: σ = √(Σ(x - μ)² / n) over 20-day window
    - Mathematical representation: σ(t) = √(Σ(r(t-i) - μ)² / 20) for i = 0 to 19

- **Correlation Measures:**
  - `market_volatility_avg_correlation`: 5-day rolling correlation between stocks
    - Formula: Average pairwise correlation over 5-day window
    - Mathematical representation: 
      ρ(t) = (1/n) × Σ[ρ(i,j,t)] for all pairs (i,j)
      where ρ(i,j,t) = Cov(r_i, r_j) / (σ_i × σ_j) over 5-day window
      and n is the number of unique pairs

Note: While the original design included additional VIX-based indicators and cross-sectional measures, we currently use these volatility measures as they:
1. Provide more granular timeframes (2-day, 5-day, 20-day)
2. Include correlation metrics which capture market relationships
3. Include raw VIX as a market fear gauge
4. Are more direct measures of actual market volatility

Future enhancements may include:
- Additional VIX indicators (VIX MA20, VIX std20)
- Market-wide volatility (20-day S&P 500 volatility)
- Volatility of volatility measures
- Cross-sectional volatility measures

The data is stored in `data/market_features/market_volatility.parquet`

## Market Sentiment

Market sentiment is derived from GDELT news data and includes the following metrics:

- **What is GDELT?**
  - The Global Database of Events, Language, and Tone (GDELT) is an open-source project that monitors the world's news media in real time, extracting information about events, people, organizations, locations, counts, themes, sources, emotions, counts, quotes, images, and events. In this project, GDELT is used to quantify daily news sentiment related to the market.

- **Moving Averages:**
  - `market_sentiment_ma5`: 5-day moving average of sentiment
    - Formula: Sentiment MA5 = SMA(5) of daily sentiment
  - `market_sentiment_ma20`: 20-day moving average of sentiment
    - Formula: Sentiment MA20 = SMA(20) of daily sentiment

- **Momentum and Volatility:**
  - `market_sentiment_momentum`: 5-day change in sentiment
    - Formula: Momentum = Sentiment(today) - Sentiment(5 days ago)
  - `market_sentiment_volatility`: 20-day standard deviation of sentiment
    - Formula: σ = √(Σ(x - μ)² / n)
  - `market_sentiment_zscore`: Standardized sentiment score relative to 20-day mean
    - Formula: z = (x - μ) / σ

The data is stored in two formats:
1. Raw GDELT data: `data/market_features/gdelt_raw.parquet`
2. Processed sentiment features: `data/market_features/market_sentiment.parquet`

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