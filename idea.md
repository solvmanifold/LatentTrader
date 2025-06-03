# LatentTrader: Vision and Architecture

## Project Vision
LatentTrader aims to be a trading assistant that generates actionable, short-term trading playbooks designed to beat the market. The system leverages both classic technical analysis and machine learning (ML) models to create, test, and iterate on predictive scoring models. The ultimate goal is to support robust experimentation, backtesting, and rapid iteration to discover and deploy the most effective trading strategies.

## Architecture Overview
Prev_Volume
### 1. Data Layer (Features)
- **Per-Ticker Features:** All raw and engineered features (technical indicators, price/volume, etc.) are stored in Parquet files, one per ticker (e.g., `features/AAPL_features.parquet`).
- **Market-Wide Features:** All market-wide and macro features (e.g., VIX, SPY, GDELT sentiment, sector ETFs) are stored in a single Parquet table (`market_features/market_features.parquet`), with one row per date and columns for each feature (e.g., `vix`, `spy_close`, `spy_return`, `gdelt_sentiment`, ...).
- **Joining:** During feature engineering, per-ticker features are joined with market-wide features by date, so every row for every ticker includes the relevant market context.
- **Update Process:** Features are updated daily (or as needed) via a dedicated command (`init-features`).
- **No Model Scores:** These files contain only features, not model outputs.

### 2. Model Layer (Outputs)
- **Model Registry:** Each model (classic, ML, ensemble, etc.) is a Python class/function with a standard interface (e.g., `score(df)` or `predict(df)`).
- **Model Runner:** A command or function loads features, runs one or more models, and outputs scores/playbooks for each ticker/date.
- **Model Outputs:** Model scores and playbooks are stored in a separate folder (e.g., `model_outputs/{model_name}/{ticker}.parquet`). These files are keyed by ticker and date, and are append-only.
- **Experimentation:** Easy to add, swap, or update models and compare their outputs.

### 3. Reporting Layer (Daily Reports)
- **Daily Markdown Reports:** For each day and model, generate a Markdown report listing the top-N tickers by score, including OHLC, technical indicators, and analyst targets. Saved as `reports/{model_name}_{date}.md`.
- **Historical Tracking:** Each report is also saved as a row in a Parquet table (`reports/{model_name}.parquet`), including the report text and top tickers/scores for that day/model.
- **No Intermediate JSON:** Reports are generated directly from model outputs, not from intermediate JSON files. There is no generic report command—use `report-daily` only.
- **Consistent, Auditable, Historical:** This approach enables easy querying, auditing, and re-generation of reports for any day/model, and supports historical review.

### 4. Prompting Layer
- **Prompt Generation from Reports:** Prompts for LLMs or other downstream tasks are generated from the daily report tables/files, ensuring consistency and leveraging the curated report content.
- **Daily Prompts:** Prompts are saved in a similar structure (e.g., `prompts/{model_name}_{date}.txt` and/or `prompts/{model_name}.parquet`). There is no generic prompt command—use `prompt-daily` only.

## Market Features

Market-wide features are stored in separate parquet files based on their category and update frequency. This separation allows for independent updates and better organization.

### Directory Structure
```
market_features/
├── metadata/
│   └── sector_mapping.parquet    # Ticker -> Sector mapping
├── breadth/
│   └── daily_breadth.parquet     # Market breadth indicators
├── sectors/
│   └── sector_performance.parquet # Sector-level metrics
├── volatility/
│   └── market_volatility.parquet # Market volatility measures
└── sentiment/
    └── market_sentiment.parquet  # Market sentiment indicators
```

### Feature Categories

1. **Market Breadth** (daily updates)
   - Number of stocks above/below key moving averages (20/50/200 day)
   - Percentage of stocks in oversold/overbought RSI conditions
   - Number of stocks with MACD crossovers
   - Volume trends across the market

2. **Sector Performance** (daily updates)
   - Daily sector returns
   - Sector momentum indicators
   - Relative strength vs. S&P 500
   - Number of stocks in each sector above/below key levels

3. **Market Volatility** (daily/intraday updates)
   - VIX and its derivatives
   - Market-wide volatility measures
   - Correlation between stocks

4. **Market Sentiment** (daily/weekly updates)
   - Put/Call ratios
   - Short interest trends
   - Analyst sentiment aggregation

### Implementation Notes
- Each category is stored in its own parquet file for independent updates
- Features are computed using individual stock data where available
- External data sources (e.g., VIX, Put/Call ratios) are integrated as needed
- Sector mapping is used to aggregate and analyze sector-level metrics

## Next Steps
1. **Refactor Reporting/Prompting:**
    - Reporting and prompting are now implemented as daily Markdown/text files and Parquet tables.
    - All generation is from model outputs and daily reports, with historical tracking.
2. **Backtesting:**
    - Re-implement backtesting to use the new model outputs and reporting structure.
3. **Model Registry & Experimentation:**
    - Make it easy to add, discover, and compare models.
    - Add support for ensembles, parameter sweeps, and advanced experiment tracking.
4. **Robustness & Testing:**
    - Expand test coverage, especially for new reporting/prompting logic and edge cases.

---

*This architecture enables rapid iteration, robust experimentation, and a clear separation of concerns between data, modeling, reporting, and prompting. The next step is to build out prompt generation from the daily report tables, followed by backtesting and further experimentation support.*
