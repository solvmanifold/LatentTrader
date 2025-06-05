# LatentTrader: Vision and Architecture

## Project Vision
LatentTrader aims to be a trading assistant that generates actionable, short-term trading playbooks designed to beat the market. The system leverages both classic technical analysis and machine learning (ML) models to create, test, and iterate on predictive scoring models. The ultimate goal is to support robust experimentation, backtesting, and rapid iteration to discover and deploy the most effective trading strategies.

## Architecture Overview

### 1. Data Layer (Features)
- **Per-Ticker Features:** All raw and engineered features (technical indicators, price/volume, etc.) are stored in Parquet files, one per ticker (e.g., `ticker_features/AAPL_features.parquet`).
- **Market-Wide Features:** Market features are stored in separate Parquet files by category (volatility, breadth, sentiment, etc.) for independent updates.
- **Sector Features:** Sector-level metrics are stored in separate files under `market_features/sectors/`.
- **Update Process:** Features are updated via the `update-data` command with flags for different feature types.
- **Feature Mappings:** Categorical features (e.g., tickers) are mapped to integers and stored in `feature_mappings.json`.

### 2. Model Layer (Outputs)
- **Model Registry:** Each model (classic, ML, ensemble, etc.) is a Python class/function with a standard interface.
- **Model Runner:** The `run-model` command loads features, runs models, and outputs scores/playbooks.
- **Model Outputs:** Model scores are stored in `model_outputs/{model_name}/{ticker}.parquet`.
- **ML Datasets:** Binary classification datasets are generated with time-series splits via `generate-classification-dataset`.

### 3. Reporting Layer (Daily Reports)
- **Daily Markdown Reports:** Generated via `report-daily` command, listing top-N tickers by score.
- **Historical Tracking:** Reports are saved as both Markdown files and Parquet tables.
- **Consistent, Auditable, Historical:** Easy querying and re-generation of reports for any day/model.

### 4. Prompting Layer
- **Prompt Generation:** The `prompt-daily` command generates LLM prompts from daily reports.
- **Historical Tracking:** Prompts are saved as both text files and Parquet tables.

## Market Features

### Directory Structure
```
data/market_features/
├── metadata/
│   └── sector_mapping.parquet    # Ticker -> Sector mapping
├── daily_breadth.parquet         # Market breadth indicators
├── market_volatility.parquet     # Market volatility measures
├── market_sentiment.parquet      # Market sentiment indicators
├── gdelt_raw.parquet             # Raw GDELT sentiment data
└── sectors/
    └── {sector_name}.parquet     # Sector-level metrics
```

### Feature Categories

1. **Market Breadth** (daily updates)
   - Number of stocks above/below key moving averages
   - Percentage of stocks in oversold/overbought RSI conditions
   - Number of stocks with MACD crossovers
   - Volume trends across the market

2. **Sector Performance** (daily updates)
   - Daily sector returns
   - Sector momentum indicators
   - Relative strength vs. S&P 500
   - Number of stocks in each sector above/below key levels

3. **Market Volatility** (daily updates)
   - VIX and its derivatives
   - Market-wide volatility measures
   - Correlation between stocks

4. **Market Sentiment** (daily/weekly updates)
   - Put/Call ratios
   - Short interest trends
   - Analyst sentiment aggregation
   - GDELT news sentiment

## Progress and Next Steps

### ✅ Completed
1. **Data Layer**
   - Market data collection and storage
   - Market breadth indicators
   - Sector performance analysis
   - Sentiment analysis (GDELT)
   - Incremental updates for market features
   - Feature mapping and preprocessing
   - ML dataset generation with time-series splits

2. **Model Layer (Basic Infrastructure)**
   - Model interface definition
   - Model registry structure
   - Model output storage
   - Basic model runner implementation

3. **Reporting Layer**
   - Daily Markdown reports
   - Historical tracking in Parquet
   - Report generation from model outputs

4. **CLI Implementation**
   - `update-data` for feature updates
   - `run-model` for model execution
   - `report-daily` for report generation
   - `prompt-daily` for prompt generation
   - `generate-classification-dataset` for ML datasets

### 🚧 In Progress/Next Steps
1. **Model Layer (Advanced Features)**
   - Model experimentation framework
   - Ensemble model support
   - Parameter sweep functionality
   - Model performance tracking
   - Model versioning system

2. **Backtesting**
   - Re-implement using new model outputs
   - Integrate with reporting structure
   - Add performance metrics and visualization

3. **Documentation**
   - Create thorough documentation for features
   - Add usage examples and guides
   - Document model interfaces and requirements

4. **Testing and Robustness**
   - Expand test coverage
   - Add edge case handling
   - Improve error messages and logging

# Next Steps for Logistic Model Project

1. **Expand Dataset**
   - Generate a larger dataset with all tickers and at least 4 years of data.
   - Ensure splits are representative and balanced across time and tickers.

2. **Re-run Cross-Validation**
   - Use the improved script to train and evaluate the model on the expanded dataset.
   - Aggregate and analyze results for stability and generalization.

3. **Model/Feature Engineering (Optional)**
   - Experiment with additional features or feature selection.
   - Try different regularization strengths or model hyperparameters.
   - Consider ensembling or stacking with other models if performance is still unstable.

4. **CLI Integration**
   - Once satisfied with model robustness and performance, integrate the logistic model into the CLI.
   - Add commands for training, evaluation, and prediction from the command line.

5. **Documentation & Visualization**
   - Document the pipeline, results, and key findings.
   - Save and visualize feature importances and confusion matrices for the expanded dataset.

6. **Further Validation**
   - Test the model on out-of-sample data or new tickers.
   - Monitor for overfitting or data leakage.

---

*These steps ensure a robust, scalable, and production-ready trading model pipeline.*
