# LatentTrader: Vision and Architecture

## Project Vision
LatentTrader aims to be a machine learning-powered trading assistant that generates actionable, short-term trading playbooks designed to beat the market. The system leverages both traditional technical analysis features and modern ML models to create, test, and iterate on predictive models. The ultimate goal is to support robust experimentation, backtesting, and rapid iteration to discover and deploy the most effective trading strategies.

## Architecture Overview

### 1. Data Layer (Features)
- **Per-Ticker Features:** All raw and engineered features (technical indicators, price/volume, etc.) are stored in Parquet files, one per ticker (e.g., `ticker_features/AAPL_features.parquet`).
- **Market-Wide Features:** Market features are stored in separate Parquet files by category (volatility, breadth, sentiment, etc.) for independent updates.
- **Sector Features:** Sector-level metrics are stored in separate files under `market_features/sectors/`.
- **Update Process:** Features are updated via the `update-data` command with flags for different feature types.
- **Feature Mappings:** Categorical features (e.g., tickers) are mapped to integers and stored in `feature_mappings.json`.

### 2. Model Layer (ML Models)
- **Model Registry:** Each model (logistic regression, ensemble, etc.) is a Python class with a standard interface.
- **Model Training:** The `train-model` command handles model training, validation, and evaluation.
- **Model Outputs:** Trained models and their artifacts are stored in `model_outputs/{model_name}/`.
- **ML Datasets:** Binary classification datasets are generated with time-series splits for training and evaluation.

### 3. Reporting Layer (Daily Reports)
- **Daily Markdown Reports:** Generated via `report-daily` command, listing top-N tickers by model predictions.
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
   - Initial logistic regression model implementation

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

### 🚧 Implementation Plan

1. **Dataset Generation Optimization** (Priority: High)
   - [ ] Fix progress bar to show progress across all splits
   - [ ] Implement dynamic dataset generation during training
   - [ ] Add memory-efficient batch processing
   - [ ] Implement feature caching for frequently used features
   - [ ] Add parallel processing for feature generation
   - [ ] Create dataset validation and quality checks

2. **CLI Integration** (Priority: High)
   - [ ] Add `train-model` command:
     - Model selection
     - Hyperparameter specification
     - Training data selection
     - Validation split configuration
   - [ ] Add `evaluate-model` command:
     - Performance metrics
     - Feature importance analysis
     - Confusion matrix generation
   - [ ] Add `predict` command:
     - Single ticker prediction
     - Batch prediction
     - Confidence scores
   - [ ] Add model management commands:
     - List available models
     - Save/load models
     - Delete models
     - Model versioning

3. **Codebase Cleanup** (Priority: Medium)
   - [ ] Remove unused files:
     - Old technical scorer outputs
     - Deprecated model files
     - Unused scripts
   - [ ] Standardize file organization:
     - Consistent directory structure
     - Clear naming conventions
     - Proper module organization
   - [ ] Update documentation:
     - Code comments
     - Function docstrings
     - README updates
     - Usage examples

4. **Model Improvements** (Priority: Medium)
   - [ ] Implement model versioning
   - [ ] Add model performance tracking
   - [ ] Create model comparison tools
   - [ ] Add ensemble model support
   - [ ] Implement parameter sweep functionality

5. **Testing and Validation** (Priority: High)
   - [ ] Add unit tests for new functionality
   - [ ] Implement integration tests
   - [ ] Add performance benchmarks
   - [ ] Create validation datasets
   - [ ] Implement cross-validation framework

6. **Documentation and Examples** (Priority: Medium)
   - [ ] Update README with new features
   - [ ] Create usage examples
   - [ ] Add API documentation
   - [ ] Create tutorial notebooks
   - [ ] Document best practices

### Timeline
- **Week 1**: Dataset Generation Optimization & CLI Integration
- **Week 2**: Codebase Cleanup & Model Improvements
- **Week 3**: Testing and Validation
- **Week 4**: Documentation and Examples

---

*This plan ensures a systematic approach to improving the codebase while maintaining focus on ML-driven trading strategies.*
