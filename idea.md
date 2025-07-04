# LatentTrader: Vision and Architecture

## Project Vision
LatentTrader aims to be a machine learning-powered trading assistant that generates actionable, short-term trading playbooks designed to beat the market. The system leverages both traditional technical analysis features and modern ML models to create, test, and iterate on predictive models. The ultimate goal is to support robust experimentation, backtesting, and rapid iteration to discover and deploy the most effective trading strategies.

## System Architecture

### 1. Data Layer (Features)
- **Per-Ticker Features:** All raw and engineered features (technical indicators, price/volume, etc.) are stored in Parquet files, one per ticker (e.g., `ticker_features/AAPL_features.parquet`).
- **Market-Wide Features:** Market features are stored in separate Parquet files by category (volatility, breadth, sentiment, etc.) for independent updates.
- **Sector Features:** Sector-level metrics are stored in separate files under `market_features/sectors/`.
- **Update Process:** Features are updated via the `update-data` command with flags for different feature types.
- **Feature Mappings:** Categorical features (e.g., tickers) are mapped to integers and stored in `feature_mappings.json`.

### 2. Model Layer (ML Models)
- **Model Registry:** Each model (logistic regression, ensemble, etc.) is a Python class with a standard interface.
- **Model Training:** Training is handled through dedicated scripts in the `scripts/` directory.
- **Model Outputs:** Trained models and their artifacts are stored in `model_outputs/{model_name}/`.
- **ML Datasets:** Binary classification datasets are generated with time-series splits for training and evaluation.

### 3. Reporting Layer (Daily Reports)
- **Daily Markdown Reports:** Generated via `report-daily` command, listing top-N tickers by model predictions.
- **Historical Tracking:** Reports are saved as both Markdown files and Parquet tables.
- **Consistent, Auditable, Historical:** Easy querying and re-generation of reports for any day/model.

### 4. Prompting Layer
- **Prompt Generation:** The `prompt-daily` command generates LLM prompts from daily reports.
- **Historical Tracking:** Prompts are saved as both text files and Parquet tables.

## Feature Architecture

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
   - Market-wide volatility measures:
     - Daily volatility
     - Weekly volatility
     - Monthly volatility
   - Average correlation between stocks
   - Ticker-specific volatility

4. **Market Sentiment** (daily/weekly updates)
   - Put/Call ratios
   - Short interest trends
   - Analyst sentiment aggregation
   - GDELT news sentiment

## Implementation Status

### Completed Features ✅

#### 1. Data Standardization
- **File Naming Convention**
  - All filenames are lowercase with underscores
  - Example: `communication_services.parquet` instead of `Communication Services.parquet`
  - Consistent naming across all feature files

- **Column Naming Convention**
  - All column names are lowercase with underscores
  - No spaces in column names
  - Market feature columns include source filename as prefix
  - Example: `daily_breadth_above_ma20` instead of `Above MA20`

- **DataFrame Structure**
  - `date` column included in all files
  - Consistent datetime format
  - Clear separation between index and columns

- **Market Features Standardization**
  - Added `daily_breadth_` prefix to market breadth columns
  - Added `market_volatility_` prefix to volatility columns
  - Added `market_sentiment_` prefix to sentiment columns
  - Added sector prefix to sector performance columns
  - Updated validation rules to enforce prefixes

#### 2. VIX Integration
- Added VIX data download and storage
- Implemented VIX indicators:
  - `market_volatility_vix`
- Added VIX validation rules
- Updated documentation

#### 3. Market-Wide Volatility
- Implemented market-wide volatility measures:
  - `market_volatility_market_volatility`
- Added validation rules
- Updated documentation

#### 4. Dataset Generation
- Implemented robust dataset generation pipeline
- Added validation checks for data quality
- Created progress tracking for long-running operations
- Implemented feature loading from pre-computed files
- Added support for sector performance metrics
- Created validation framework for dataset consistency
- Implemented efficient feature storage and retrieval
- Added support for feature versioning and tracking

#### 5. Feature Normalization
- Implemented normalization as part of dataset generation pipeline
- Added support for multiple normalization strategies:
  - StandardScaler (z-score)
  - MinMaxScaler
  - RobustScaler
- Integrated with `update-data` and `generate-dataset` commands
- Implemented efficient storage of normalization parameters
- Added validation for normalization consistency
- Created comprehensive logging and monitoring
- Added support for feature-specific normalization
- Implemented normalization quality checks
- Added validation for normalization parameters
- Created monitoring for normalization drift

### In Progress Features 🔄

#### 1. Model Development
- **Model Architecture**
  - Create standardized model interface
  - Implement model versioning and tracking
  - Support for model evaluation metrics
  - Integration with dataset generation pipeline
  - Support for model persistence and loading
  - Implementation of prediction pipeline

- **Initial Models**
  - Logistic Regression with feature selection
  - Random Forest for feature importance
  - Gradient Boosting for improved performance
  - Ensemble methods for robust predictions

- **Model Evaluation**
  - Time-series aware cross-validation
  - Performance metrics tracking
  - Feature importance analysis
  - Model comparison framework
  - Backtesting integration

- **Implementation Tasks**
  - [ ] Create `ModelBase` class with standard interface
  - [ ] Implement model training pipeline
  - [ ] Add model evaluation framework
  - [ ] Create model persistence system
  - [ ] Implement prediction pipeline
  - [ ] Add model versioning support
  - [ ] Create comprehensive tests
  - [ ] Add documentation and examples

### Pending Features ⏳

#### 1. Model Training Enhancement
- [ ] Create model training pipeline:
  - [ ] Implement hyperparameter tuning
  - [ ] Add cross-validation framework
  - [ ] Create model comparison tools
  - [ ] Add performance visualization
  - [ ] Implement model persistence
  - [ ] Add prediction pipeline
  - [ ] Create model documentation
  - [ ] Add example notebooks

#### 2. Model Evaluation Framework
- [ ] Implement evaluation metrics:
  - [ ] Accuracy, precision, recall
  - [ ] ROC and AUC analysis
  - [ ] Feature importance visualization
  - [ ] Performance over time analysis
  - [ ] Risk-adjusted returns
  - [ ] Drawdown analysis
  - [ ] Sharpe ratio calculation
  - [ ] Maximum drawdown tracking

#### 3. Model Deployment
- [ ] Create deployment pipeline:
  - [ ] Model versioning system
  - [ ] A/B testing framework
  - [ ] Performance monitoring
  - [ ] Automated retraining
  - [ ] Model rollback capability
  - [ ] Prediction logging
  - [ ] Error tracking
  - [ ] Performance alerts

#### 4. Market-Wide Volatility (Remaining)
- [ ] `market_volatility_vol_of_vol`
- [ ] `market_volatility_cross_sectional_vol`

## Timeline
- **Week 1**: Complete Feature Normalization
- **Week 2**: Complete Dataset Generation
- **Week 3**: Complete Prediction Pipeline & Model Training
- **Week 4**: Complete Testing & Documentation

---

*This plan ensures a systematic approach to improving the codebase while maintaining focus on ML-driven trading strategies.*
