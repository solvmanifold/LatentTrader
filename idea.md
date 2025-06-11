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

### In Progress Features 🔄

#### 1. Feature Normalization
- **Normalization Architecture**
  - Create `FeatureNormalizer` class as part of dataset generation pipeline
  - Support multiple normalization strategies:
    - StandardScaler (z-score)
    - MinMaxScaler
    - RobustScaler
    - Custom normalization for specific features
  - Implement normalization versioning and tracking
  - Support for incremental updates and online learning
  - Integration with feature store for consistent access

- **Implementation Tasks**
  - [ ] Create `FeatureNormalizer` base class
  - [ ] Implement normalization strategies
  - [ ] Add normalization statistics storage
  - [ ] Create normalization versioning system
  - [ ] Implement incremental update support
  - [ ] Add validation for normalization consistency
  - [ ] Create migration script for existing datasets
  - [ ] Add support for feature-specific normalization
  - [ ] Implement normalization monitoring
  - [ ] Add comprehensive tests
  - [ ] Create documentation and examples

- **Technical Requirements**
  - Integration with `DatasetGenerator` class
  - Support for both batch and online normalization
  - Efficient storage of normalization parameters
  - Validation of normalization consistency
  - Support for feature versioning
  - Performance optimization for large datasets
  - Clear error handling and reporting
  - Comprehensive logging and monitoring

- **Data Quality and Validation**
  - Implement normalization quality checks
  - Add validation for normalization parameters
  - Create monitoring for normalization drift
  - Support for normalization diagnostics
  - Integration with data quality pipeline
  - Clear error reporting and handling
  - Support for normalization debugging

#### 2. Dataset Generation Enhancement
- **Dataset Generation Pipeline**
  - Create `DatasetGenerator` class with standardized interface
  - Implement time-series aware train/test/validation splits
  - Focus on binary classification targets
  - Configurable feature selection and filtering
  - Built-in data quality checks and validation
  - Support for feature versioning and tracking
  - Integration with feature store for consistent access

- **Label Generation Strategy**
  - Binary classification based on multi-factor criteria:
    - Price return vs S&P 500 (e.g., 1% absolute, 0.5% alpha)
    - Volume requirements (e.g., 100k shares)
    - Risk limits (e.g., 2x SPY volatility, 2% max drawdown)
  - Configurable risk profiles:
    - Conservative: Higher thresholds, stricter limits
    - Moderate: Medium thresholds, balanced limits
    - Aggressive: Lower thresholds, more lenient limits
  - Quality control:
    - Label distribution monitoring
    - Market regime correlation
    - Performance metrics tracking

- **Feature Collection**
  - Single interface for gathering features via `FeatureCollector` class
  - Support for both batch and single-row predictions
  - Consistent feature ordering across all uses
  - Clear error handling for missing features
  - Support for feature imputation with stored statistics
  - Validation of feature completeness and quality
  - Caching layer for frequently accessed features
  - Support for feature subset selection and filtering

- **Prediction Support**
  - Enable single-row predictions using dataset metadata
  - Support for new data points outside training set
  - Consistent feature processing pipeline
  - Clear documentation of required features
  - Integration with model prediction pipeline
  - Support for feature validation and error reporting
  - Real-time feature computation for live predictions
  - Support for feature versioning and tracking

- **Data Versioning and Reproducibility**
  - Implement dataset versioning system
  - Store dataset metadata and configuration
  - Support for dataset comparison and validation
  - Track feature versions and dependencies
  - Enable dataset regeneration from stored configs
  - Support for dataset lineage tracking
  - Integration with experiment tracking

- **Technical Requirements**
  - Python 3.8+ compatibility
  - Pandas for data manipulation
  - NumPy for numerical operations
  - scikit-learn for data splitting and validation
  - PyArrow for efficient data storage
  - Type hints and documentation
  - Comprehensive test coverage
  - Performance benchmarks for critical paths

- **Implementation Tasks**
  - [ ] Create `DatasetGenerator` base class
  - [ ] Implement time-series aware splits
  - [ ] Add feature collection interface
  - [ ] Implement data quality checks
  - [ ] Add feature versioning support
  - [ ] Create dataset versioning system
  - [ ] Implement caching layer
  - [ ] Add performance benchmarks
  - [ ] Create comprehensive tests
  - [ ] Add documentation and examples

### Pending Features ⏳

#### 1. VIX Integration (Remaining)
- [ ] `market_volatility_vix_ma20`
- [ ] `market_volatility_vix_std20`

#### 2. Market-Wide Volatility (Remaining)
- [ ] `market_volatility_vol_of_vol`
- [ ] `market_volatility_cross_sectional_vol`

#### 3. Model Training Enhancement
- [ ] Enhance `train_logistic2.py`:
  - [ ] Add hyperparameter tuning
  - [ ] Implement cross-validation
  - [ ] Add early stopping
  - [ ] Improve logging and visualization
- [ ] Create `train_ensemble.py`:
  - [ ] Implement bagging/boosting
  - [ ] Add model stacking
  - [ ] Support multiple base models
- [ ] Add model evaluation:
  - [ ] Performance metrics
  - [ ] Feature importance analysis
  - [ ] Confusion matrix generation
- [ ] Create model management:
  - [ ] Model versioning
  - [ ] Model comparison
  - [ ] Model deployment

#### 4. Testing and Validation
- [ ] Expand test coverage:
  - [ ] Add unit tests for new functionality
  - [ ] Implement integration tests
  - [ ] Add performance benchmarks
- [ ] Create validation datasets:
  - [ ] Implement cross-validation framework
  - [ ] Add data quality checks
  - [ ] Create validation reports

#### 5. Documentation and Examples
- [ ] Update documentation:
  - [ ] Code comments
  - [ ] Function docstrings
  - [ ] README updates
- [ ] Create examples:
  - [ ] Usage examples
  - [ ] Tutorial notebooks
  - [ ] Best practices guide

## Timeline
- **Week 1**: Complete Feature Normalization
- **Week 2**: Complete Dataset Generation
- **Week 3**: Complete Prediction Pipeline & Model Training
- **Week 4**: Complete Testing & Documentation

---

*This plan ensures a systematic approach to improving the codebase while maintaining focus on ML-driven trading strategies.*
