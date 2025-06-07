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

## Data Layer Improvements

### 1. Data Standardization (Priority: High)
- **File Naming Convention**
  - All filenames should be lowercase with underscores
  - Example: `communication_services.parquet` instead of `Communication Services.parquet`
  - Consistent naming across all feature files

- **Column Naming Convention**
  - All column names should be lowercase with underscores
  - No spaces allowed in column names
  - Market feature columns should include source filename as prefix
  - Example: `daily_breadth_above_ma20` instead of `Above MA20`

- **DataFrame Structure**
  - Always include a `date` column (even when using date as index)
  - Consistent datetime format across all files
  - Clear separation between index and columns

### 2. Feature Normalization (Priority: High)
- **Normalization Process**
  - Move normalization to dataset generation phase
  - Store normalization statistics (mean, std) in a pickle file
  - Enable consistent normalization for new predictions
  - Support incremental updates with existing normalization

- **Implementation Tasks**
  - [ ] Move normalization to dataset generation
  - [ ] Create normalization statistics storage
  - [ ] Implement normalization for new predictions
  - [ ] Add validation for normalization consistency
  - [ ] Create migration script for existing datasets
  - [ ] Add support for different normalization strategies
  - [ ] Implement normalization versioning
  - [ ] Document feature normalization dependencies
  - [ ] Add data quality checks for normalization
  - [ ] Create normalization monitoring system

### 3. Dataset Generation Enhancement
- **Feature Collection**
  - Single interface for gathering features
  - Support for both batch and single-row predictions
  - Consistent feature ordering across all uses
  - Clear error handling for missing features
  - Support for feature imputation with stored statistics
  - Validation of feature completeness and quality

- **Prediction Support**
  - Enable single-row predictions using dataset metadata
  - Support for new data points outside training set
  - Consistent feature processing pipeline
  - Clear documentation of required features
  - Integration with model prediction pipeline
  - Support for feature validation and error reporting

- **Feature Consistency**
  - Ensure same features used in training and prediction
  - Support for feature versioning and tracking
  - Validation of feature availability and quality
  - Clear error messages for missing or invalid features
  - Support for feature subset selection

### Implementation Plan

1. **Phase 1: Data Standardization** ✅
   - [x] Create new data structure
   - [x] Write migration scripts
   - [x] Test with subset of data
   - [x] Full data migration
   - [x] Update `update-data` to enforce naming conventions
   - [x] Add column name standardization
   - [x] Implement consistent date column handling
   - [x] Add validation for file and column names
   - [x] Create data validation framework
   - [x] Integrate validation into data pipeline:
     - [x] Add validation during data loading
     - [x] Add validation before processing
     - [x] Add validation before storage
     - [x] Add validation during updates
   - [x] Establish data quality metrics and monitoring

2. **Phase 2: Feature Normalization** (In Progress)
   - [ ] Implement new normalization
   - [ ] Create statistics storage
   - [ ] Test with existing models
   - [ ] Update prediction pipeline
   - [ ] Move normalization to dataset generation
   - [ ] Create normalization statistics storage
   - [ ] Implement normalization for new predictions
   - [ ] Add validation for normalization consistency
   - [ ] Create migration script for existing datasets
   - [ ] Add support for different normalization strategies
   - [ ] Implement normalization versioning
   - [ ] Document feature normalization dependencies
   - [ ] Add data quality checks for normalization
   - [ ] Create normalization monitoring system

3. **Phase 3: Dataset Generation** (Next Up)
   - [ ] Update feature collection
   - [ ] Implement single-row support
   - [ ] Test with existing models
   - [ ] Update documentation
   - [ ] Update feature collection interface
   - [ ] Implement single-row prediction support
   - [ ] Add feature validation and error handling
   - [ ] Create documentation for feature requirements
   - [ ] Add tests for feature consistency
   - [ ] Implement feature versioning system
   - [ ] Add support for feature subset selection
   - [ ] Create feature quality metrics
   - [ ] Implement automated feature validation

4. **Phase 4: Prediction Pipeline**
   - [ ] Create unified prediction interface
   - [ ] Implement feature validation in predictions
   - [ ] Add support for feature imputation
   - [ ] Create prediction quality metrics
   - [ ] Implement prediction logging
   - [ ] Add support for batch predictions
   - [ ] Create prediction validation system

5. **Phase 5: Model Training Enhancement**
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

6. **Phase 6: Testing and Validation**
   - [ ] Expand test coverage:
     - [ ] Add unit tests for new functionality
     - [ ] Implement integration tests
     - [ ] Add performance benchmarks
   - [ ] Create validation datasets:
     - [ ] Implement cross-validation framework
     - [ ] Add data quality checks
     - [ ] Create validation reports

7. **Phase 7: Documentation and Examples**
   - [ ] Update documentation:
     - [ ] Code comments
     - [ ] Function docstrings
     - [ ] README updates
   - [ ] Create examples:
     - [ ] Usage examples
     - [ ] Tutorial notebooks
     - [ ] Best practices guide

### Timeline
- **Week 1**: Complete Phase 2 (Feature Normalization)
- **Week 2**: Complete Phase 3 (Dataset Generation)
- **Week 3**: Complete Phase 4 (Prediction Pipeline) & Phase 5 (Model Training)
- **Week 4**: Complete Phase 6 (Testing) & Phase 7 (Documentation)

---

*This plan ensures a systematic approach to improving the codebase while maintaining focus on ML-driven trading strategies.*
