# Trading Advisor

A Python-based trading advisor that generates technical analysis and trading recommendations.

## Features

- **Market Data Collection**
  - Historical stock data retrieval using `yfinance`
  - Market breadth indicators
  - Sector performance analysis
  - Market sentiment analysis (GDELT)
  - Volatility measures

- **Feature Engineering**
  - Technical indicators (RSI, MACD, etc.)
  - Market-wide features
  - Sector-level metrics
  - Sentiment indicators
  - Feature preprocessing and normalization
  - Data validation framework ([docs](docs/validation.md))

- **Machine Learning Models**
  - Logistic regression for binary classification
  - Time-series aware cross-validation
  - Feature importance analysis
  - Model performance tracking
  - Ensemble model support (coming soon)

- **Reporting and Analysis**
  - Daily markdown-formatted reports
  - Interactive charts and visualizations
  - Model performance metrics
  - Feature importance plots
  - Historical tracking and analysis

## Installation

```bash
pip install -e .
```

## Usage

### Core CLI Commands

The Trading Advisor provides several command-line tools for data management and reporting:

```bash
# Generate a report for a specific date
python -m trading_advisor report-daily --date 2024-03-20

# Generate a prompt for a specific date
python -m trading_advisor prompt-daily --date 2024-03-20

# Update data for all tickers
python -m trading_advisor update-data

# Generate a dataset for machine learning
python -m trading_advisor generate-dataset --start-date 2023-01-01 --end-date 2024-01-01
```

### Model Training Scripts

Model training and evaluation are handled through separate Python scripts in the `scripts/` directory:

```bash
# Train a logistic regression model on a specific dataset
python scripts/train_logistic2.py --dataset test_run

# Train on a different dataset
python scripts/train_logistic2.py --dataset large_test

# Evaluate a previously trained model
python scripts/train_logistic2.py --dataset test_run --evaluate

# Show detailed logging information
python scripts/train_logistic2.py --dataset test_run --verbose
```

The training script will:
1. Load the specified dataset from `data/ml_datasets/{dataset_name}/`
2. Train a logistic regression model with the following parameters:
   - C: 0.1 (strong regularization)
   - class_weight: 'balanced' (handles class imbalance)
   - max_iter: 1000
   - solver: 'liblinear' (good for small datasets)
3. Save model outputs to `model_outputs/logistic2/{dataset_name}/`:
   - `logistic_model.pkl`: Trained model
   - `scaler.pkl`: Feature scaler
   - `features.pkl`: Feature column names
   - `metrics.json`: Performance metrics
   - `confusion_matrix_split_0.png`: Confusion matrix plot
   - `feature_importance_split_0.png`: Feature importance plot

The model outputs are organized by dataset name to prevent overwriting when training on different datasets.

When using the `--evaluate` flag, the script will:
1. Load a previously trained model and its artifacts
2. Evaluate the model on the test set
3. Generate new performance metrics and visualizations
4. Update the metrics.json file with the latest results

By default, the script shows only essential information. Use the `--verbose` flag to see detailed logging including feature statistics and intermediate steps.

# Analyze dataset labels
python scripts/analyze_labels.py --dataset-dir data/ml_datasets/test_run
```

## Project Structure

```
LatentTrader/
├── data/
│   ├── market_features/     # Market-wide features
│   ├── ticker_features/     # Per-ticker features
│   └── ml_datasets/         # ML training datasets
├── model_outputs/
│   └── logistic2/          # Model outputs and artifacts
├── scripts/
│   ├── train_logistic2.py  # Model training script
│   ├── evaluate_logistic.py # Model evaluation
│   └── analyze_labels.py   # Dataset analysis
├── src/
│   └── trading_advisor/
│       ├── models/         # Model implementations
│       ├── features/       # Feature engineering
│       └── reporting/      # Report generation
└── tests/                  # Test suite
```

## Recent Improvements

- Added logistic regression model with cross-validation
- Implemented feature importance analysis
- Added model performance tracking
- Improved dataset generation with time-series splits
- Enhanced reporting with model predictions

## Next Steps

1. **Dataset Generation Optimization**
   - Implement dynamic dataset generation
   - Add memory-efficient batch processing
   - Improve feature caching

2. **Model Improvements**
   - Add ensemble model support
   - Implement parameter sweeps
   - Add model versioning

3. **Testing and Validation**
   - Expand test coverage
   - Add performance benchmarks
   - Implement cross-validation framework

## Logging

All activity is logged to both the terminal and log files in the `logs/` directory. Log files are organized by date and component.

## Test Coverage

Run the test suite with:
```bash
pytest tests/
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- yfinance
- plotly
- pytest (for testing)

See `requirements.txt` for a complete list of dependencies.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 