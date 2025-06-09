# LatentTrader

A machine learning-powered trading assistant that generates actionable, short-term trading playbooks designed to beat the market.

## Features

- **Market Data Collection**
  - Historical stock data retrieval using `yfinance`
  - Market breadth indicators
  - Sector performance analysis
  - Market sentiment analysis (GDELT)
  - Volatility measures (VIX, market-wide)

- **Feature Engineering**
  - Technical indicators (RSI, MACD, etc.)
  - Market-wide features
  - Sector-level metrics
  - Sentiment indicators
  - Feature preprocessing and normalization
  - Data validation framework

- **Machine Learning Models**
  - Logistic regression for binary classification
  - Time-series aware cross-validation
  - Feature importance analysis
  - Model performance tracking
  - Ensemble model support (coming soon)

- **Reporting and Analysis**
  - Daily markdown-formatted reports
  - Model performance metrics
  - Feature importance analysis
  - Historical tracking and analysis
  - LLM-ready prompts for trading decisions

## Installation

```bash
pip install -e .
```

## Usage

### Core CLI Commands

```bash
# Update data for all tickers
python -m trading_advisor update-data

# Generate a dataset for machine learning
python -m trading_advisor generate-dataset --start-date 2023-01-01 --end-date 2024-01-01

# Run a model on specific tickers
python -m trading_advisor run-model --model-name TechnicalScorer --tickers AAPL,MSFT,GOOGL

# Generate a report for a specific date
python -m trading_advisor report-daily --date 2024-03-20

# Generate a prompt for a specific date
python -m trading_advisor prompt-daily --date 2024-03-20
```

### Model Training

Model training and evaluation are handled through dedicated scripts:

```bash
# Train a logistic regression model
python scripts/train_logistic2.py --dataset test_run

# Evaluate a trained model
python scripts/train_logistic2.py --dataset test_run --evaluate

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
├── model_outputs/           # Model outputs and artifacts
├── reports/                 # Daily reports
├── prompts/                # LLM prompts
├── scripts/
│   ├── train_logistic2.py  # Model training
│   └── analyze_labels.py   # Dataset analysis
├── src/
│   └── trading_advisor/
│       ├── models/         # Model implementations
│       ├── features/       # Feature engineering
│       └── output/         # Report generation
└── tests/                  # Test suite
```

## Data Validation

The project includes a comprehensive data validation framework that ensures:

- **Data Quality**: Validation of data types, missing values, and outliers
- **Data Completeness**: Checking for missing trading days and data gaps
- **Data Consistency**: Ensuring consistency across files and calculations
- **Performance**: Testing with large datasets and monitoring resource usage

## Logging

All activity is logged to both the terminal and log files in the `logs/` directory. Log files are organized by date and component.

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- yfinance
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