# LatentTrader

A machine learning-based trading advisor that uses logistic regression for market predictions.

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

## CLI Commands

### Data Management

```bash
# Update market and ticker data
trading-advisor update-data \
    --tickers AAPL,MSFT,GOOGL \
    --days 60 \
    --update-tickers \
    --update-market

# Generate ML dataset
trading-advisor generate-dataset \
    --tickers AAPL,MSFT,GOOGL \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --train-months 6 \
    --val-months 2

# Generate labels for ML dataset
trading-advisor generate-labels \
    --input-dir data/ml_datasets \
    --label-types short_term_profit,risk_adjusted
```

### Model Operations

```bash
# Run model predictions
trading-advisor run-model \
    --model-name logistic \
    --tickers AAPL,MSFT,GOOGL \
    --date 2024-03-20

# List available models
trading-advisor list-models
```

### Reporting

```bash
# Generate daily report
trading-advisor report-daily \
    --model-name TechnicalScorer \
    --date 2024-03-20 \
    --top-n 6

# Generate LLM prompt
trading-advisor prompt-daily \
    --model-name TechnicalScorer \
    --date 2024-03-20 \
    --deep-research
```

## Inference

The system supports generating normalized feature data for inference using the same preprocessing as training:

```python
from trading_advisor.dataset_v2 import DatasetGeneratorV2
from datetime import datetime

# Initialize generator with same directories used for training
generator = DatasetGeneratorV2(
    market_features_dir="data/market_features",
    ticker_features_dir="data/ticker_features"
)

# Prepare single row for inference
date = datetime(2024, 1, 2)
inference_data = generator.prepare_inference_data('AAPL', date)
```

The inference data will be normalized using the same statistics (mean, std) that were calculated from the training set, ensuring consistency between training and inference.

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