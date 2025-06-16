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
    --val-months 2 \
    --min-samples-per-ticker 30

# Generate swing trade labels for ML dataset
trading-advisor generate-labels \
    --input-dir data/ml_datasets \
    --output-dir data/ml_datasets \
    --lookback-days 3 \
    --forward-days 5 \
    --min-return 0.02 \
    --max-drawdown 0.01
```

The label generation command creates swing trade labels based on:
- Lookback period for trend confirmation (default: 3 days)
- Forward period for profit target (default: 5 days)
- Minimum return required (default: 2%)
- Maximum allowed drawdown (default: 1%)

Labels are:
- 1: Long signal (uptrend + profit target + controlled drawdown)
- -1: Short signal (downtrend + profit target + controlled drawdown)
- 0: No trade (default)

### Model Operations

```bash
# Train a new model
python scripts/train_model.py <data_dir> <output_dir> <label_type> <model_type>

# Example:
python scripts/train_model.py data/2024_datasets/swing_trade models/output swing_trade logistic

# Run model predictions
trading-advisor run-model \
    --model-name logistic \
    --tickers AAPL,MSFT,GOOGL \
    --date 2024-03-20

# List available models
trading-advisor list-models
```

The training script (`train_model.py`) trains a new model and saves it along with performance metrics and visualizations. The script takes the following arguments:

- `data_dir`: Directory containing the training data
- `output_dir`: Directory where the model and artifacts will be saved
- `label_type`: Type of labels used for training (e.g., 'swing_trade')
- `model_type`: Type of model to train (e.g., 'logistic')

The script will:
1. Load and split the data into train/validation/test sets
2. Train the model using cross-validation
3. Evaluate performance on the test set
4. Generate performance metrics and visualizations
5. Save the model and all artifacts to the output directory

The output directory will contain:
- `model`: The trained model file
- `model.metadata`: Detailed model metadata
- `metrics.csv`: Performance metrics
- `confusion_matrix.png`: Confusion matrix visualization
- `feature_importance.png`: Feature importance plot
- `training_config.json`: Training configuration

### Reporting

```bash
# Generate daily report
trading-advisor report-daily \
    --model-name TechnicalScorer \
    --date 2024-03-20 \
    --top-n 6 \
    --positions-csv positions.csv  # Optional: include current positions

# Generate LLM prompt
trading-advisor prompt-daily \
    --model-name TechnicalScorer \
    --date 2024-03-20 \
    --deep-research  # Optional: generate detailed research prompt
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