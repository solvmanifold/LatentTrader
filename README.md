# LatentTrader

A machine learning-powered trading assistant that generates actionable, short-term trading playbooks designed to beat the market. The system leverages both traditional technical analysis features and modern ML models to create, test, and iterate on predictive models.

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

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LatentTrader.git
cd LatentTrader
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Management
```bash
# Update market data
python -m trading_advisor update-data

# Generate ML dataset
python -m trading_advisor generate-classification-dataset
```

### Model Training and Evaluation
```bash
# Train a model
python -m trading_advisor train-model --model logistic --data-path data/ml_datasets

# Evaluate model performance
python -m trading_advisor evaluate-model --model logistic --split test
```

### Reporting
```bash
# Generate daily report
python -m trading_advisor report-daily --model logistic

# Generate trading prompt
python -m trading_advisor prompt-daily --model logistic
```

## Project Structure

```
LatentTrader/
├── data/
│   ├── market_features/     # Market-wide features
│   ├── ticker_features/     # Per-ticker features
│   └── ml_datasets/         # ML training datasets
├── model_outputs/
│   └── logistic/           # Model outputs and artifacts
├── scripts/
│   ├── train_logistic.py   # Model training script
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

2. **CLI Integration**
   - Add model training commands
   - Implement model evaluation
   - Add prediction functionality

3. **Model Improvements**
   - Add ensemble model support
   - Implement parameter sweeps
   - Add model versioning

4. **Testing and Validation**
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