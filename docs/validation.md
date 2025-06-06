# Data Validation Framework

The LatentTrader project includes a comprehensive data validation framework to ensure data quality and consistency across all data files. This framework helps maintain our data standards and catch issues early in the data pipeline.

## Overview

The validation framework provides several validators that check:
- File naming conventions
- Column naming conventions
- Data types and formats
- Required columns
- Data quality metrics

## Usage

### Basic Usage

The simplest way to validate a Parquet file is using the `validate_parquet_file` function:

```python
from trading_advisor.validation import validate_parquet_file
import numpy as np

# Define expected types for your columns
expected_types = {
    'date': np.datetime64,
    'close_price': np.floating,
    'volume': np.integer
}

# Define required columns
required_columns = ['date', 'close_price']

# Validate a file
is_valid = validate_parquet_file(
    'data/market_features/daily_breadth.parquet',
    expected_types=expected_types,
    required_columns=required_columns
)
```

### Individual Validators

For more specific validation needs, you can use the individual validators:

```python
from trading_advisor.validation import (
    FileNameValidator,
    ColumnNameValidator,
    DataTypeValidator,
    RequiredColumnValidator,
    DataQualityValidator
)

# Validate file name
file_validator = FileNameValidator('data/market_features/daily_breadth.parquet')
if file_validator.validate():
    print("File name is valid")
else:
    print("File name validation errors:", file_validator.get_errors())

# Validate DataFrame
df = pd.read_parquet('data/market_features/daily_breadth.parquet')

# Validate column names
col_validator = ColumnNameValidator(df)
if col_validator.validate():
    print("Column names are valid")
else:
    print("Column name validation errors:", col_validator.get_errors())

# Validate data quality
quality_validator = DataQualityValidator(df)
quality_validator.validate()
print("Data quality warnings:", quality_validator.get_warnings())
```

## Validation Rules

### File Names
- Must be lowercase
- Use underscores instead of spaces
- Must have .parquet extension
- Example: `daily_breadth.parquet`

### Column Names
- Must be lowercase
- Use underscores instead of spaces
- Can only contain letters, numbers, and underscores
- Example: `close_price`, `volume_ma20`

### Data Types
- Date columns must be datetime64
- Price columns must be floating point
- Volume columns must be integer
- Custom types can be specified using numpy types

### Data Quality
The framework checks for:
- Missing values (reported as warnings)
- Infinite values (reported as errors)
- Duplicate rows (reported as warnings)

## Integration with Data Pipeline

The validation framework is designed to be integrated into your data processing pipeline. Here's an example of how to use it in a data update function:

```python
def update_market_features():
    # Generate or update features
    df = generate_market_features()
    
    # Validate before saving
    expected_types = {
        'date': np.datetime64,
        'close_price': np.floating,
        'volume': np.integer
    }
    required_columns = ['date', 'close_price']
    
    if validate_parquet_file('data/market_features/daily_breadth.parquet',
                           expected_types=expected_types,
                           required_columns=required_columns):
        df.to_parquet('data/market_features/daily_breadth.parquet')
    else:
        raise ValueError("Data validation failed")
```

## Running Tests

The validation framework includes comprehensive tests. To run them:

```bash
python -m pytest tests/test_validation.py -v
```

## Error Handling

The framework provides detailed error messages to help identify and fix issues:

```python
validator = FileNameValidator('data/market_features/DailyBreadth.parquet')
if not validator.validate():
    for error in validator.get_errors():
        print(f"Error: {error}")
```

## Contributing

When adding new features to the validation framework:
1. Add new validator classes if needed
2. Update tests to cover new functionality
3. Update this documentation
4. Follow the existing error/warning pattern

## Additional Examples

### Real-world Examples

Here are examples of validation configurations for different types of data files:

```python
# Market Features Example
market_expected_types = {
    'date': np.datetime64,
    'vix': np.floating,
    'vix_ma20': np.floating,
    'market_volatility': np.floating,
    'cross_sectional_vol': np.floating
}
market_required_columns = ['date', 'vix', 'market_volatility']

# Ticker Features Example
ticker_expected_types = {
    'date': np.datetime64,
    'close_price': np.floating,
    'volume': np.integer,
    'rsi_14': np.floating,
    'macd': np.floating,
    'macd_signal': np.floating
}
ticker_required_columns = ['date', 'close_price', 'volume']
```

### Common Validation Patterns

```python
# Validating multiple files in a directory
def validate_directory(directory: str, expected_types: Dict[str, type], required_columns: List[str]):
    for file in Path(directory).glob('*.parquet'):
        if not validate_parquet_file(str(file), expected_types, required_columns):
            print(f"Validation failed for {file}")

# Validating with custom quality thresholds
class CustomQualityValidator(DataQualityValidator):
    def validate(self) -> bool:
        # Check for missing values above threshold
        null_ratio = self.df.isnull().mean()
        if (null_ratio > 0.1).any():
            self.add_error(f"Columns with >10% missing values: {null_ratio[null_ratio > 0.1].index.tolist()}")
        return len(self.errors) == 0
```

### Integration Examples

```python
# In market_breadth.py
def update_market_breadth():
    df = calculate_market_breadth()
    
    # Validate before saving
    if not validate_parquet_file(
        'data/market_features/daily_breadth.parquet',
        expected_types=market_expected_types,
        required_columns=market_required_columns
    ):
        raise ValueError("Market breadth data validation failed")
    
    df.to_parquet('data/market_features/daily_breadth.parquet')
```

### Error Recovery Examples

```python
# Handling validation errors with recovery
def safe_update_features():
    try:
        df = generate_features()
        if not validate_parquet_file('features.parquet', expected_types, required_columns):
            # Try to fix common issues
            df = df.fillna(method='ffill')  # Forward fill missing values
            df = df.replace([np.inf, -np.inf], np.nan)  # Replace infinite values
            
            # Validate again
            if not validate_parquet_file('features.parquet', expected_types, required_columns):
                raise ValueError("Could not fix validation issues")
        
        df.to_parquet('features.parquet')
    except Exception as e:
        logger.error(f"Feature update failed: {e}")
        # Implement fallback or notification
```

### Custom Validator Examples

```python
# Custom validator for time series data
class TimeSeriesValidator(DataValidator):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df
        # Get list of trading days (excluding weekends and holidays)
        self.trading_days = pd.bdate_range(
            start=df['date'].min(),
            end=df['date'].max(),
            freq='B'  # Business day frequency
        )
    
    def validate(self) -> bool:
        # Check for date ordering
        if not self.df['date'].is_monotonic_increasing:
            self.add_error("Dates are not in ascending order")
        
        # Check for missing trading days
        missing_days = set(self.trading_days) - set(self.df['date'])
        if missing_days:
            self.add_warning(f"Missing {len(missing_days)} trading days")
        
        # Check for duplicate dates
        if self.df['date'].duplicated().any():
            self.add_error("Duplicate dates found in time series")
        
        return len(self.errors) == 0
```

This TimeSeriesValidator now:
1. Uses `pd.bdate_range` to get business days (excluding weekends)
2. Checks for missing trading days instead of calendar days
3. Adds a check for duplicate dates
4. Properly handles the trading day calendar 