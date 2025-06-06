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