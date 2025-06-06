# Data Validation Framework

This document describes the validation framework used in the LatentTrader project to ensure data quality and consistency.

## Overview

The validation framework consists of several components that work together to validate different aspects of the data:

1. **File Name Validation**
   - Ensures consistent file naming conventions
   - Validates file extensions
   - Enforces lowercase with underscores

2. **Column Name Validation**
   - Ensures consistent column naming conventions
   - Validates column name format
   - Enforces lowercase with underscores

3. **Data Type Validation**
   - Validates data types of columns
   - Ensures consistent datetime formats
   - Validates numeric data types

4. **Required Column Validation**
   - Ensures all required columns are present
   - Validates column dependencies
   - Checks for missing columns

5. **Data Quality Validation**
   - Checks for missing values
   - Validates data ranges
   - Checks for outliers
   - Validates data consistency

6. **Data Completeness Validation**
   - Ensures all trading days are present
   - Validates data gaps
   - Checks for missing data points

7. **Data Consistency Validation**
   - Ensures consistency across files
   - Validates sector mappings
   - Checks for data alignment

8. **Performance Validation**
   - Tests with large datasets
   - Validates memory usage
   - Checks computation time

## Validation Tests

### Basic Validation Tests

```python
def test_file_name_validator():
    """Test file name validation."""
    # Tests for valid and invalid file names
    # Tests for file extensions
    # Tests for naming conventions

def test_column_name_validator():
    """Test column name validation."""
    # Tests for valid and invalid column names
    # Tests for naming conventions
    # Tests for required columns

def test_data_type_validator():
    """Test data type validation."""
    # Tests for data types
    # Tests for datetime formats
    # Tests for numeric types

def test_required_column_validator():
    """Test required column validation."""
    # Tests for required columns
    # Tests for column dependencies
    # Tests for missing columns

def test_data_quality_validator():
    """Test data quality validation."""
    # Tests for missing values
    # Tests for data ranges
    # Tests for outliers
    # Tests for data consistency
```

### Advanced Validation Tests

```python
def test_data_completeness():
    """Test data completeness."""
    # Tests for missing trading days
    # Tests for data gaps
    # Tests for data alignment

def test_data_consistency():
    """Test data consistency."""
    # Tests for consistency across files
    # Tests for sector mappings
    # Tests for data alignment

def test_edge_cases():
    """Test edge cases."""
    # Tests for empty DataFrames
    # Tests for single-row DataFrames
    # Tests for all NaN values
    # Tests for extreme values

def test_performance():
    """Test performance."""
    # Tests for large datasets
    # Tests for memory usage
    # Tests for computation time

def test_sector_mapping_consistency():
    """Test sector mapping consistency."""
    # Tests for valid sectors
    # Tests for multiple sectors
    # Tests for sector dependencies

def test_market_features_consistency():
    """Test market features consistency."""
    # Tests for feature alignment
    # Tests for feature dependencies
    # Tests for feature calculations

def test_data_pipeline():
    """Test data pipeline."""
    # Tests for data download
    # Tests for data processing
    # Tests for data storage
    # Tests for data updates

def test_data_updates():
    """Test data updates."""
    # Tests for incremental updates
    # Tests for data preservation
    # Tests for data consistency

def test_data_versioning():
    """Test data versioning."""
    # Tests for version metadata
    # Tests for version consistency
    # Tests for version updates
```

## Usage

The validation framework can be used in several ways:

1. **Command Line**
   ```bash
   python -m trading_advisor.validation validate_data
   ```

2. **Python API**
   ```python
   from trading_advisor.validation import validate_parquet_file
   
   # Validate a single file
   validate_parquet_file('data/market_features/daily_breadth.parquet')
   
   # Validate multiple files
   validate_parquet_file('data/market_features/*.parquet')
   ```

3. **Integration with Data Pipeline**
   ```python
   from trading_advisor.validation import DataQualityValidator
   
   # Validate data during processing
   validator = DataQualityValidator(df)
   if not validator.validate():
       raise ValidationError(validator.get_errors())
   ```

## Best Practices

1. **Run Validation Early**
   - Validate data as soon as it's loaded
   - Validate data before processing
   - Validate data before storage

2. **Validate Consistently**
   - Use the same validation rules across the project
   - Validate all data files
   - Validate all data updates

3. **Handle Validation Errors**
   - Log validation errors
   - Raise appropriate exceptions
   - Provide clear error messages

4. **Monitor Validation Performance**
   - Track validation time
   - Monitor memory usage
   - Optimize validation rules

## Future Improvements

1. **Enhanced Validation Rules**
   - Add more validation rules
   - Improve error messages
   - Add validation documentation

2. **Performance Optimization**
   - Optimize validation speed
   - Reduce memory usage
   - Add parallel validation

3. **Integration Improvements**
   - Add CI/CD integration
   - Add monitoring integration
   - Add reporting integration 