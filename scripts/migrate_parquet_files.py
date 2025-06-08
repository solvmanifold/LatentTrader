#!/usr/bin/env python3
"""
Script to migrate existing parquet files to remove duplicate date columns.
This script follows the new data structure requirements where:
- All data must be indexed by date using a DatetimeIndex
- No duplicate date columns are allowed
"""

import os
import pandas as pd
from pathlib import Path
from typing import List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_parquet_files(base_dir: str) -> List[Path]:
    """Get all parquet files in the given directory and its subdirectories."""
    base_path = Path(base_dir)
    return list(base_path.rglob("*.parquet"))

def migrate_file(file_path: Path) -> None:
    """Migrate a single parquet file to remove duplicate date columns."""
    try:
        # Read the parquet file
        df = pd.read_parquet(file_path)
        
        # Check if file needs migration
        if 'date' in df.columns and isinstance(df.index, pd.DatetimeIndex):
            logger.info(f"Migrating {file_path}")
            
            # Drop the date column
            df = df.drop(columns=['date'])
            
            # Save back to parquet
            df.to_parquet(file_path)
            logger.info(f"Successfully migrated {file_path}")
        else:
            logger.debug(f"No migration needed for {file_path}")
            
    except Exception as e:
        logger.error(f"Error migrating {file_path}: {str(e)}")

def main():
    """Main function to migrate all parquet files."""
    # Directories to process
    directories = [
        'data/ticker_features',
        'data/market_features',
        'data/market_features/sectors'
    ]
    
    # Process each directory
    for directory in directories:
        logger.info(f"Processing directory: {directory}")
        parquet_files = get_parquet_files(directory)
        
        for file_path in parquet_files:
            migrate_file(file_path)
    
    logger.info("Migration completed")

if __name__ == "__main__":
    main() 