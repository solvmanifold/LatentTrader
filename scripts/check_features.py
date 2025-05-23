"""Script to check feature table lengths and clean up corrupted ones."""

import os
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_feature_tables(features_dir: str = "features", min_rows: int = 1257):
    """Check all feature tables and report/clean up corrupted ones.
    
    Args:
        features_dir: Directory containing feature parquet files
        min_rows: Minimum number of rows expected in each table
    """
    features_path = Path(features_dir)
    if not features_path.exists():
        logger.error(f"Features directory {features_dir} does not exist")
        return
    
    corrupted_files = []
    for file in features_path.glob("*_features.parquet"):
        try:
            df = pd.read_parquet(file)
            if len(df) < min_rows:
                logger.warning(f"{file.name}: {len(df)} rows (expected at least {min_rows})")
                corrupted_files.append(file)
            else:
                logger.info(f"{file.name}: {len(df)} rows (OK)")
        except Exception as e:
            logger.error(f"Error reading {file.name}: {e}")
            corrupted_files.append(file)
    
    if corrupted_files:
        logger.info(f"\nFound {len(corrupted_files)} corrupted files:")
        for file in corrupted_files:
            logger.info(f"  - {file.name}")
        
        # Ask for confirmation before deleting
        response = input("\nDelete corrupted files? (y/N): ")
        if response.lower() == 'y':
            for file in corrupted_files:
                try:
                    file.unlink()
                    logger.info(f"Deleted {file.name}")
                except Exception as e:
                    logger.error(f"Error deleting {file.name}: {e}")
        else:
            logger.info("No files were deleted")
    else:
        logger.info("No corrupted files found")

if __name__ == "__main__":
    check_feature_tables() 