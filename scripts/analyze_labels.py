"""Analyze label distributions across all splits."""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_split(df: pd.DataFrame, split_name: str) -> None:
    """Analyze label distribution for a single split.
    
    Args:
        df: DataFrame containing the split data
        split_name: Name of the split (e.g., 'train', 'val', 'test')
    """
    labels = df['label'].values
    unique_labels = np.unique(labels)
    
    logger.info(f"\n{split_name.upper()} Split Analysis:")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Unique label values: {unique_labels}")
    logger.info(f"Label distribution: {np.bincount(labels.astype(int))}")
    logger.info(f"Positive label percentage: {100 * np.mean(labels):.2f}%")
    
    # Check for non-binary values
    non_binary = labels[~np.isin(labels, [0, 1])]
    if len(non_binary) > 0:
        logger.warning(f"Found {len(non_binary)} non-binary values: {non_binary}")
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='label', bins=50)
    plt.title(f'Label Distribution - {split_name.capitalize()} Split')
    plt.xlabel('Label Value')
    plt.ylabel('Count')
    plt.savefig(f'analysis/label_dist_{split_name}.png')
    plt.close()

def main():
    # Create analysis directory if it doesn't exist
    Path('analysis').mkdir(exist_ok=True)
    
    # Load splits
    splits = {
        'train': pd.read_parquet('data/ml_datasets/split_1/train.parquet'),
        'val': pd.read_parquet('data/ml_datasets/split_1/val.parquet'),
        'test': pd.read_parquet('data/ml_datasets/split_1/test.parquet')
    }
    
    # Analyze each split
    for split_name, df in splits.items():
        analyze_split(df, split_name)
    
    # Compare positive label percentages across splits
    percentages = {
        split_name: 100 * np.mean(df['label'].values)
        for split_name, df in splits.items()
    }
    
    logger.info("\nPositive Label Percentages Across Splits:")
    for split_name, pct in percentages.items():
        logger.info(f"{split_name}: {pct:.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.bar(percentages.keys(), percentages.values())
    plt.title('Positive Label Percentages Across Splits')
    plt.ylabel('Percentage')
    plt.ylim(0, 100)
    plt.savefig('analysis/label_percentages.png')
    plt.close()

if __name__ == '__main__':
    main() 