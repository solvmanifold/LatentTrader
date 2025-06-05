"""Check label values in the dataset."""

import pandas as pd
import numpy as np

def check_labels(df: pd.DataFrame, split_name: str):
    """Check label values in a DataFrame.
    
    Args:
        df: DataFrame containing the data
        split_name: Name of the split (e.g., 'train', 'val', 'test')
    """
    print(f"\n{split_name.upper()} Split Labels:")
    print("Unique values:", np.unique(df['label'].values))
    print("Value counts:")
    print(df['label'].value_counts().sort_index())
    print("\nSample of non-binary values (if any):")
    non_binary = df[~df['label'].isin([0, 1])]
    if len(non_binary) > 0:
        print(non_binary[['date', 'ticker', 'label']].head())

def main():
    # Load splits
    splits = {
        'train': pd.read_parquet('data/ml_datasets/split_1/train.parquet'),
        'val': pd.read_parquet('data/ml_datasets/split_1/val.parquet'),
        'test': pd.read_parquet('data/ml_datasets/split_1/test.parquet')
    }
    
    # Check each split
    for split_name, df in splits.items():
        check_labels(df, split_name)

if __name__ == '__main__':
    main() 