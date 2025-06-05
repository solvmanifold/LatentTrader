"""Main entry point for the dataset module."""

import argparse
import logging
from pathlib import Path
from .dataset import DatasetGenerator
import os
import sys

def main():
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(description='Generate machine learning datasets')
    parser.add_argument('--tickers', nargs='+', required=True, help='List of tickers to generate dataset for')
    parser.add_argument('--start-date', required=True, help='Start date for dataset')
    parser.add_argument('--end-date', required=True, help='End date for dataset')
    parser.add_argument('--target-days', type=int, default=5, help='Number of days to look ahead for target')
    parser.add_argument('--target-return', type=float, default=0.02, help='Target return threshold')
    parser.add_argument('--train-months', type=int, default=3, help='Number of months for training')
    parser.add_argument('--val-months', type=int, default=1, help='Number of months for validation')
    parser.add_argument('--test-months', type=int, default=1, help='Number of months for testing')
    parser.add_argument('--min-samples', type=int, default=10,
                      help='Minimum number of samples required')
    parser.add_argument('--log-level', type=str, default='WARNING',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level')
    parser.add_argument('--output', type=str, default='data/ml_datasets',
                      help='Directory to save output files')
    parser.add_argument('--force', action='store_true',
                      help='Overwrite existing files if they exist')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set root logger level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Set specific logger levels
    logging.getLogger('trading_advisor').setLevel(getattr(logging, args.log_level))
    
    # Parse tickers
    if args.tickers == ["all"]:
        ticker_list = load_tickers("all")
    elif len(args.tickers) == 1 and os.path.isfile(args.tickers[0]):
        with open(args.tickers[0]) as f:
            ticker_list = [line.strip() for line in f if line.strip()]
    else:
        ticker_list = args.tickers
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset generator
    generator = DatasetGenerator(
        market_features_dir='data/market_features',
        ticker_features_dir='data/ticker_features',
        output_dir=output_dir
    )
    
    # Generate dataset
    try:
        datasets = generator.generate_dataset(
            tickers=ticker_list,
            start_date=args.start_date,
            end_date=args.end_date,
            target_days=args.target_days,
            target_return=args.target_return,
            train_months=args.train_months,
            val_months=args.val_months,
            test_months=args.test_months,
            min_samples=args.min_samples,
            output=args.output,
            force=args.force
        )
        
        print(f"Successfully generated datasets with shapes:")
        print(f"Train: {datasets['train'].shape}")
        print(f"Validation: {datasets['val'].shape}")
        print(f"Test: {datasets['test'].shape}")
        
    except ValueError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error generating dataset: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main() 