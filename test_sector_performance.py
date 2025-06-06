import pandas as pd
import numpy as np
from trading_advisor.sector_performance import calculate_sector_performance, get_sp500_data
import matplotlib.pyplot as plt
import seaborn as sns

def test_sector_performance():
    # Create sample ticker data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    sectors = ['Technology', 'Technology', 'Technology', 'Consumer', 'Technology']
    
    # Create sample price data
    data = []
    for ticker, sector in zip(tickers, sectors):
        # Generate random walk prices
        prices = 100 * (1 + np.random.normal(0.001, 0.02, len(dates))).cumprod()
        volumes = np.random.randint(1000000, 10000000, len(dates))
        
        for date, price, volume in zip(dates, prices, volumes):
            data.append({
                'date': date,
                'ticker': ticker,
                'sector': sector,
                'close': price,
                'volume': volume
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df.set_index(['date', 'ticker'], inplace=True)
    
    # Ensure the index is properly formatted
    df.index = df.index.set_levels([pd.to_datetime(df.index.levels[0]), df.index.levels[1]])
    
    # Calculate sector performance
    sector_dfs = calculate_sector_performance(df, 'data/market_features')
    
    # Plot results for Technology sector
    tech_df = sector_dfs['Technology']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot price and relative strength
    ax1.plot(tech_df.index, tech_df['price'], label='Sector Price')
    ax1.set_title('Technology Sector Price')
    ax1.set_ylabel('Price')
    ax1.legend()
    
    # Plot relative strength metrics
    ax2.plot(tech_df.index, tech_df['relative_strength'], label='Relative Strength')
    ax2.plot(tech_df.index, tech_df['relative_strength_ratio'], label='Relative Strength Ratio')
    ax2.set_title('Relative Strength Metrics')
    ax2.set_ylabel('Value')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('sector_performance_test.png')
    plt.close()
    
    # Print summary statistics
    print("\nTechnology Sector Summary Statistics:")
    print(tech_df[['price', 'relative_strength', 'relative_strength_ratio']].describe())
    
    # Print correlation between metrics
    print("\nCorrelation between metrics:")
    print(tech_df[['price', 'relative_strength', 'relative_strength_ratio']].corr())

if __name__ == "__main__":
    test_sector_performance() 