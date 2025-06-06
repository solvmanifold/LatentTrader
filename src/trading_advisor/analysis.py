"""Technical analysis and scoring functionality."""

import logging
from typing import Dict, Optional, Tuple

import pandas as pd
import ta
import yfinance as yf
import json

from trading_advisor.config import SCORE_WEIGHTS, MAX_RAW_SCORE, MACD_STRONG_DIVERGENCE, MACD_WEAK_DIVERGENCE

logger = logging.getLogger(__name__)

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for the given DataFrame."""
    if len(df) < 50:
        raise ValueError("Insufficient data points for calculating technical indicators. Need at least 50 days of data.")

    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'])
    df['bb_upper'] = bollinger.bollinger_hband()
    df['bb_lower'] = bollinger.bollinger_lband()
    df['bb_middle'] = bollinger.bollinger_mavg()
    df['bb_pband'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Moving Averages
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['sma_100'] = ta.trend.sma_indicator(df['close'], window=100)
    df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
    df['ema_100'] = ta.trend.ema_indicator(df['close'], window=100)
    df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
    
    return df

def get_analyst_targets(ticker: str) -> Optional[Dict]:
    """Get analyst price targets for a ticker."""
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        
        if 'targetMeanPrice' in info and info['targetMeanPrice'] is not None:
            return {
                'current_price': info.get('currentPrice', info.get('regularMarketPrice')),
                'median_target': info['targetMeanPrice'],
                'low_target': info.get('targetLowPrice'),
                'high_target': info.get('targetHighPrice')
            }
    except Exception as e:
        logger.error(f"Error getting analyst targets for {ticker}: {e}")
    
    return None

def calculate_score(df_or_row, analyst_targets=None, window=3):
    """
    Calculate a technical score for a DataFrame (using last `window` rows for smoothing)
    or for a single row (Series).
    Returns (score, score_details).
    """
    def score_row(row, analyst_targets=None):
        score = 0.0
        score_details = {}
        
        # RSI Score
        rsi = row.get('rsi', float('nan'))
        if pd.notna(rsi):
            if rsi < 30:
                score += min(SCORE_WEIGHTS['rsi_oversold'], 2.0)
                score_details['rsi'] = min(SCORE_WEIGHTS['rsi_oversold'], 2.0)
            elif rsi > 70:
                score += min(SCORE_WEIGHTS['rsi_overbought'], 2.0)
                score_details['rsi'] = min(SCORE_WEIGHTS['rsi_overbought'], 2.0)
            else:
                score_details['rsi'] = 0.0
        else:
            score_details['rsi'] = 0.0
            
        # Bollinger Bands Score
        bb_lower = row.get('bb_lower', float('nan'))
        bb_upper = row.get('bb_upper', float('nan'))
        bb_pband = row.get('bb_pband', float('nan'))
        if pd.notna(bb_pband):
            if bb_pband < 0.05:
                weight = SCORE_WEIGHTS['bollinger_low']
                score += weight
                score_details['bollinger'] = weight
            elif bb_pband > 0.95:
                weight = SCORE_WEIGHTS['bollinger_high']
                score += weight
                score_details['bollinger'] = weight
            else:
                score_details['bollinger'] = 0.0
        else:
            score_details['bollinger'] = 0.0
            
        # MACD Score
        macd = row.get('macd', float('nan'))
        macd_signal = row.get('macd_signal', float('nan'))
        macd_hist = row.get('macd_hist', float('nan'))
        if pd.notna(macd_hist) and pd.notna(macd) and pd.notna(macd_signal):
            if macd_hist > MACD_STRONG_DIVERGENCE and macd > macd_signal:
                score += min(SCORE_WEIGHTS['macd_strong_divergence'], 2.0)
                score_details['macd'] = min(SCORE_WEIGHTS['macd_strong_divergence'], 2.0)
            elif macd_hist > MACD_WEAK_DIVERGENCE and macd > macd_signal:
                score += min(SCORE_WEIGHTS['macd_moderate_divergence'], 2.0)
                score_details['macd'] = min(SCORE_WEIGHTS['macd_moderate_divergence'], 2.0)
            elif macd_hist < -MACD_STRONG_DIVERGENCE and macd < macd_signal:
                score += min(SCORE_WEIGHTS['macd_crossover'], 2.0)
                score_details['macd'] = min(SCORE_WEIGHTS['macd_crossover'], 2.0)
            else:
                score_details['macd'] = 0.0
        else:
            score_details['macd'] = 0.0
            
        # Moving Averages Score
        sma_20 = row.get('sma_20', float('nan'))
        sma_50 = row.get('sma_50', float('nan'))
        price = row.get('close', float('nan'))
        if pd.notna(price) and pd.notna(sma_20):
            if price > sma_20 * 1.02:
                weight = SCORE_WEIGHTS.get('sma_strong_above', 2.0)
                score += weight
                score_details['moving_averages'] = weight
            elif price < sma_20 * 0.98:
                weight = SCORE_WEIGHTS.get('sma_strong_below', 2.0)
                score -= weight
                score_details['moving_averages'] = -weight
            elif price > sma_20:
                weight = SCORE_WEIGHTS.get('sma_above', 1.0)
                score += weight
                score_details['moving_averages'] = weight
            elif pd.notna(sma_50) and price > sma_50:
                weight = SCORE_WEIGHTS.get('sma_above_50', 1.0)
                score += weight
                score_details['moving_averages'] = weight
            else:
                score_details['moving_averages'] = 0.0
        else:
            score_details['moving_averages'] = 0.0
            
        # Volume Spike Score
        prev_volume = row.get('Prev_Volume', float('nan'))
        volume = row.get('Volume', float('nan'))
        if pd.notna(prev_volume) and prev_volume != 0 and pd.notna(volume):
            volume_change = (volume - prev_volume) / prev_volume * 100
            if abs(volume_change) > 20:
                score += min(SCORE_WEIGHTS['volume_spike'], 2.0)
                score_details['volume'] = min(SCORE_WEIGHTS['volume_spike'], 2.0)
            else:
                score_details['volume'] = 0.0
        else:
            score_details['volume'] = 0.0
            
        # Analyst Targets Score (optional, only for latest row in backtest)
        if analyst_targets and hasattr(analyst_targets, 'get'):
            current_price = analyst_targets.get('current_price', None)
            median_target = analyst_targets.get('median_target', None)
            if median_target and current_price:
                upside = ((median_target - current_price) / current_price) * 100
                weight = min(max(upside / 10, 0), 2.0)
                score += weight
                score_details['analyst_targets'] = weight
            else:
                score_details['analyst_targets'] = 0.0
        else:
            score_details['analyst_targets'] = 0.0
            
        # Normalize score to 0-10 range
        normalized_score = float(min(max((score / MAX_RAW_SCORE) * 10, 0), 10))
        
        # Ensure all tracked keys are present in score_details
        for key in ['macd', 'rsi', 'bollinger', 'moving_averages', 'volume', 'analyst_targets']:
            if key not in score_details:
                score_details[key] = 0.0
                
        return normalized_score, score_details

    if isinstance(df_or_row, pd.Series):
        return score_row(df_or_row, analyst_targets)
    elif isinstance(df_or_row, pd.DataFrame):
        if df_or_row.empty:
            return 0.0, {}
        window_df = df_or_row.iloc[-window:]
        row = window_df.mean(numeric_only=True)
        # For non-numeric columns (e.g., 'close'), take the last value
        for col in ['close', 'Volume', 'sma_20', 'sma_50', 'bb_lower', 'bb_upper', 'bb_pband']:
            if col in window_df.columns:
                row[col] = window_df[col].iloc[-1]
        return score_row(row, analyst_targets)
    else:
        raise ValueError("Input must be a pandas DataFrame or Series")

def analyze_stock(ticker: str, df: pd.DataFrame) -> Tuple[float, Dict, Optional[Dict]]:
    """Analyze a stock and return its score and details."""
    # No need to calculate technical indicators here; already done in download_stock_data
    # Get analyst targets
    analyst_targets = get_analyst_targets(ticker)
    # Calculate score
    score, score_details = calculate_score(df, analyst_targets)
    return score, score_details, analyst_targets 