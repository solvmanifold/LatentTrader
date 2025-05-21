"""Technical analysis and scoring functionality."""

import logging
from typing import Dict, Optional, Tuple

import pandas as pd
import ta
import yfinance as yf

from config import SCORE_WEIGHTS, MAX_RAW_SCORE, MACD_STRONG_DIVERGENCE, MACD_WEAK_DIVERGENCE

logger = logging.getLogger(__name__)

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for the given DataFrame."""
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_Upper'] = bollinger.bollinger_hband()
    df['BB_Lower'] = bollinger.bollinger_lband()
    df['BB_Middle'] = bollinger.bollinger_mavg()
    df['BB_Pband'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Moving Averages (only 20-day for short-term trading)
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    
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

def calculate_score(df: pd.DataFrame, analyst_targets: Optional[Dict] = None) -> Tuple[float, Dict]:
    """Calculate a technical score based on various indicators."""
    if df.empty:
        return 0.0, {}
    
    latest = df.iloc[-1]
    score = 0.0
    score_details = {}
    
    # RSI Score
    rsi = df['RSI'].iloc[-3:].mean()
    if rsi < 30:
        score += min(SCORE_WEIGHTS['rsi_oversold'], 2.0)
        score_details['rsi'] = min(SCORE_WEIGHTS['rsi_oversold'], 2.0)
    elif rsi > 70:
        score += min(SCORE_WEIGHTS['rsi_overbought'], 2.0)
        score_details['rsi'] = min(SCORE_WEIGHTS['rsi_overbought'], 2.0)
    else:
        score_details['rsi'] = 0
    
    # Bollinger Bands Score
    bb_lower = df['BB_Lower'].iloc[-1]
    bb_upper = df['BB_Upper'].iloc[-1]
    bb_pband = df['BB_Pband'].iloc[-1]
    if bb_pband < 0.05:
        score += min(SCORE_WEIGHTS['bollinger'], 2.0)
        score_details['bollinger'] = min(SCORE_WEIGHTS['bollinger'], 2.0)
    elif bb_pband > 0.95:
        score -= min(SCORE_WEIGHTS['bollinger'], 2.0)
        score_details['bollinger'] = -min(SCORE_WEIGHTS['bollinger'], 2.0)
    else:
        score_details['bollinger'] = 0.0
    
    # MACD Score
    macd = df['MACD'].iloc[-3:].mean()
    macd_signal = df['MACD_Signal'].iloc[-3:].mean()
    macd_hist = df['MACD_Hist'].iloc[-3:].mean()
    
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
        score_details['macd'] = 0
    
    # Moving Averages Score
    sma_20 = latest['SMA_20']
    price = latest['Close']
    if price > sma_20 * 1.02:
        score += min(2.0, 2.0)
        score_details['moving_averages'] = min(2.0, 2.0)
    elif price < sma_20 * 0.98:
        score -= min(2.0, 2.0)
        score_details['moving_averages'] = -min(2.0, 2.0)
    elif price > sma_20:
        score += min(1.0, 2.0)
        score_details['moving_averages'] = min(1.0, 2.0)
    else:
        score_details['moving_averages'] = 0
    
    # Volume Spike Score
    prev_volume = df.iloc[-2]['Volume']
    if prev_volume == 0:
        volume_change = 0
    else:
        volume_change = (latest['Volume'] - prev_volume) / prev_volume * 100
    if abs(volume_change) > 20:
        score += min(SCORE_WEIGHTS['volume_spike'], 2.0)
        score_details['volume'] = min(SCORE_WEIGHTS['volume_spike'], 2.0)
    else:
        score_details['volume'] = 0
    
    # Analyst Targets Score
    if analyst_targets:
        current_price = analyst_targets['current_price']
        median_target = analyst_targets['median_target']
        
        if median_target:
            upside = ((median_target - current_price) / current_price) * 100
            weight = min(max(upside / 10, 0), 2.0)
            score += weight
            score_details['analyst_targets'] = weight
    
    # Normalize score to 0-10 range
    normalized_score = min(max((score / MAX_RAW_SCORE) * 10, 0), 10)
    
    # Ensure all tracked keys are present in score_details
    for key in ['macd', 'rsi', 'bollinger', 'moving_averages', 'volume', 'analyst_targets']:
        if key not in score_details:
            score_details[key] = 0
    
    return normalized_score, score_details

def analyze_stock(ticker: str, df: pd.DataFrame) -> Tuple[float, Dict, Optional[Dict]]:
    """Analyze a stock and return its score and details."""
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Get analyst targets
    analyst_targets = get_analyst_targets(ticker)
    
    # Calculate score
    score, score_details = calculate_score(df, analyst_targets)
    
    return score, score_details, analyst_targets 