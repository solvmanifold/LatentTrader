"""Technical analysis and scoring functionality."""

import logging
from typing import Dict, Optional, Tuple

import pandas as pd
import ta
import yfinance as yf

from .config import SCORE_WEIGHTS, MAX_RAW_SCORE

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
    
    # RSI Score (0-2)
    rsi = latest['RSI']
    if rsi < 30:
        score += 2  # Oversold
        score_details['rsi'] = 2
    elif rsi < 40:
        score += 1  # Approaching oversold
        score_details['rsi'] = 1
    elif rsi > 70:
        score += 0  # Overbought
        score_details['rsi'] = 0
    else:
        score += 0.5  # Neutral
        score_details['rsi'] = 0.5
    
    # Bollinger Bands Score (0-2)
    price = latest['Close']
    bb_lower = latest['BB_Lower']
    bb_upper = latest['BB_Upper']
    bb_middle = latest['BB_Middle']
    
    if price < bb_lower:
        score += 2  # Price below lower band
        score_details['bollinger'] = 2
    elif price < bb_middle:
        score += 1  # Price below middle band
        score_details['bollinger'] = 1
    elif price > bb_upper:
        score += 0  # Price above upper band
        score_details['bollinger'] = 0
    else:
        score += 0.5  # Price between middle and upper band
        score_details['bollinger'] = 0.5
    
    # MACD Score (0-2)
    macd = latest['MACD']
    macd_signal = latest['MACD_Signal']
    macd_hist = latest['MACD_Hist']
    
    if macd_hist > 0 and macd > macd_signal:
        score += 2  # Strong bullish signal
        score_details['macd'] = 2
    elif macd_hist > 0:
        score += 1  # Weak bullish signal
        score_details['macd'] = 1
    elif macd_hist < 0 and macd < macd_signal:
        score += 0  # Strong bearish signal
        score_details['macd'] = 0
    else:
        score += 0.5  # Weak bearish signal
        score_details['macd'] = 0.5
    
    # Moving Averages Score (0-2) - only using 20-day MA
    sma_20 = latest['SMA_20']
    
    if price > sma_20:
        score += 2  # Price above 20-day MA (bullish)
        score_details['moving_averages'] = 2
    elif price < sma_20:
        score += 0  # Price below 20-day MA (bearish)
        score_details['moving_averages'] = 0
    else:
        score += 1  # Price at 20-day MA (neutral)
        score_details['moving_averages'] = 1
    
    # Analyst Targets Score (0-4)
    if analyst_targets:
        current_price = analyst_targets['current_price']
        median_target = analyst_targets['median_target']
        low_target = analyst_targets.get('low_target')
        high_target = analyst_targets.get('high_target')
        
        if median_target:
            upside = ((median_target - current_price) / current_price) * 100
            
            if upside > 20:
                score += 4  # Strong upside potential
                score_details['analyst_targets'] = 4
            elif upside > 10:
                score += 3  # Moderate upside potential
                score_details['analyst_targets'] = 3
            elif upside > 0:
                score += 2  # Slight upside potential
                score_details['analyst_targets'] = 2
            else:
                score += 0  # No upside potential
                score_details['analyst_targets'] = 0
    
    # Normalize score to 0-10 range
    normalized_score = (score / MAX_RAW_SCORE) * 10
    
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