"""Report generation and formatting functionality."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import SCORE_WEIGHTS

logger = logging.getLogger(__name__)

def generate_technical_summary(
    ticker: str,
    df: pd.DataFrame,
    score: float,
    score_details: Dict,
    analyst_targets: Optional[Dict] = None,
    position: Optional[Dict] = None
) -> str:
    """Generate a markdown-formatted technical summary for a stock."""
    if df.empty:
        return f"## {ticker}\nNo data available.\n"
    
    latest = df.iloc[-1]
    prev_day = df.iloc[-2]
    
    # Calculate price changes
    price_change = latest['Close'] - prev_day['Close']
    price_change_pct = (price_change / prev_day['Close']) * 100
    
    # Calculate volume changes
    volume_change = latest['Volume'] - prev_day['Volume']
    volume_change_pct = (volume_change / prev_day['Volume']) * 100
    
    # Start building the summary
    summary = [f"## {ticker}"]
    
    # Add position information if available
    if position:
        summary.append("\n### Current Position")
        summary.append(f"- Quantity: {position['quantity']:.2f}")
        summary.append(f"- Cost Basis: ${position['cost_basis']:.2f}")
        summary.append(f"- Market Value: ${position['market_value']:.2f}")
        summary.append(f"- Gain/Loss: {position['gain_pct']:.1f}%")
        summary.append(f"- Account %: {position['account_pct']:.1f}%")
    
    # Add price information
    summary.append("\n### Price Data")
    summary.append(f"- Current Price: ${latest['Close']:.2f}")
    summary.append(f"- 5-Day Change: {price_change_pct:.1f}%")
    summary.append(f"- Volume: {latest['Volume']:,.0f}")
    summary.append(f"- Volume Change: {volume_change_pct:.1f}%")
    
    # Add technical indicators
    summary.append("\n### Technical Indicators")
    summary.append(f"- RSI: {latest['RSI']:.1f}")
    summary.append(f"- MACD: {latest['MACD']:.2f}")
    summary.append(f"- MACD Signal: {latest['MACD_Signal']:.2f}")
    summary.append(f"- MACD Histogram: {latest['MACD_Hist']:.2f}")
    summary.append(f"- Bollinger Bands:")
    summary.append(f"  - Upper: ${latest['BB_Upper']:.2f}")
    summary.append(f"  - Middle: ${latest['BB_Middle']:.2f}")
    summary.append(f"  - Lower: ${latest['BB_Lower']:.2f}")
    summary.append(f"- Moving Averages:")
    summary.append(f"  - 20-day: ${latest['SMA_20']:.2f}")
    
    # Add analyst targets if available
    if analyst_targets:
        summary.append("\n### Analyst Targets")
        summary.append(f"- Current Price: ${analyst_targets['current_price']:.2f}")
        summary.append(f"- Median Target: ${analyst_targets['median_target']:.2f}")
        if analyst_targets.get('low_target'):
            summary.append(f"- Low Target: ${analyst_targets['low_target']:.2f}")
        if analyst_targets.get('high_target'):
            summary.append(f"- High Target: ${analyst_targets['high_target']:.2f}")
    
    # Add technical score
    summary.append(f"\n### Technical Score: {score:.1f}/10")
    summary.append("\nScore Breakdown:")
    for component, value in score_details.items():
        # Get the weight for this component from config
        weight = SCORE_WEIGHTS.get(component, 1.0)
        # Calculate normalized score (out of 10)
        normalized = (value / weight) * 10
        summary.append(f"- {component.replace('_', ' ').title()}: {value:.1f}/{weight:.1f} ({normalized:.1f}/10)")
    
    return "\n".join(summary)

def generate_structured_data(
    ticker: str,
    df: pd.DataFrame,
    score: float,
    score_details: Dict,
    analyst_targets: Optional[Dict] = None,
    position: Optional[Dict] = None
) -> Dict:
    """Generate structured data for a stock."""
    if df.empty:
        return {"ticker": ticker, "error": "No data available"}
    
    latest = df.iloc[-1]
    prev_day = df.iloc[-2]
    
    # Calculate price changes
    price_change = latest['Close'] - prev_day['Close']
    price_change_pct = (price_change / prev_day['Close']) * 100
    
    # Calculate volume changes
    volume_change = latest['Volume'] - prev_day['Volume']
    volume_change_pct = (volume_change / prev_day['Volume']) * 100
    
    data = {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "price_data": {
            "current_price": latest['Close'],
            "price_change": price_change,
            "price_change_pct": price_change_pct,
            "volume": latest['Volume'],
            "volume_change": volume_change,
            "volume_change_pct": volume_change_pct
        },
        "technical_indicators": {
            "rsi": latest['RSI'],
            "macd": {
                "value": latest['MACD'],
                "signal": latest['MACD_Signal'],
                "histogram": latest['MACD_Hist']
            },
            "bollinger_bands": {
                "upper": latest['BB_Upper'],
                "middle": latest['BB_Middle'],
                "lower": latest['BB_Lower']
            },
            "moving_averages": {
                "sma_20": latest['SMA_20']
            }
        },
        "score": {
            "total": score,
            "details": score_details
        }
    }
    
    if analyst_targets:
        data["analyst_targets"] = analyst_targets
    
    if position:
        data["position"] = position
    
    return data

def save_json_report(data: Dict, output_path: Path):
    """Save the structured data to a JSON file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving JSON report to {output_path}: {e}")

def generate_report(
    positions: List[Tuple[str, float, str]],
    new_picks: List[Tuple[str, float, str]],
    structured_data: Dict,
    output_path: Optional[Path] = None,
    save_json: Optional[Path] = None
) -> str:
    """Generate the complete report in markdown format."""
    report = ["# Weekly Trading Advisor Report"]
    report.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Add Deep Research Prompt at the top
    report.append("""
---

**Deep Research Prompt:**

You are an equity analyst. Here is the technical summary for several S&P 500 stocks.

The first section contains current positions. Please advise whether to hold, sell, or adjust (e.g. trailing stop).
The second section contains new picks flagged by our model this week. Please assess each for trade viability.

---
""")
    
    # Add current positions section
    if positions:
        report.append("\n## Current Positions")
        for ticker, score, summary in positions:
            report.append(summary)
    
    # Add new picks section
    if new_picks:
        report.append("\n## New Technical Picks")
        for ticker, score, summary in new_picks:
            report.append(summary)
    
    # Save the report
    if output_path:
        try:
            with open(output_path, 'w') as f:
                f.write("\n".join(report))
        except Exception as e:
            logger.error(f"Error saving report to {output_path}: {e}")
    
    # Save structured data if requested
    if save_json:
        save_json_report(structured_data, save_json)
    
    return "\n".join(report) 