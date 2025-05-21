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
    """Generate a concise, actionable markdown-formatted technical summary for a stock."""
    if df.empty:
        return f"## {ticker}\nNo data available.\n"
    
    latest = df.iloc[-1]
    prev_day = df.iloc[-2]
    
    # Calculate price and volume changes
    price_change = latest['Close'] - prev_day['Close']
    price_change_pct = (price_change / prev_day['Close']) * 100
    volume_change = latest['Volume'] - prev_day['Volume']
    volume_change_pct = (volume_change / prev_day['Volume']) * 100
    
    # --- Interpretations ---
    # RSI band
    rsi = latest['RSI']
    if rsi >= 70:
        rsi_band = "overbought"
        rsi_icon = "🧯"
    elif rsi <= 30:
        rsi_band = "oversold"
        rsi_icon = "📈"
    else:
        rsi_band = "neutral"
        rsi_icon = ""
    
    # MACD trend
    macd_hist = latest['MACD_Hist']
    macd_trend = "Bullish" if macd_hist > 0 else ("Bearish" if macd_hist < 0 else "Flat")
    macd_icon = "📈" if macd_trend == "Bullish" else ("❗" if macd_trend == "Bearish" else "")
    
    # Bollinger Band context
    price = latest['Close']
    bb_upper = latest['BB_Upper']
    bb_lower = latest['BB_Lower']
    bb_middle = latest['BB_Middle']
    if price > bb_upper:
        bb_context = f"Above upper band (${bb_upper:.2f})"
    elif price < bb_lower:
        bb_context = f"Below lower band (${bb_lower:.2f})"
    elif price > bb_middle:
        bb_context = f"Near upper band (${bb_upper:.2f})"
    elif price < bb_middle:
        bb_context = f"Near lower band (${bb_lower:.2f})"
    else:
        bb_context = "At middle band"
    
    # Volume context
    volume_context = f"{latest['Volume']/1e6:.1f}M"
    unusual_volume = False
    if abs(volume_change_pct) > 20:
        volume_context += f" ({volume_change_pct:+.1f}% vs prev day)"
        unusual_volume = True
    
    # Analyst target context
    analyst_line = ""
    upside = None
    analyst_range = ""
    if analyst_targets:
        current_price = analyst_targets['current_price']
        median_target = analyst_targets['median_target']
        low_target = analyst_targets.get('low_target')
        high_target = analyst_targets.get('high_target')
        upside = ((median_target - current_price) / current_price) * 100 if median_target and current_price else None
        analyst_line = f"Median analyst target: ${median_target:.0f}"
        if upside is not None:
            analyst_line += f" → {upside:+.1f}% upside"
        if low_target and high_target:
            analyst_range = f" (range: ${low_target:.0f}–${high_target:.0f})"
            analyst_line += analyst_range
    
    # MA trend
    ma_trend = "Bullish" if price > latest['SMA_20'] else ("Bearish" if price < latest['SMA_20'] else "Neutral")
    ma_icon = "📈" if ma_trend == "Bullish" else ("❗" if ma_trend == "Bearish" else "")
    
    # TL;DR summary
    tldr_icons = []
    if macd_trend == "Bullish" and upside and upside > 10:
        tldr_icons.append("📈")
    if rsi_band == "overbought":
        tldr_icons.append("🧯")
    if unusual_volume:
        tldr_icons.append("📈")
    if macd_trend == "Bearish" and position:
        tldr_icons.append("❗")
    tldr = f"💡 {ticker} {macd_trend.lower()} MACD, RSI {rsi_band}, {bb_context.lower()}"
    if upside and upside > 10:
        tldr += f", {upside:.0f}% upside to analyst target"
    if ma_trend == "Bullish":
        tldr += ", above 20d MA"
    elif ma_trend == "Bearish":
        tldr += ", below 20d MA"
    if unusual_volume:
        tldr += ", volume spike"
    if tldr_icons:
        tldr += " " + " ".join(tldr_icons)
    
    # --- Markdown summary ---
    summary = [f"\n📊 ${ticker} — {'Current Position' if position else 'New Technical Pick'}"]
    summary.append(f"\n{tldr}")
    
    # Position info
    if position:
        summary.append(f"Position: {position['quantity']:.0f} shares @ ${position['cost_basis']:.2f} | {position['gain_pct']:+.1f}% | {position['account_pct']:.1f}% of account")
    
    # Price and volume
    summary.append(f"Price: ${latest['Close']:.2f} (5d: {price_change_pct:+.1f}%)")
    if unusual_volume:
        summary.append(f"📈 Volume: {volume_change_pct:+.1f}% vs prev day — unusual activity")
    else:
        summary.append(f"Volume: {volume_context}")
    
    # Technicals
    summary.append(f"RSI: {rsi:.1f} — {rsi_band} {rsi_icon}")
    summary.append(f"MACD: {macd_trend} (Hist: {macd_hist:+.2f}) {macd_icon}")
    summary.append(f"BB: {bb_context}")
    summary.append(f"MA Trend: {ma_trend} (20d) {ma_icon}")
    if analyst_line:
        summary.append(analyst_line)
    summary.append(f"➡️ Technical Score: {score:.1f}/10")
    
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
    
    # Add improved Deep Research Prompt at the top
    report.append("""
---

**Prompt for LLM:**

You are a tactical swing trader managing a technical scan and open positions.
Your task is to return an actionable 1–2 week trading playbook for each stock listed below.
""")
    
    if positions:
        report.append("""
For each Current Position:
- Recommend Hold, Sell, or Adjust
- If "Adjust", provide a tactical move: e.g., raise stop, set trailing stop, scale out
- Include a recommended stop-loss level and optional profit-taking level
- Keep risk in mind—prioritize capital preservation if signals are weakening
""")
    
    report.append("""
For each New Technical Pick:
- Decide if it's a viable trade this week
- If yes, provide:
  - Entry strategy: buy now, wait for pullback, wait for breakout, etc.
  - Stop-loss: price level or % below
  - Profit target: based on analyst target, momentum, or resistance
  - Confidence level: High / Medium / Low

Assume:
- A 1–2 week swing trade horizon
- Technicals and analyst targets are the primary inputs
- The investor is risk-aware but willing to act on strong short-term setups

Be concise, tactical, and make clear, justified recommendations.

Focus most attention on the 💡 summary line. Use the structured data only to support or refine the thesis.

---
""")
    
    # Add current positions section
    if positions:
        report.append("\n## Current Positions")
        for ticker, score, summary in positions:
            report.append(summary)
            report.append("")  # Add extra newline after each position
    
    # Add new picks section
    if new_picks:
        report.append("\n## New Technical Picks")
        for ticker, score, summary in new_picks:
            report.append(summary)
            report.append("")  # Add extra newline after each pick
    
    # Save report to file if specified
    if output_path:
        output_path.write_text("\n".join(report))
    
    # Save structured data if specified
    if save_json:
        structured_data["timestamp"] = datetime.now().isoformat()
        save_json.write_text(json.dumps(structured_data, indent=2))
    
    return "\n".join(report) 