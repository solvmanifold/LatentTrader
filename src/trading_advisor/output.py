"""Report generation and formatting functionality."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from trading_advisor.config import SCORE_WEIGHTS

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

def generate_report(structured_data):
    """Generate a markdown report from structured data."""
    report = []
    
    # Add timestamp
    timestamp = datetime.fromisoformat(structured_data['timestamp'])
    report.append(f"# Trading Advisor Report")
    report.append(f"Generated on: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Add current positions
    if structured_data['positions']:
        report.append("\n## Current Positions")
        for position in structured_data['positions']:
            entry = [f"### {position['ticker']}",
                     f"**Technical Score:** {position['score']['total']:.1f}/10"]
            # Add position details
            if position['position']:
                entry.append(f"**Position Details:**")
                entry.append(f"- Quantity: {position['position']['quantity']}")
                if 'price' in position['position']:
                    entry.append(f"- Entry Price: ${position['position']['price']:.2f}")
                if 'market_value' in position['position']:
                    entry.append(f"- Market Value: ${position['position']['market_value']:.2f}")
                if 'gain_pct' in position['position']:
                    entry.append(f"- Gain/Loss: {position['position']['gain_pct']:.1f}%")
            # Add technical indicators
            entry.append(f"**Technical Indicators:**")
            for indicator, value in position['technical_indicators'].items():
                if isinstance(value, dict):
                    for sub_indicator, sub_value in value.items():
                        entry.append(f"- {indicator.upper()} {sub_indicator}: {sub_value:.2f}")
                else:
                    entry.append(f"- {indicator.upper()}: {value:.2f}")
            # Add stop-loss level if we have price data
            if isinstance(position['price_data'], list) and position['price_data']:
                current_price = position['price_data'][-1]['Close']
            elif isinstance(position['price_data'], dict) and 'current_price' in position['price_data']:
                current_price = position['price_data']['current_price']
            else:
                current_price = None
            if current_price is not None:
                entry.append(f"🛑 Stop-loss level: ${current_price * 0.97:.2f} (3% below current)")
            # Add analyst targets if available
            if position['analyst_targets']:
                entry.append(f"**Analyst Targets:**")
                entry.append(f"- Median: ${position['analyst_targets']['median_target']:.2f}")
                entry.append(f"- Range: ${position['analyst_targets']['low_target']:.2f} - ${position['analyst_targets']['high_target']:.2f}")
            report.append("\n".join(entry))
    
    # Add new picks
    if structured_data['new_picks']:
        report.append("\n## New Trading Opportunities")
        # Sort new picks by score
        sorted_picks = sorted(structured_data['new_picks'], key=lambda x: x['score']['total'], reverse=True)
        # Show top 2 setups
        report.append("### Top 2 Setups by Confidence x Upside")
        for pick in sorted_picks[:2]:
            entry = [f"#### {pick['ticker']}",
                     f"**Technical Score:** {pick['score']['total']:.1f}/10",
                     f"**Technical Indicators:**"]
            for indicator, value in pick['technical_indicators'].items():
                if isinstance(value, dict):
                    for sub_indicator, sub_value in value.items():
                        entry.append(f"- {indicator.upper()} {sub_indicator}: {sub_value:.2f}")
                else:
                    entry.append(f"- {indicator.upper()}: {value:.2f}")
            if pick['analyst_targets']:
                entry.append(f"**Analyst Targets:**")
                entry.append(f"- Median: ${pick['analyst_targets']['median_target']:.2f}")
                entry.append(f"- Range: ${pick['analyst_targets']['low_target']:.2f} - ${pick['analyst_targets']['high_target']:.2f}")
            report.append("\n".join(entry))
    
    return "\n\n".join(report)

def get_indicator(technical, key):
    return technical.get(key) or technical.get(key.upper()) or technical.get(key.lower())

def generate_research_prompt(structured_data: Dict) -> str:
    """Generate a ChatGPT-ready prompt for deep research analysis, using a tactical swing trading playbook format."""
    prompt = []
    # Header instructions
    prompt.append("You are a tactical swing trader managing a technical scan and open positions.\n"
                  "Your task is to return an actionable 1–2 week trading playbook for each stock listed below.\n\n"
                  "Return all responses in this exact bullet format for each stock:\n\n"
                  "✅ Action (e.g. Buy Now, Hold, Adjust)\n"
                  "🎯 Entry strategy (limit or breakout entry, price conditions, timing)\n"
                  "🛑 Stop-loss level (specific price or %)\n"
                  "💰 Profit-taking strategy (target price, resistance level, or trailing stop)\n"
                  "🔍 Confidence level (High / Medium / Low)\n"
                  "🧠 Rationale (1–2 lines)\n\n"
                  "If a setup is weak or ambiguous, say 'No trade this week' and explain why.\n\n"
                  "Assume:\n"
                  "- A 1–2 week swing trade horizon\n"
                  "- Technicals and analyst targets are the primary inputs\n"
                  "- The investor is risk-aware but willing to act on strong short-term setups\n\n"
                  "Be concise, tactical, and make clear, justified recommendations.\n\n"
                  "---\n\n")

    def generate_summary(ticker: str, technical: Dict, price_data: Dict, analyst: Optional[Dict] = None) -> str:
        """Generate a concise summary of technical indicators and analyst targets."""
        summary = [f"💡 {ticker}:"]
        
        # MACD trend
        macd = get_indicator(technical, 'macd')
        macd_trend = "Bullish" if macd['histogram'] > 0 else ("Bearish" if macd['histogram'] < 0 else "Flat")
        summary.append(f"{macd_trend} MACD")
        
        # RSI band
        rsi = get_indicator(technical, 'rsi')
        rsi_band = "overbought" if rsi >= 70 else ("oversold" if rsi <= 30 else "neutral")
        summary.append(f"RSI {rsi_band}")
        
        # Bollinger Band context
        bb = get_indicator(technical, 'bollinger_bands')
        price = price_data['current_price']
        if price > bb['upper']:
            bb_context = f"above upper band (${bb['upper']:.2f})"
        elif price < bb['lower']:
            bb_context = f"below lower band (${bb['lower']:.2f})"
        elif price > bb['middle']:
            bb_context = f"near upper band (${bb['upper']:.2f})"
        else:
            bb_context = f"near lower band (${bb['lower']:.2f})"
        summary.append(bb_context)
        
        # Analyst target
        if analyst:
            upside = ((analyst['median_target'] - analyst['current_price']) / analyst['current_price']) * 100
            if upside > 0:
                summary.append(f"{upside:.0f}% upside to analyst target")
        
        # MA trend
        ma = get_indicator(technical, 'moving_averages')
        ma_trend = "above" if price > ma['sma_20'] else ("below" if price < ma['sma_20'] else "at")
        summary.append(f"{ma_trend} 20d MA")
        
        # Volume spike
        if abs(price_data['volume_change_pct']) > 20:
            summary.append("volume spike")
        
        # Add icons
        icons = []
        if macd_trend == "Bullish" and analyst and upside > 10:
            icons.append("📈")
        if rsi_band == "overbought":
            icons.append("🧯")
        if abs(price_data['volume_change_pct']) > 20:
            icons.append("📈")
        if macd_trend == "Bearish":
            icons.append("❗")
        if icons:
            summary.append(" ".join(icons))
        
        return ", ".join(summary)

    # Current Positions
    if structured_data['positions']:
        prompt.append("## Current Positions\n")
        for position in structured_data['positions']:
            ticker = position['ticker']
            score = position['score']['total']
            price_data = position['price_data']
            technical = position['technical_indicators']
            analyst = position.get('analyst_targets')
            pos = position.get('position')
            
            # Position header
            prompt.append(f"📊 ${ticker} — Current Position\n")
            
            # Generate summary and first data line
            summary = generate_summary(ticker, technical, price_data, analyst)
            if pos:
                prompt.append(f"{summary} Position: {pos['quantity']:.0f} shares @ ${pos['cost_basis']:.2f} | {pos['gain_pct']:+.1f}% | {pos['account_pct']:.1f}% of account")
            else:
                prompt.append(f"{summary} Price: ${price_data['current_price']:.2f} (5d: {price_data['price_change_pct']:+.1f}%)")
            
            # Price and volume (if not already added)
            if pos:
                prompt.append(f"Price: ${price_data['current_price']:.2f} (5d: {price_data['price_change_pct']:+.1f}%)")
            if abs(price_data['volume_change_pct']) > 20:
                prompt.append(f"📈 Volume: {price_data['volume_change_pct']:+.1f}% vs prev day — unusual activity")
            else:
                prompt.append(f"Volume: {price_data['volume']/1e6:.1f}M")
            
            # Technical indicators
            rsi = get_indicator(technical, 'rsi')
            rsi_band = "overbought" if rsi >= 70 else ("oversold" if rsi <= 30 else "neutral")
            prompt.append(f"RSI: {rsi:.1f} — {rsi_band}")
            
            macd = get_indicator(technical, 'macd')
            macd_trend = "Bullish" if macd['histogram'] > 0 else ("Bearish" if macd['histogram'] < 0 else "Flat")
            prompt.append(f"MACD: {macd_trend} (Hist: {macd['histogram']:+.2f}) 📈" if macd_trend == "Bullish" else f"MACD: {macd_trend} (Hist: {macd['histogram']:+.2f})")
            
            bb = get_indicator(technical, 'bollinger_bands')
            price = price_data['current_price']
            if price > bb['upper']:
                bb_context = f"Above upper band (${bb['upper']:.2f})"
            elif price < bb['lower']:
                bb_context = f"Below lower band (${bb['lower']:.2f})"
            elif price > bb['middle']:
                bb_context = f"Near upper band (${bb['upper']:.2f})"
            else:
                bb_context = f"Near lower band (${bb['lower']:.2f})"
            prompt.append(f"BB: {bb_context}")
            
            ma = get_indicator(technical, 'moving_averages')
            ma_trend = "Bullish" if price > ma['sma_20'] else ("Bearish" if price < ma['sma_20'] else "Neutral")
            prompt.append(f"MA Trend: {ma_trend} (20d) 📈" if ma_trend == "Bullish" else f"MA Trend: {ma_trend} (20d)")
            
            # Analyst targets
            if analyst:
                upside = ((analyst['median_target'] - analyst['current_price']) / analyst['current_price']) * 100
                prompt.append(f"Median analyst target: ${analyst['median_target']:.0f} → {upside:+.1f}% upside (range: ${analyst['low_target']:.0f}–${analyst['high_target']:.0f})")
            
            # Technical score
            prompt.append(f"➡️ Technical Score: {score:.1f}/10\n")

    # New Technical Picks
    if structured_data['new_picks']:
        prompt.append("## New Technical Picks\n")
        sorted_picks = sorted(structured_data['new_picks'], key=lambda x: x['score']['total'], reverse=True)
        for pick in sorted_picks:
            ticker = pick['ticker']
            score = pick['score']['total']
            price_data = pick['price_data']
            technical = pick['technical_indicators']
            analyst = pick.get('analyst_targets')
            
            # Pick header
            prompt.append(f"📊 ${ticker} — New Technical Pick\n")
            
            # Generate summary and first data line
            summary = generate_summary(ticker, technical, price_data, analyst)
            prompt.append(f"{summary} Price: ${price_data['current_price']:.2f} (5d: {price_data['price_change_pct']:+.1f}%)")
            
            # Volume
            if abs(price_data['volume_change_pct']) > 20:
                prompt.append(f"📈 Volume: {price_data['volume_change_pct']:+.1f}% vs prev day — unusual activity")
            else:
                prompt.append(f"Volume: {price_data['volume']/1e6:.1f}M")
            
            # Technical indicators
            rsi = get_indicator(technical, 'rsi')
            rsi_band = "overbought" if rsi >= 70 else ("oversold" if rsi <= 30 else "neutral")
            prompt.append(f"RSI: {rsi:.1f} — {rsi_band}")
            
            macd = get_indicator(technical, 'macd')
            macd_trend = "Bullish" if macd['histogram'] > 0 else ("Bearish" if macd['histogram'] < 0 else "Flat")
            prompt.append(f"MACD: {macd_trend} (Hist: {macd['histogram']:+.2f}) 📈" if macd_trend == "Bullish" else f"MACD: {macd_trend} (Hist: {macd['histogram']:+.2f})")
            
            bb = get_indicator(technical, 'bollinger_bands')
            price = price_data['current_price']
            if price > bb['upper']:
                bb_context = f"Above upper band (${bb['upper']:.2f})"
            elif price < bb['lower']:
                bb_context = f"Below lower band (${bb['lower']:.2f})"
            elif price > bb['middle']:
                bb_context = f"Near upper band (${bb['upper']:.2f})"
            else:
                bb_context = f"Near lower band (${bb['lower']:.2f})"
            prompt.append(f"BB: {bb_context}")
            
            ma = get_indicator(technical, 'moving_averages')
            ma_trend = "Bullish" if price > ma['sma_20'] else ("Bearish" if price < ma['sma_20'] else "Neutral")
            prompt.append(f"MA Trend: {ma_trend} (20d) 📈" if ma_trend == "Bullish" else f"MA Trend: {ma_trend} (20d)")
            
            # Analyst targets
            if analyst:
                upside = ((analyst['median_target'] - analyst['current_price']) / analyst['current_price']) * 100
                prompt.append(f"Median analyst target: ${analyst['median_target']:.0f} → {upside:+.1f}% upside (range: ${analyst['low_target']:.0f}–${analyst['high_target']:.0f})")
            
            # Technical score
            prompt.append(f"➡️ Technical Score: {score:.1f}/10\n")

    # Final instruction
    prompt.append("---\n\nAfter reviewing all positions and new picks above, please rank the top 2 setups by confidence × upside and briefly explain your reasoning.")

    return "\n".join(prompt)

def generate_deep_research_prompt(structured_data: Dict) -> str:
    """Generate a deep research prompt for tactical swing trading analysis."""
    prompt = []
    
    # Add instructions
    prompt.append("You are a tactical swing trader and market strategist evaluating a technical scan and open positions. Your task is to develop an actionable 1–2 week trading playbook for each stock listed below.")
    prompt.append("\nIn addition to the provided technical summaries and analyst targets, use current market context, news events, earnings calendars, and public sentiment (e.g. Reddit, Twitter, financial media) to support or override your recommendations.")
    prompt.append("\nReturn your response in this exact bullet format for each stock:")
    prompt.append("\n✅ Action (e.g. Buy Now, Hold, Adjust)  \n🎯 Entry strategy (limit or breakout entry, price conditions, timing)  \n🛑 Stop-loss level (specific price or %)  \n💰 Profit-taking strategy (target price, resistance level, or trailing stop)  \n🔍 Confidence level (High / Medium / Low)  \n🧠 Rationale (1–2 lines)")
    prompt.append("\nBegin each ticker with a 💡 summary that integrates both technical and real-time context.")
    prompt.append("\nIf a setup is weak or conflicting, say 'No trade this week' and explain why.")
    prompt.append("\nAssume:")
    prompt.append("- A 1–2 week swing trade horizon")
    prompt.append("- Technicals are important, but can be overridden by breaking news, macro conditions, or earnings catalysts")
    prompt.append("- The investor is risk-aware but ready to act decisively on high-conviction short-term setups")
    prompt.append("\nPrioritize quality over quantity. Be specific and tactical in your recommendations.")
    prompt.append("\n---\n")
    
    # Add current positions
    if structured_data['positions']:
        prompt.append("\n📊 Current Positions:")
        for position in structured_data['positions']:
            # Create a DataFrame with both current and previous day's data
            tech_data = {
                'RSI': [position['technical_indicators']['rsi'], position['technical_indicators']['rsi']],
                'MACD': [position['technical_indicators']['macd']['value'], position['technical_indicators']['macd']['value']],
                'MACD_Signal': [position['technical_indicators']['macd']['signal'], position['technical_indicators']['macd']['signal']],
                'MACD_Hist': [position['technical_indicators']['macd']['histogram'], position['technical_indicators']['macd']['histogram']],
                'BB_Upper': [position['technical_indicators']['bollinger_bands']['upper'], position['technical_indicators']['bollinger_bands']['upper']],
                'BB_Lower': [position['technical_indicators']['bollinger_bands']['lower'], position['technical_indicators']['bollinger_bands']['lower']],
                'BB_Middle': [position['technical_indicators']['bollinger_bands']['middle'], position['technical_indicators']['bollinger_bands']['middle']],
                'SMA_20': [position['technical_indicators']['moving_averages']['sma_20'], position['technical_indicators']['moving_averages']['sma_20']],
                'Close': [position['price_data']['current_price'], position['price_data']['current_price'] * 0.99],  # Simulate previous day's price
                'Volume': [position['price_data']['volume'], position['price_data']['volume'] * 0.95]  # Simulate previous day's volume
            }
            df = pd.DataFrame(tech_data)
            
            prompt.append(generate_technical_summary(
                position['ticker'],
                df,
                position['score']['total'],
                position['score']['details'],
                position.get('analyst_targets'),
                position.get('position')
            ))
    
    # Add new picks
    if structured_data['new_picks']:
        prompt.append("\n📊 New Technical Picks:")
        for pick in structured_data['new_picks']:
            # Create a DataFrame with both current and previous day's data
            tech_data = {
                'RSI': [pick['technical_indicators']['rsi'], pick['technical_indicators']['rsi']],
                'MACD': [pick['technical_indicators']['macd']['value'], pick['technical_indicators']['macd']['value']],
                'MACD_Signal': [pick['technical_indicators']['macd']['signal'], pick['technical_indicators']['macd']['signal']],
                'MACD_Hist': [pick['technical_indicators']['macd']['histogram'], pick['technical_indicators']['macd']['histogram']],
                'BB_Upper': [pick['technical_indicators']['bollinger_bands']['upper'], pick['technical_indicators']['bollinger_bands']['upper']],
                'BB_Lower': [pick['technical_indicators']['bollinger_bands']['lower'], pick['technical_indicators']['bollinger_bands']['lower']],
                'BB_Middle': [pick['technical_indicators']['bollinger_bands']['middle'], pick['technical_indicators']['bollinger_bands']['middle']],
                'SMA_20': [pick['technical_indicators']['moving_averages']['sma_20'], pick['technical_indicators']['moving_averages']['sma_20']],
                'Close': [pick['price_data']['current_price'], pick['price_data']['current_price'] * 0.99],  # Simulate previous day's price
                'Volume': [pick['price_data']['volume'], pick['price_data']['volume'] * 0.95]  # Simulate previous day's volume
            }
            df = pd.DataFrame(tech_data)
            
            prompt.append(generate_technical_summary(
                pick['ticker'],
                df,
                pick['score']['total'],
                pick['score']['details'],
                pick.get('analyst_targets')
            ))
    
    return "\n".join(prompt) 