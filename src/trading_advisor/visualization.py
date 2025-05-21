"""Visualization module for generating stock charts and summaries."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any, Optional
import os

def create_stock_chart(
    df: pd.DataFrame,
    ticker: str,
    indicators: Dict[str, Any],
    output_dir: str = "output/charts"
) -> str:
    """
    Create an interactive HTML chart for a stock with price and indicators.
    
    Args:
        df: DataFrame with OHLCV data
        ticker: Stock ticker symbol
        indicators: Dictionary containing technical indicators
        output_dir: Directory to save the chart
        
    Returns:
        Path to the saved HTML file
    """
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{ticker} Price", "Volume", "RSI")
    )

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )

    # Add Bollinger Bands
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Upper'],
                name='BB Upper',
                line=dict(color='rgba(250, 0, 0, 0.3)')
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Lower'],
                name='BB Lower',
                line=dict(color='rgba(0, 250, 0, 0.3)'),
                fill='tonexty'
            ),
            row=1, col=1
        )
    # Add middle Bollinger Band
    if 'BB_Middle' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Middle'],
                name='BB Middle',
                line=dict(color='black', dash='dash')
            ),
            row=1, col=1
        )

    # Add moving averages
    if 'MA20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MA20'],
                name='20-day MA',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
    if 'MA50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MA50'],
                name='50-day MA',
                line=dict(color='orange')
            ),
            row=1, col=1
        )

    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color='rgba(0, 0, 255, 0.3)'
        ),
        row=2, col=1
    )

    # Add RSI
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                name='RSI',
                line=dict(color='purple')
            ),
            row=3, col=1
        )
        # Add RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # Update layout
    fig.update_layout(
        title=f"{ticker} Technical Analysis",
        xaxis_rangeslider_visible=False,
        height=800,
        template='plotly_white'
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the chart
    output_path = os.path.join(output_dir, f"{ticker}_chart.html")
    fig.write_html(output_path)
    
    return output_path

def create_score_breakdown(
    ticker: str,
    score: float,
    indicators: Dict[str, Any],
    output_dir: str = "output/charts"
) -> str:
    """
    Create a comprehensive visualization of the technical score breakdown.
    
    Args:
        ticker: Stock ticker symbol
        score: Total technical score
        indicators: Dictionary containing technical indicators and their contributions
        output_dir: Directory to save the chart
        
    Returns:
        Path to the saved HTML file
    """
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "indicator"}, {"type": "bar"}],
               [{"type": "table", "colspan": 2}, None]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
        subplot_titles=(
            "Overall Score",
            "Component Breakdown",
            "Detailed Analysis"
        )
    )
    
    # Add gauge chart for total score
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Technical Score"},
            gauge={
                'axis': {'range': [0, 10]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 3], 'color': "lightgray"},
                    {'range': [3, 7], 'color': "gray"},
                    {'range': [7, 10], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 7
                }
            }
        ),
        row=1, col=1
    )
    
    # Extract and prepare score components
    score_components = {
        'RSI': indicators.get('rsi_score', 0),
        'Bollinger Bands': indicators.get('bb_score', 0),
        'MACD': indicators.get('macd_score', 0),
        'Moving Averages': indicators.get('ma_score', 0),
        'Volume': indicators.get('volume_score', 0),
        'Analyst Targets': indicators.get('analyst_targets_score', 0),
    }
    
    # Add horizontal bar chart for component breakdown
    def get_bar_color(v):
        if v > 0:
            return 'rgba(50, 171, 96, 0.7)'  # green
        elif v < 0:
            return 'rgba(219, 64, 82, 0.7)'  # red
        else:
            return 'rgba(200, 200, 200, 0.7)'  # gray
    colors = [get_bar_color(v) for v in score_components.values()]
    
    fig.add_trace(
        go.Bar(
            y=list(score_components.keys()),
            x=list(score_components.values()),
            orientation='h',
            marker_color=colors,
            text=[f"{v:.1f}" for v in score_components.values()],
            textposition='auto',
        ),
        row=1, col=2
    )
    
    # Add details as a table in the second row
    analyst_target = indicators.get('analyst_target', 0)
    target_upside = indicators.get('target_upside', 0)
    analyst_range = ''
    if 'analyst_target_low' in indicators and 'analyst_target_high' in indicators:
        analyst_range = f", range: ${indicators['analyst_target_low']:.0f}–${indicators['analyst_target_high']:.0f}"
    analyst_targets_str = f"${analyst_target:.2f} ({target_upside:+.1f}% upside{analyst_range})" if analyst_target else "N/A"
    details = {
        'RSI': f"{indicators.get('rsi', 0):.1f} ({'Oversold' if indicators.get('rsi', 0) < 30 else 'Overbought' if indicators.get('rsi', 0) > 70 else 'Neutral'})",
        'MACD': f"{indicators.get('macd', 0):.2f} ({'Bullish' if indicators.get('macd', 0) > 0 else 'Bearish'})",
        'BB Position': f"{indicators.get('bb_position', 0):.1f}% ({'Upper' if indicators.get('bb_position', 0) > 80 else 'Lower' if indicators.get('bb_position', 0) < 20 else 'Middle'})",
        'MA Trend': indicators.get('ma_trend', 'Neutral'),
        'Volume': f"{indicators.get('volume_change', 0):.1f}% vs prev day",
        'Analyst Targets': analyst_targets_str
    }
    fig.add_trace(
        go.Table(
            header=dict(values=["<b>Indicator</b>", "<b>Value</b>"], fill_color="paleturquoise", align="left"),
            cells=dict(values=[list(details.keys()), list(details.values())], fill_color="lavender", align="left")
        ),
        row=2, col=1
    )
    
    # Update layout with a single concise annotation for price and target
    concise_text = f"Last: ${indicators.get('last_price', 0):.2f} | Target: ${indicators.get('analyst_target', 0):.2f} ({indicators.get('target_upside', 0):+.1f}%)"
    # Remove the previous technical score annotation and add a new one below the table
    score_blurb = (
        "<b>Technical Score:</b> The score (0–10) is a weighted sum of signals from RSI, MACD, Bollinger Bands, Moving Averages, Volume, and Analyst Targets.<br>"
        "Each bar shows the contribution of a component to the total score.<br>"
        "<span style='font-size:10px'><b>How each score is computed:</b><br>"
        "RSI: +2 if oversold (&lt;30), -1 if overbought (&gt;70), 0 otherwise.<br>"
        "MACD: +2 for strong bullish divergence, +1 for weak bullish, -2 for strong bearish, 0 otherwise.<br>"
        "Bollinger Bands: +2 if price is near/below lower band, -2 if above upper band, 0 otherwise.<br>"
        "Moving Averages: +2 if price &gt;2% above 20d MA, -2 if &lt;2% below, +1 if just above, 0 otherwise.<br>"
        "Volume: +1 if volume spike &gt;20% vs previous day, 0 otherwise.<br>"
        "Analyst Targets: Up to +2 for high upside to analyst target (scaled by % upside)."
        "</span>"
    )
    fig.update_layout(
        title={
            'text': f"{ticker} Technical Analysis Score",
            'x': 0.5,
            'xanchor': 'center'
        },
        height=800,
        width=1100,
        template='plotly_white',
        showlegend=False,
        annotations=[
            dict(
                text=concise_text,
                xref="paper", yref="paper",
                x=0.5, y=1.04,
                showarrow=False,
                font=dict(size=14),
                align="center"
            ),
            dict(
                text=score_blurb,
                xref="paper", yref="paper",
                x=0.4,
                y=-0.0,
                showarrow=False,
                font=dict(size=10),
                align="left",
                bgcolor="white",
                opacity=0.95
            )
        ]
    )
    # Ensure all y-axis labels are shown for the bar chart
    fig.update_yaxes(automargin=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the chart
    output_path = os.path.join(output_dir, f"{ticker}_score.html")
    fig.write_html(output_path)
    
    return output_path 