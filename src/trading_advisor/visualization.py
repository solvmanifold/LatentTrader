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
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bb_upper'],
                name='BB Upper',
                line=dict(color='rgba(250, 0, 0, 0.3)')
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bb_lower'],
                name='BB Lower',
                line=dict(color='rgba(0, 250, 0, 0.3)'),
                fill='tonexty'
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
    Create a bar chart showing the breakdown of the technical score.
    
    Args:
        ticker: Stock ticker symbol
        score: Total technical score
        indicators: Dictionary containing technical indicators and their contributions
        output_dir: Directory to save the chart
        
    Returns:
        Path to the saved HTML file
    """
    # Extract score components from indicators
    score_components = {
        'RSI': indicators.get('rsi_score', 0),
        'Bollinger Bands': indicators.get('bb_score', 0),
        'MACD': indicators.get('macd_score', 0),
        'Moving Averages': indicators.get('ma_score', 0),
        'Volume': indicators.get('volume_score', 0)
    }
    
    # Create figure
    fig = go.Figure()
    
    # Add horizontal bar chart
    fig.add_trace(
        go.Bar(
            y=list(score_components.keys()),
            x=list(score_components.values()),
            orientation='h',
            marker_color='lightblue'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Score Breakdown (Total: {score:.1f})",
        xaxis_title="Score Contribution",
        yaxis_title="Indicator",
        height=400,
        template='plotly_white'
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the chart
    output_path = os.path.join(output_dir, f"{ticker}_score.html")
    fig.write_html(output_path)
    
    return output_path 