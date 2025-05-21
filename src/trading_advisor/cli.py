"""Command-line interface functionality."""

import logging
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime
import pandas as pd

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from trading_advisor import __version__
from trading_advisor.analysis import analyze_stock
from trading_advisor.data import download_stock_data, ensure_data_dir, load_positions, load_tickers
from trading_advisor.output import generate_report, generate_structured_data, generate_technical_summary, save_json_report
from trading_advisor.config import SCORE_WEIGHTS
from trading_advisor.visualization import create_stock_chart, create_score_breakdown

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("trading_advisor")
console = Console()

app = typer.Typer(
    name="trading-advisor",
    help="Generate trading advice based on technical indicators and analyst targets.",
    add_completion=False,
)

def version_callback(value: bool):
    """Print version and exit."""
    if value:
        console.print(f"[bold blue]Trading Advisor v{__version__}[/bold blue]")
        raise typer.Exit()

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    )
):
    """Trading Advisor - A tool for generating trading advice based on technical indicators."""
    if len(sys.argv) == 1:
        # Show help menu if no arguments provided
        typer.echo(ctx.get_help())
        raise typer.Exit()

@app.command()
def analyze(
    tickers: Path = typer.Option(
        ...,
        "--tickers", "-t",
        help="Path to file containing ticker symbols"
    ),
    positions: Optional[Path] = typer.Option(
        None,
        "--positions", "-p",
        help="Path to positions CSV file"
    ),
    positions_only: bool = typer.Option(
        False,
        "--positions-only",
        help="Only analyze positions, skip new picks"
    ),
    output: Path = typer.Option(
        "output/analysis.json",
        "--output", "-o",
        help="Path to output JSON file"
    ),
    history_days: int = typer.Option(
        100,
        "--days", "-d",
        help="Number of days of historical data to analyze"
    )
):
    """Analyze stocks and output structured JSON data."""
    try:
        # Create output directory if it doesn't exist
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # Load tickers and positions
        ticker_list = load_tickers(tickers)
        positions_data = load_positions(positions) if positions else {}
        
        # Initialize results
        results = {
            "timestamp": datetime.now().isoformat(),
            "positions": [],
            "new_picks": []
        }
        
        # Analyze positions
        for symbol, position in positions_data.items():
            ticker = symbol
            df = download_stock_data(ticker, history_days=history_days)
            score, score_details, analyst_targets = analyze_stock(ticker, df)
            
            position_data = {
                "ticker": ticker,
                "price_data": df.to_dict(orient="records"),
                "technical_indicators": {
                    "rsi": df["RSI"].tolist(),
                    "macd": {
                        "macd": df["MACD"].tolist(),
                        "signal": df["MACD_Signal"].tolist(),
                        "histogram": df["MACD_Hist"].tolist()
                    },
                    "bollinger_bands": {
                        "upper": df["BB_Upper"].tolist(),
                        "middle": df["BB_Middle"].tolist(),
                        "lower": df["BB_Lower"].tolist()
                    },
                    "moving_averages": {
                        "sma_20": df["SMA_20"].tolist(),
                        "sma_50": df["SMA_50"].tolist(),
                        "sma_200": df["SMA_200"].tolist()
                    }
                },
                "score": {
                    "total": score,
                    "details": score_details
                },
                "position": position,
                "analyst_targets": analyst_targets
            }
            results["positions"].append(position_data)
        
        # Analyze new picks if not positions-only
        if not positions_only:
            for ticker in ticker_list:
                # Skip if already analyzed as a position
                if any(p["ticker"] == ticker for p in results["positions"]):
                    continue
                    
                df = download_stock_data(ticker, history_days=history_days)
                score, score_details, analyst_targets = analyze_stock(ticker, df)
                
                pick_data = {
                    "ticker": ticker,
                    "price_data": df.to_dict(orient="records"),
                    "technical_indicators": {
                        "rsi": df["RSI"].tolist(),
                        "macd": {
                            "macd": df["MACD"].tolist(),
                            "signal": df["MACD_Signal"].tolist(),
                            "histogram": df["MACD_Hist"].tolist()
                        },
                        "bollinger_bands": {
                            "upper": df["BB_Upper"].tolist(),
                            "middle": df["BB_Middle"].tolist(),
                            "lower": df["BB_Lower"].tolist()
                        },
                        "moving_averages": {
                            "sma_20": df["SMA_20"].tolist(),
                            "sma_50": df["SMA_50"].tolist(),
                            "sma_200": df["SMA_200"].tolist()
                        }
                    },
                    "score": {
                        "total": score,
                        "details": score_details
                    },
                    "analyst_targets": analyst_targets
                }
                results["new_picks"].append(pick_data)
        
        # Write results to JSON file
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
            
        typer.echo(f"Analysis complete. Results written to {output}")
        
    except Exception as e:
        print(f"DEBUG ERROR: {e}")
        typer.echo(f"Error during analysis: {str(e)}", err=True)
        raise typer.Exit(1)

@app.command()
def chart(
    ticker: str = typer.Argument(..., help="Stock ticker symbol"),
    output_dir: Path = typer.Option(
        "output/charts",
        "--output-dir", "-o",
        help="Directory to save the charts"
    ),
    history_days: int = typer.Option(
        100,
        "--days", "-d",
        help="Number of days of historical data to include"
    )
):
    """Generate interactive charts for a stock."""
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download and analyze stock data
        df = download_stock_data(ticker, history_days=history_days)
        score, score_details, analyst_targets = analyze_stock(ticker, df)
        
        # Generate charts
        chart_path = create_stock_chart(
            ticker,
            df,
            score_details,
            output_dir=output_dir
        )
        
        score_path = create_score_breakdown(
            ticker,
            score,
            score_details,
            analyst_targets,
            output_dir=output_dir
        )
        
        typer.echo(f"Charts generated successfully:")
        typer.echo(f"  - Price chart: {chart_path}")
        typer.echo(f"  - Score breakdown: {score_path}")
        
    except Exception as e:
        typer.echo(f"Error generating charts: {str(e)}", err=True)
        raise typer.Exit(1)

@app.command()
def version():
    """Show the version of the Trading Advisor."""
    console.print("Trading Advisor v1.0.0")

def run():
    """Run the CLI application."""
    app() 