"""Command-line interface functionality."""

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from trading_advisor import __version__
from trading_advisor.analysis import analyze_stock
from trading_advisor.data import download_stock_data, ensure_data_dir, load_positions, load_tickers
from trading_advisor.output import generate_report, generate_structured_data, generate_technical_summary, save_json_report
from trading_advisor.config import SCORE_WEIGHTS

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

@app.command(no_args_is_help=True)
def analyze(
    tickers: str = typer.Option(
        None,
        "--tickers",
        "-t",
        help="Path to file containing ticker symbols, or 'all' for S&P 500.",
    ),
    positions: Optional[Path] = typer.Option(
        None,
        "--positions",
        "-p",
        help="Path to brokerage CSV file containing current positions.",
    ),
    top_n: int = typer.Option(
        5,
        "--top-n",
        help="Number of top stocks to recommend.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save the markdown report.",
    ),
    save_json: Optional[Path] = typer.Option(
        None,
        "--save-json",
        help="Path to save structured JSON data.",
    ),
    history_days: int = typer.Option(
        100,
        "--history-days",
        help="Number of days of historical data to analyze.",
    ),
    positions_only: bool = typer.Option(
        False,
        "--positions-only",
        help="Only analyze current positions.",
    ),
):
    """Analyze stocks and generate trading advice."""
    try:
        # Ensure data directory exists
        ensure_data_dir()
        
        # Load tickers and positions
        ticker_list = load_tickers(tickers)
        positions_dict = load_positions(positions)
        
        # Union of tickers from tickers.txt and positions file
        all_tickers = set(ticker_list)
        if positions_dict:
            all_tickers.update(positions_dict.keys())
        all_tickers = list(all_tickers)
        
        # Initialize lists for results
        position_results = []
        new_pick_results = []
        structured_data = {
            "timestamp": None,
            "positions": [],
            "new_picks": []
        }
        
        # Create progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            # Create main task
            main_task = progress.add_task(
                "[cyan]Analyzing stocks...",
                total=len(all_tickers)
            )
            
            # Process each ticker
            for ticker in all_tickers:
                # Update progress description
                progress.update(main_task, description=f"[cyan]Analyzing {ticker}...")
                
                # Download and analyze data
                df = download_stock_data(ticker, history_days)
                if df.empty:
                    logger.warning(f"No data available for {ticker}")
                    progress.advance(main_task)
                    continue
                
                # Analyze the stock
                score, score_details, analyst_targets = analyze_stock(ticker, df)
                
                # Generate summary
                position = positions_dict.get(ticker)
                summary = generate_technical_summary(
                    ticker, df, score, score_details, analyst_targets, position
                )
                
                # Add to structured data
                stock_data = generate_structured_data(
                    ticker, df, score, score_details, analyst_targets, position
                )
                
                # Add to appropriate results list
                if position and not positions_only:
                    position_results.append((ticker, score, summary))
                    structured_data["positions"].append(stock_data)
                elif not positions_only:
                    new_pick_results.append((ticker, score, summary))
                    structured_data["new_picks"].append(stock_data)
                
                # Update progress
                progress.advance(main_task)
        
        # Sort results by score
        position_results.sort(key=lambda x: x[1], reverse=True)
        new_pick_results.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N new picks for markdown report only
        report_new_picks = new_pick_results[:top_n]
        
        # Sort structured data by score
        structured_data["positions"].sort(key=lambda x: x["score"]["total"], reverse=True)
        structured_data["new_picks"].sort(key=lambda x: x["score"]["total"], reverse=True)
        
        # Generate and save report
        report = generate_report(
            position_results,
            report_new_picks,  # Use filtered list for markdown
            structured_data,   # Use full sorted list for JSON
            output,
            save_json
        )
        
        # Print report to console
        console.print(report)
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise typer.Exit(1)

def run():
    """Run the CLI application."""
    app() 