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
from trading_advisor.output import generate_report, generate_structured_data, generate_technical_summary, save_json_report, generate_research_prompt, generate_deep_research_prompt
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
    tickers: Optional[Path] = typer.Option(
        None,
        "--tickers", "-t",
        help="Path to file containing ticker symbols (required unless --positions-only is specified)"
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
        # Validate inputs
        if not positions_only and not tickers:
            typer.echo("Error: --tickers is required unless --positions-only is specified", err=True)
            raise typer.Exit(1)
        
        # Create output directory if it doesn't exist
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # Load tickers and positions
        ticker_list = load_tickers(tickers) if tickers else []
        positions_data = load_positions(positions) if positions else {}
        
        # Initialize results
        results = {
            "timestamp": datetime.now().isoformat(),
            "positions": [],
            "new_picks": []
        }
        
        # Analyze positions with progress bar
        if positions_data:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn()) as progress:
                task = progress.add_task("Analyzing positions...", total=len(positions_data))
                for symbol, position in positions_data.items():
                    ticker = symbol
                    df = download_stock_data(ticker, history_days=history_days)
                    score, score_details, analyst_targets = analyze_stock(ticker, df)
                    
                    # Generate structured data for position
                    position_data = generate_structured_data(
                        ticker,
                        df,
                        score,
                        score_details,
                        analyst_targets,
                        position=position
                    )
                    results["positions"].append(position_data)
                    progress.update(task, advance=1)
        else:
            for symbol, position in positions_data.items():
                ticker = symbol
                df = download_stock_data(ticker, history_days=history_days)
                score, score_details, analyst_targets = analyze_stock(ticker, df)
                position_data = generate_structured_data(
                    ticker,
                    df,
                    score,
                    score_details,
                    analyst_targets,
                    position=position
                )
                results["positions"].append(position_data)
        
        # Analyze new picks with progress bar
        new_picks = [ticker for ticker in ticker_list if not any(p["ticker"] == ticker for p in results["positions"])]
        if not positions_only and new_picks:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn()) as progress:
                task = progress.add_task("Analyzing new picks...", total=len(new_picks))
                for ticker in new_picks:
                    df = download_stock_data(ticker, history_days=history_days)
                    score, score_details, analyst_targets = analyze_stock(ticker, df)
                    pick_data = generate_structured_data(
                        ticker,
                        df,
                        score,
                        score_details,
                        analyst_targets
                    )
                    results["new_picks"].append(pick_data)
                    progress.update(task, advance=1)
        elif not positions_only:
            for ticker in new_picks:
                df = download_stock_data(ticker, history_days=history_days)
                score, score_details, analyst_targets = analyze_stock(ticker, df)
                pick_data = generate_structured_data(
                    ticker,
                    df,
                    score,
                    score_details,
                    analyst_targets
                )
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
            df=df,
            ticker=ticker,
            indicators=score_details,
            output_dir=output_dir
        )
        
        score_path = create_score_breakdown(
            ticker=ticker,
            score=score,
            indicators=score_details,
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

@app.command()
def report(
    json_file: Path = typer.Option(
        ...,
        "--json", "-j",
        help="Path to the JSON output file from the analyze command"
    ),
    output: Path = typer.Option(
        "output/report.md",
        "--output", "-o",
        help="Path to save the markdown report"
    )
):
    """Generate a markdown report from the JSON output of the analyze command."""
    try:
        # Load JSON data
        with open(json_file, "r") as f:
            data = json.load(f)

        # Generate markdown report
        report_content = generate_report(data)

        # Save report to file
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(report_content)

        typer.echo(report_content)

    except json.JSONDecodeError:
        typer.echo("Error loading JSON data", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error generating report: {str(e)}", err=True)
        raise typer.Exit(1)

@app.command()
def prompt(
    json_file: str = typer.Option(..., help='Path to analysis JSON file'),
    output: str = typer.Option(..., help='Path to output prompt file')
):
    """Generate a ChatGPT-ready prompt for deep research analysis."""
    try:
        # Load analysis data
        with open(json_file, 'r') as f:
            structured_data = json.load(f)
        
        # Generate prompt
        prompt = generate_research_prompt(structured_data)
        
        # Save prompt
        with open(output, 'w') as f:
            f.write(prompt)
        
        typer.echo(f"Research prompt written to {output}")
        
    except Exception as e:
        typer.echo(f"Error generating research prompt: {str(e)}", err=True)
        raise typer.Exit(1)

@app.command()
def deep_research(
    json_file: Path = typer.Option(
        ...,
        "--json", "-j",
        help="Path to the JSON output file from the analyze command"
    ),
    output: Path = typer.Option(
        "output/deep_research_prompt.md",
        "--output", "-o",
        help="Path to save the deep research prompt"
    )
):
    """Generate a deep research prompt for tactical swing trading analysis."""
    try:
        # Load JSON data
        with open(json_file, "r") as f:
            data = json.load(f)

        # Generate deep research prompt
        prompt_content = generate_deep_research_prompt(data)

        # Save prompt to file
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(prompt_content)

        typer.echo(f"Deep research prompt written to {output}")

    except json.JSONDecodeError:
        typer.echo("Error loading JSON data", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error generating deep research prompt: {str(e)}", err=True)
        raise typer.Exit(1)

def run():
    """Run the CLI application."""
    app() 