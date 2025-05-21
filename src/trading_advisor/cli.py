"""Command-line interface functionality."""

import logging
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime
import pandas as pd
import os

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import plotly.io as pio
from fpdf import FPDF
from PIL import Image

from trading_advisor import __version__
from trading_advisor.analysis import analyze_stock
from trading_advisor.data import download_stock_data, ensure_data_dir, load_positions, load_tickers
from trading_advisor.output import generate_report, generate_structured_data, generate_technical_summary, save_json_report, generate_research_prompt, generate_deep_research_prompt
from trading_advisor.config import SCORE_WEIGHTS
from trading_advisor.visualization import create_stock_chart, create_score_breakdown, create_combined_visualization

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
        
        # Validate positions file if specified
        if positions:
            if not positions.exists():
                typer.echo(f"Error: Positions file '{positions}' does not exist.", err=True)
                raise typer.Exit(1)
        
        # Create output directory if it doesn't exist
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # Load tickers and positions
        ticker_list = load_tickers(tickers) if tickers else []
        positions_data = load_positions(positions) if positions else {}
        
        # If positions_only and no positions loaded, warn and exit
        if positions_only and not positions_data:
            typer.echo(f"Warning: No positions loaded from '{positions}'. Check the file path and format.", err=True)
            raise typer.Exit(1)
        
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
        
        # Sort positions and new_picks by score['total'] descending
        results['positions'].sort(key=lambda x: x.get('score', {}).get('total', 0), reverse=True)
        results['new_picks'].sort(key=lambda x: x.get('score', {}).get('total', 0), reverse=True)
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
    ),
    pdf: bool = typer.Option(
        False,
        "--pdf",
        help="Export both charts as images and combine into a single PDF"
    )
):
    """Generate interactive charts for a stock."""
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download and analyze stock data
        df = download_stock_data(ticker, history_days=history_days)
        score, score_details, analyst_targets = analyze_stock(ticker, df)

        # Extract latest values for richer indicators
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        # Calculate BB position (percent between lower and upper)
        if 'BB_Lower' in latest and 'BB_Upper' in latest and (latest['BB_Upper'] - latest['BB_Lower']) != 0:
            bb_position = (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower']) * 100
        else:
            bb_position = 0.0
        # Calculate MA trend
        ma_trend = (
            'Bullish' if latest.get('Close', 0) > latest.get('SMA_20', 0)
            else 'Bearish' if latest.get('Close', 0) < latest.get('SMA_20', 0)
            else 'Neutral'
        )
        # Calculate volume change
        if prev['Volume'] != 0:
            volume_change = (latest['Volume'] - prev['Volume']) / prev['Volume'] * 100
        else:
            volume_change = 0.0
        # Analyst target and upside
        analyst_target = analyst_targets['median_target'] if analyst_targets and 'median_target' in analyst_targets else 0.0
        last_price = latest.get('Close', 0.0)
        target_upside = ((analyst_target - last_price) / last_price * 100) if last_price else 0.0
        # MACD value
        macd_value = latest.get('MACD_Hist', 0.0)
        # Build indicators dict for score breakdown
        indicators = {
            # Score contributions (old keys for backward compatibility)
            'rsi_score': score_details.get('rsi', 0.0),
            'bb_score': score_details.get('bollinger', 0.0),
            'macd_score': score_details.get('macd', 0.0),
            'ma_score': score_details.get('moving_averages', 0.0),
            'volume_score': score_details.get('volume', 0.0),
            'analyst_targets_score': score_details.get('analyst_targets', 0.0),
            # Actual indicator values
            'rsi': latest.get('RSI', 0.0),
            'macd': macd_value,
            'bb_position': bb_position,
            'ma_trend': ma_trend,
            'volume_change': volume_change,
            'last_price': last_price,
            'analyst_target': analyst_target,
            'target_upside': target_upside,
        }
        # Sanitize indicators to remove any <br> or \n from string values
        for k, v in indicators.items():
            if isinstance(v, str):
                indicators[k] = v.replace('<br>', ' ').replace('\n', ' ')
        # Generate charts
        chart_path, chart_fig = create_stock_chart(
            df=df,
            ticker=ticker,
            indicators=score_details,
            output_dir=output_dir,
            return_fig=True
        )
        score_path, score_fig = create_score_breakdown(
            ticker=ticker,
            score=score,
            indicators=indicators,
            output_dir=output_dir,
            return_fig=True
        )
        typer.echo(f"Charts generated successfully:")
        typer.echo(f"  - Price chart: {chart_path}")
        typer.echo(f"  - Score breakdown: {score_path}")

        # Set high-res layout for export (A4 at 1654x1170 px)
        a4_px_width = 1654
        a4_px_height = 1170
        chart_fig.update_layout(width=a4_px_width, height=a4_px_height)
        score_fig.update_layout(width=a4_px_width, height=a4_px_height)
        if pdf:
            # Export HTML charts to PNG at native A4 size
            chart_img = os.path.splitext(chart_path)[0] + ".png"
            score_img = os.path.splitext(score_path)[0] + ".png"
            chart_fig.write_image(chart_img, format="png", scale=1)
            score_fig.write_image(score_img, format="png", scale=1)
            # Combine into PDF
            pdf_path = output_dir / f"{ticker}_charts.pdf"
            pdf = FPDF(unit="pt", format=[a4_px_width * 0.75, a4_px_height * 0.75])
            for img_path in [score_img, chart_img]:
                cover = Image.open(img_path)
                width, height = cover.size
                pdf.add_page()
                # Insert image at native size (no resizing)
                pdf.image(img_path, 0, 0, width * 0.75, height * 0.75)
            pdf.output(str(pdf_path))
            typer.echo(f"PDF exported: {pdf_path}")

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
    output: str = typer.Option(..., help='Path to output prompt file'),
    deep_research: bool = typer.Option(
        False,
        "--deep-research",
        help="Generate a deep research prompt with tactical swing trading format"
    ),
    top_n: int = typer.Option(
        6,
        "--top-n",
        help="Number of highest-scoring new picks to include in the prompt"
    )
):
    """Generate a research prompt from analysis data."""
    try:
        # Ensure output directory exists
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Generate prompt
        if deep_research:
            prompt_text = generate_deep_research_prompt(data, top_n=top_n)
        else:
            prompt_text = generate_research_prompt(data, top_n=top_n)
        
        # Save prompt to file
        with open(output_path, 'w') as f:
            f.write(prompt_text)
        
        if deep_research:
            typer.echo(f"Deep research prompt written to {output}")
        else:
            typer.echo(f"Prompt written to {output}")
        
    except Exception as e:
        import traceback
        print('PROMPT DEBUG ERROR:', e)
        traceback.print_exc()
        typer.echo(f"Error generating research prompt: {str(e)}", err=True)
        raise typer.Exit(1)

def run():
    """Run the CLI application."""
    app() 