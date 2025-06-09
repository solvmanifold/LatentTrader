"""Command-line interface functionality."""

import logging
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime
import pandas as pd
import os
from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler
import numpy as np
import click
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import plotly.io as pio
from fpdf import FPDF
from PIL import Image
import pandas_market_calendars as mcal

from trading_advisor import __version__
from trading_advisor.analysis import analyze_stock, calculate_technical_indicators, calculate_score, get_analyst_targets
from trading_advisor.data import download_stock_data, ensure_data_dir, load_positions, load_tickers, normalize_ticker
from trading_advisor.output import generate_report, generate_structured_data, generate_technical_summary, save_json_report, generate_research_prompt, generate_deep_research_prompt
from trading_advisor.config import SCORE_WEIGHTS
from trading_advisor.visualization import create_stock_chart, create_score_breakdown, create_combined_visualization
from trading_advisor.backtest import run_backtest
from trading_advisor.features import update_features as update_stock_features
from trading_advisor.market_breadth import calculate_market_breadth
from trading_advisor.sector_performance import calculate_sector_performance
from trading_advisor.sentiment import MarketSentiment
from trading_advisor.market_features import MarketFeatures
from trading_advisor.models.dataset import (
    DatasetGenerator, 
    remove_unnecessary_features,
    save_feature_mappings,
    apply_feature_mappings
)

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# File handler with rotation
file_handler = RotatingFileHandler(
    "logs/trading_advisor.log",
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

# Terminal handler (RichHandler for pretty output)
console_handler = RichHandler(rich_tracebacks=True)
console_handler.setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
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
def chart(
    tickers: list[str] = typer.Argument(None, help="Stock ticker symbols (can specify multiple)", show_default=False),
    output_dir: Path = typer.Option(
        "output/charts",
        "--output-dir", "-o",
        help="Directory to save the charts"
    ),
    history_days: int = typer.Option(
        100,
        "--days", "-d",
        help="Number of days of historical data to include (ignored if --start-date is provided)"
    ),
    pdf: bool = typer.Option(
        False,
        "--pdf",
        help="Export both charts as images and combine into a single PDF"
    ),
    json_file: Optional[Path] = typer.Option(
        None,
        "--json", "-j",
        help="Path to analysis JSON file (from analyze command)"
    ),
    start_date: Optional[str] = typer.Option(
        None,
        "--start-date",
        help="Start date (YYYY-MM-DD) for historical data (overrides --days)"
    ),
    end_date: Optional[str] = typer.Option(
        None,
        "--end-date",
        help="End date (YYYY-MM-DD) for historical data (default: today)"
    )
):
    """Generate interactive charts for one or more stocks, optionally using an analysis JSON file."""
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Special case: if a single positional argument is given and it looks like a .json file, suggest using --json
        if tickers and len(tickers) == 1 and str(tickers[0]).lower().endswith('.json') and not json_file:
            typer.echo(f"Error: It looks like you provided a JSON file as a ticker. Did you mean to use --json {tickers[0]}?\n\nUsage: trading-advisor chart --json {tickers[0]}", err=True)
            raise typer.Exit(1)

        # Load tickers and analysis data from JSON if provided
        analysis_data = None
        tickers_from_json = []
        if json_file:
            if not json_file.exists():
                typer.echo(f"Error: JSON file '{json_file}' does not exist.", err=True)
                raise typer.Exit(1)
            with open(json_file, "r") as f:
                analysis_data = json.load(f)
            # Collect all tickers from positions and new_picks
            tickers_from_json = [p["ticker"] for p in analysis_data.get("positions", [])]
            tickers_from_json += [p["ticker"] for p in analysis_data.get("new_picks", [])]
            tickers_from_json = list(set(tickers_from_json))

        # Determine which tickers to chart
        if json_file and not tickers:
            tickers_to_chart = tickers_from_json
        elif json_file and tickers:
            tickers_to_chart = [t for t in tickers if t in tickers_from_json]
            if not tickers_to_chart:
                typer.echo("Error: None of the specified tickers are present in the JSON file.\n\nIf you want to chart all tickers from the JSON, use:\n  trading-advisor chart --json <file.json>\nIf you want to chart a subset, use:\n  trading-advisor chart TICKER1 TICKER2 --json <file.json>", err=True)
                raise typer.Exit(1)
        elif not json_file and tickers:
            tickers_to_chart = tickers
        else:
            typer.echo("Error: You must specify at least one ticker or provide a JSON file.\n\nUsage examples:\n  trading-advisor chart AAPL MSFT\n  trading-advisor chart --json output/analysis.json\n  trading-advisor chart AAPL --json output/analysis.json", err=True)
            raise typer.Exit(1)

        # Build a lookup for analysis data if available
        analysis_lookup = {}
        if analysis_data:
            for p in analysis_data.get("positions", []) + analysis_data.get("new_picks", []):
                analysis_lookup[p["ticker"]] = p

        # Process each ticker with progress bar
        chart_infos = []
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn()) as progress:
            task = progress.add_task("Generating charts...", total=len(tickers_to_chart))

            for ticker in tickers_to_chart:
                # Use analysis data from JSON if available, else download/analyze
                if ticker in analysis_lookup:
                    p = analysis_lookup[ticker]
                    # Reconstruct DataFrame from price_data if possible, else fallback
                    # For now, just use the score and indicators from JSON
                    score = p.get("score", {}).get("total", 0.0)
                    score_details = p.get("score", {}).get("details", {})
                    analyst_targets = p.get("analyst_targets", None)
                    # Download data for charting (could be optimized to store full df in JSON)
                    if end_date is not None:
                        end_date_dt = pd.to_datetime(end_date)
                    else:
                        end_date_dt = pd.to_datetime('today')
                    if start_date is not None:
                        start_date_dt = pd.to_datetime(start_date)
                    else:
                        start_date_dt = end_date_dt - pd.Timedelta(days=history_days)
                    df = download_stock_data(ticker, start_date=start_date_dt, end_date=end_date_dt)
                else:
                    if end_date is not None:
                        end_date_dt = pd.to_datetime(end_date)
                    else:
                        end_date_dt = pd.to_datetime('today')
                    if start_date is not None:
                        start_date_dt = pd.to_datetime(start_date)
                    else:
                        start_date_dt = end_date_dt - pd.Timedelta(days=history_days)
                    df = download_stock_data(ticker, start_date=start_date_dt, end_date=end_date_dt)
                score, score_details, analyst_targets = analyze_stock(ticker, df)
                
                # Extract latest values for richer indicators
                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else latest
                if 'BB_Lower' in latest and 'BB_Upper' in latest and (latest['BB_Upper'] - latest['BB_Lower']) != 0:
                    bb_position = (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower']) * 100
                else:
                    bb_position = 0.0
                ma_trend = (
                    'Bullish' if latest.get('Close', 0) > latest.get('SMA_20', 0)
                    else 'Bearish' if latest.get('Close', 0) < latest.get('SMA_20', 0)
                    else 'Neutral'
                )
                if prev['Volume'] != 0:
                    volume_change = (latest['Volume'] - prev['Volume']) / prev['Volume'] * 100
                else:
                    volume_change = 0.0
                analyst_target = analyst_targets['median_target'] if analyst_targets and 'median_target' in analyst_targets else 0.0
                last_price = latest.get('Close', 0.0)
                target_upside = ((analyst_target - last_price) / last_price * 100) if last_price else 0.0
                macd_value = latest.get('MACD_Hist', 0.0)
                # Build indicators dict for score breakdown and chart overlays
                indicators = {
                    'rsi_score': score_details.get('rsi', 0.0),
                    'bb_score': score_details.get('bollinger', 0.0),
                    'macd_score': score_details.get('macd', 0.0),
                    'ma_score': score_details.get('moving_averages', 0.0),
                    'volume_score': score_details.get('volume', 0.0),
                    'analyst_targets_score': score_details.get('analyst_targets', 0.0),
                    # Actual indicator values for overlays
                    'rsi': latest.get('RSI', 0.0),
                    'macd': macd_value,
                    'bb_position': bb_position,
                    'ma_trend': ma_trend,
                    'volume_change': volume_change,
                    'last_price': last_price,
                    'analyst_target': analyst_target,
                    'target_upside': target_upside,
                    # Overlay values
                    'MA20': latest.get('MA20', None),
                    'MA50': latest.get('MA50', None),
                    'BB_Upper': latest.get('BB_Upper', None),
                    'BB_Lower': latest.get('BB_Lower', None),
                    'BB_Middle': latest.get('BB_Middle', None),
                }
                for k, v in indicators.items():
                    if isinstance(v, str):
                        indicators[k] = v.replace('<br>', ' ').replace('\n', ' ')
                chart_path, chart_fig = create_stock_chart(
                    df=df,
                    ticker=ticker,
                    indicators=indicators,
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
                # Collect chart info for later logging
                chart_infos.append((ticker, chart_path, score_path))

                a4_px_width = 1654
                a4_px_height = 1170
                chart_fig.update_layout(width=a4_px_width, height=a4_px_height)
                score_fig.update_layout(width=a4_px_width, height=a4_px_height)
                
                if pdf:
                    chart_img = os.path.splitext(chart_path)[0] + ".png"
                    score_img = os.path.splitext(score_path)[0] + ".png"
                    chart_fig.write_image(chart_img, format="png", scale=1)
                    score_fig.write_image(score_img, format="png", scale=1)
                    if ticker == tickers_to_chart[0]:
                        pdf_path = output_dir / f"charts.pdf"
                        pdf_obj = FPDF(unit="pt", format=[a4_px_width * 0.75, a4_px_height * 0.75])
                    for img_path in [score_img, chart_img]:
                        cover = Image.open(img_path)
                        width, height = cover.size
                        pdf_obj.add_page()
                        pdf_obj.image(img_path, 0, 0, width * 0.75, height * 0.75)
                    if ticker == tickers_to_chart[-1]:
                        pdf_obj.output(str(pdf_path))
                        typer.echo(f"PDF exported: {pdf_path}")
                progress.update(task, advance=1)

        # After progress bar, print all chart info
        for ticker, chart_path, score_path in chart_infos:
            console.print(f"[bold green]Charts generated for {ticker}:[/bold green]")
            console.print(f"  - Price chart: {chart_path}")
            console.print(f"  - Score breakdown: {score_path}")
        
    except Exception as e:
        typer.echo(f"Error generating charts: {str(e)}", err=True)
        raise typer.Exit(1)

@app.command()
def version():
    """Show the version of the Trading Advisor."""
    console.print("Trading Advisor v1.0.0")

@app.command()
def report_daily(
    model_name: str = typer.Option("TechnicalScorer", help="Model to report on (e.g., 'TechnicalScorer')"),
    tickers: str = typer.Option("all", help="Comma-separated tickers, path to file, or 'all' for all tickers"),
    date: str = typer.Option(..., help="Generate report for this date (YYYY-MM-DD)"),
    top_n: int = typer.Option(6, help="Number of top tickers to include in the report"),
    model_outputs_dir: str = typer.Option("model_outputs", help="Directory with model output Parquet files"),
    reports_dir: str = typer.Option("reports", help="Directory to save report files and Parquet table"),
    force: bool = typer.Option(False, help="Overwrite existing report if it exists"),
    positions_csv: str = typer.Option(None, help="CSV file of current positions to always include in the report")
):
    """
    Generate a Markdown report for the top-N tickers by score for a given date and model, including OHLC and technical/analyst info.
    Save as both a Markdown file and a row in a Parquet table for historical tracking.
    Optionally, always include tickers from a positions CSV, marked as current positions.
    """
    import os
    import pandas as pd
    from trading_advisor.data import load_tickers
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    import json

    logger = logging.getLogger("trading_advisor.report_daily")

    # Parse tickers
    if tickers == "all":
        ticker_list = load_tickers("all")
    elif os.path.isfile(tickers):
        with open(tickers) as f:
            ticker_list = [line.strip() for line in f if line.strip()]
    else:
        ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]

    # Parse positions CSV if provided
    positions_set = set()
    if positions_csv:
        try:
            pos_df = pd.read_csv(positions_csv, skiprows=2)
            if "Symbol" in pos_df.columns:
                # Only keep rows where Symbol looks like a ticker (A-Z, 0-9, no spaces)
                valid = pos_df["Symbol"].astype(str).str.match(r"^[A-Z0-9\.\-]+$")
                positions_set = set(pos_df.loc[valid, "Symbol"].dropna().astype(str).str.strip())
                print(f"[DEBUG] Parsed positions_set: {positions_set}")
            else:
                print("[DEBUG] 'Symbol' column not found in positions CSV.")
        except Exception as e:
            typer.echo(f"Error reading positions CSV: {e}", err=True)
            raise typer.Exit(1)

    # Create reports directory
    os.makedirs(reports_dir, exist_ok=True)
    report_md_path = os.path.join(reports_dir, f"{model_name}_{date}.md")
    report_parquet_path = os.path.join(reports_dir, f"{model_name}.parquet")
    model_output_dir = os.path.join(model_outputs_dir, model_name)

    # Check if report already exists
    if os.path.exists(report_md_path) and not force:
        typer.echo(f"Report already exists: {report_md_path}. Use --force to overwrite.", err=True)
        raise typer.Exit(1)

    # Load existing report table if it exists
    if os.path.exists(report_parquet_path):
        report_df = pd.read_parquet(report_parquet_path)
    else:
        report_df = pd.DataFrame()

    # Collect data for the report
    rows = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn()) as progress:
        task = progress.add_task(f"Gathering data for {model_name} on {date}...", total=len(ticker_list))
        for ticker in ticker_list:
            out_path = os.path.join(model_output_dir, f"{ticker}.parquet")
            if not os.path.exists(out_path):
                logger.warning(f"Model output not found for {ticker}: {out_path}")
                progress.update(task, advance=1)
                continue
            try:
                df = pd.read_parquet(out_path)
                if date not in df.index:
                    logger.warning(f"No data for {ticker} on {date}")
                    progress.update(task, advance=1)
                    continue
                row = df.loc[date]
                # If row is a DataFrame (multi-index), take the first
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                rows.append({
                    "ticker": ticker,
                    "score": row.get("score", None),
                    "Open": row.get("Open", None),
                    "High": row.get("High", None),
                    "Low": row.get("Low", None),
                    "Close": row.get("Close", None),
                    "RSI": row.get("RSI", None),
                    "MACD": row.get("MACD", None),
                    "MACD_Signal": row.get("MACD_Signal", None),
                    "MACD_Hist": row.get("MACD_Hist", None),
                    "BB_Upper": row.get("BB_Upper", None),
                    "BB_Middle": row.get("BB_Middle", None),
                    "BB_Lower": row.get("BB_Lower", None),
                    "SMA_20": row.get("SMA_20", None),
                    "analyst_targets": row.get("analyst_targets", None),
                    "is_position": ticker in positions_set
                })
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
            progress.update(task, advance=1)

    # Rank by score and select top N (positions do not count toward top_n)
    rows = [r for r in rows if r["score"] is not None]
    # Separate positions and non-positions
    positions_rows = [r for r in rows if r["is_position"]]
    non_position_rows = [r for r in rows if not r["is_position"]]
    # Sort non-positions by score and take top_n
    top_non_positions = sorted(non_position_rows, key=lambda x: x["score"], reverse=True)[:top_n]
    # Add all positions not already in top_n
    top_rows = top_non_positions.copy()
    top_tickers_set = {r["ticker"] for r in top_rows}
    for r in positions_rows:
        if r["ticker"] not in top_tickers_set:
            top_rows.append(r)
    # If a position is also in top_n, it will appear only once, marked as a position
    # Round all numeric values to the hundredth place
    def round_val(val):
        if isinstance(val, (int, float)):
            return round(val, 2)
        return val
    for r in top_rows:
        for k in ["score", "Open", "High", "Low", "Close", "RSI", "MACD", "MACD_Signal", "MACD_Hist", "BB_Upper", "BB_Middle", "BB_Lower", "SMA_20"]:
            if k in r and r[k] is not None:
                r[k] = round_val(r[k])
        # Analyst targets (if present and parseable)
        at = r.get("analyst_targets")
        if at:
            try:
                at_obj = json.loads(at) if isinstance(at, str) else at
                for key in ["median_target", "low_target", "high_target"]:
                    if key in at_obj and at_obj[key] is not None:
                        at_obj[key] = round_val(at_obj[key])
                r["analyst_targets"] = at_obj
            except Exception:
                pass

    # Generate Markdown report
    from datetime import datetime as dt
    md_lines = [
        f"# Trading Advisor Report\n",
        f"Generated on: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    ]
    # Current Positions section
    if positions_rows:
        md_lines.append(f"## Current Positions\n")
        for r in positions_rows:
            md_lines.append(f"### {r['ticker']}")
            md_lines.append(f"**Technical Score:** {r['score']:.2f}/10")
            md_lines.append(f"**OHLC:** Open: {r['Open']}, High: {r['High']}, Low: {r['Low']}, Close: {r['Close']}")
            md_lines.append(f"**Current Position**")
            md_lines.append(f"**Technical Indicators:**")
            md_lines.append(f"- RSI: {r['RSI']}")
            md_lines.append(f"- MACD value: {r['MACD']}")
            md_lines.append(f"- MACD signal: {r['MACD_Signal']}")
            md_lines.append(f"- MACD histogram: {r['MACD_Hist']}")
            md_lines.append(f"- BOLLINGER_BANDS upper: {r['BB_Upper']}")
            md_lines.append(f"- BOLLINGER_BANDS middle: {r['BB_Middle']}")
            md_lines.append(f"- BOLLINGER_BANDS lower: {r['BB_Lower']}")
            md_lines.append(f"- MOVING_AVERAGES sma_20: {r['SMA_20']}")
            analyst_targets = r.get("analyst_targets")
            if analyst_targets:
                try:
                    at = analyst_targets
                    md_lines.append(f"**Analyst Targets:**")
                    if "median_target" in at:
                        md_lines.append(f"- Median: ${at['median_target']}")
                    if "low_target" in at and "high_target" in at:
                        md_lines.append(f"- Range: ${at['low_target']} - ${at['high_target']}")
                except Exception:
                    md_lines.append(f"**Analyst Targets:** {analyst_targets}")
            md_lines.append("")
    # New Technical Picks section
    if top_non_positions:
        md_lines.append(f"## New Technical Picks (Top {top_n}) for {date} ({model_name})\n")
        for r in top_non_positions:
            md_lines.append(f"### {r['ticker']}")
            md_lines.append(f"**Technical Score:** {r['score']:.2f}/10")
            md_lines.append(f"**OHLC:** Open: {r['Open']}, High: {r['High']}, Low: {r['Low']}, Close: {r['Close']}")
            md_lines.append(f"**Technical Indicators:**")
            md_lines.append(f"- RSI: {r['RSI']}")
            md_lines.append(f"- MACD value: {r['MACD']}")
            md_lines.append(f"- MACD signal: {r['MACD_Signal']}")
            md_lines.append(f"- MACD histogram: {r['MACD_Hist']}")
            md_lines.append(f"- BOLLINGER_BANDS upper: {r['BB_Upper']}")
            md_lines.append(f"- BOLLINGER_BANDS middle: {r['BB_Middle']}")
            md_lines.append(f"- BOLLINGER_BANDS lower: {r['BB_Lower']}")
            md_lines.append(f"- MOVING_AVERAGES sma_20: {r['SMA_20']}")
            analyst_targets = r.get("analyst_targets")
            if analyst_targets:
                try:
                    at = analyst_targets
                    md_lines.append(f"**Analyst Targets:**")
                    if "median_target" in at:
                        md_lines.append(f"- Median: ${at['median_target']}")
                    if "low_target" in at and "high_target" in at:
                        md_lines.append(f"- Range: ${at['low_target']} - ${at['high_target']}")
                except Exception:
                    md_lines.append(f"**Analyst Targets:** {analyst_targets}")
            md_lines.append("")

    # Write Markdown report
    with open(report_md_path, "w") as f:
        f.write("\n".join(md_lines))
    typer.echo(f"Markdown report written to {report_md_path}")

    # Save to Parquet table for historical tracking
    # Store the report text and the top-N tickers/scores as a row
    new_row = {
        "date": date,
        "model": model_name,
        "top_tickers": [r["ticker"] for r in top_rows],
        "scores": [r["score"] for r in top_rows],
        "report_md": "\n".join(md_lines)
    }
    if not report_df.empty:
        # Remove any existing row for this date/model
        report_df = report_df[~((report_df["date"] == date) & (report_df["model"] == model_name))]
        report_df = pd.concat([report_df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        report_df = pd.DataFrame([new_row])
    report_df = report_df.sort_values(["date", "model"]).reset_index(drop=True)
    report_df.to_parquet(report_parquet_path)
    typer.echo(f"Report row saved to {report_parquet_path}")

@app.command()
def prompt_daily(
    model_name: str = typer.Option("TechnicalScorer", help="Model to prompt for (e.g., 'TechnicalScorer')"),
    date: str = typer.Option(..., help="Generate prompt for this date (YYYY-MM-DD)"),
    reports_dir: str = typer.Option("reports", help="Directory with report files and Parquet table"),
    prompts_dir: str = typer.Option("prompts", help="Directory to save prompt files and Parquet table"),
    top_n: int = typer.Option(6, help="Number of top tickers to include in the prompt"),
    force: bool = typer.Option(False, help="Overwrite existing prompt if it exists"),
    deep_research: bool = typer.Option(False, help="Generate a deep research prompt with tactical swing trading format")
):
    """
    Generate a daily LLM-ready prompt for a model and date, based on the daily report. Save as both a text file and a row in a Parquet table.
    Use --deep-research for a more detailed, research-oriented prompt format.
    """
    import os
    import pandas as pd
    import re

    # Paths
    os.makedirs(prompts_dir, exist_ok=True)
    prompt_suffix = "_DR" if deep_research else ""
    prompt_txt_path = os.path.join(prompts_dir, f"{model_name}_{date}{prompt_suffix}.txt")
    prompt_parquet_path = os.path.join(prompts_dir, f"{model_name}.parquet")
    report_parquet_path = os.path.join(reports_dir, f"{model_name}.parquet")

    # Check if prompt already exists
    if os.path.exists(prompt_txt_path) and not force:
        typer.echo(f"Prompt already exists: {prompt_txt_path}. Use --force to overwrite.", err=True)
        raise typer.Exit(1)

    # Load the report table
    if not os.path.exists(report_parquet_path):
        typer.echo(f"Report table not found: {report_parquet_path}", err=True)
        raise typer.Exit(1)
    report_df = pd.read_parquet(report_parquet_path)
    row = report_df[(report_df["date"] == date) & (report_df["model"] == model_name)]
    if row.empty:
        typer.echo(f"No report found for {model_name} on {date}", err=True)
        raise typer.Exit(1)
    row = row.iloc[0]
    report_md = row["report_md"]

    # Parse the report Markdown for sections
    current_positions_blocks = []
    new_picks_blocks = []
    in_current = False
    in_new = False
    for line in report_md.splitlines():
        if line.strip().startswith("## Current Positions"):
            in_current = True
            in_new = False
            continue
        if line.strip().startswith("## New Technical Picks"):
            in_current = False
            in_new = True
            continue
        if line.strip().startswith("## "):
            in_current = False
            in_new = False
            continue
        if in_current:
            current_positions_blocks.append(line)
        if in_new:
            new_picks_blocks.append(line)
    def split_ticker_blocks(block_lines):
        blocks = []
        current = []
        for l in block_lines:
            if l.strip().startswith("### ") and current:
                blocks.append("\n".join(current).strip())
                current = [l]
            else:
                current.append(l)
        if current:
            blocks.append("\n".join(current).strip())
        return [b.strip() for b in blocks if b.strip()]
    current_blocks = split_ticker_blocks(current_positions_blocks)
    new_blocks = split_ticker_blocks(new_picks_blocks)
    new_blocks = new_blocks[:top_n]

    def join_blocks_with_leading_blank(blocks):
        # For each block except the first, prefix with two blank lines
        if not blocks:
            return ""
        result = [blocks[0].strip()]
        for b in blocks[1:]:
            result.append("\n\n" + b.strip())
        return "".join(result)

    if deep_research:
        # Deep research prompt template
        prompt = (
            "You are a tactical swing trader and market strategist evaluating a technical scan and open positions. Your task is to develop an actionable 1–2 week trading playbook for each stock listed below.\n\n"
            "In addition to the provided technical summaries and analyst targets, use current market context, news events, earnings calendars, and public sentiment to support or override your recommendations.\n\n"
            "When helpful, include macro events (e.g. Fed, CPI), earnings dates, or notable sentiment drivers (Reddit, upgrades/downgrades, insider activity).\n\n"
            "Use real-time information from search, Reddit (e.g., r/stocks, r/wallstreetbets), financial media (e.g., CNBC, Bloomberg), and social sentiment tools if relevant.\n\n"
            "Return your response in this exact bullet format for each stock:\n\n"
            "✅ Action (e.g. Buy Now, Hold, Adjust)  \n"
            "🎯 Entry strategy (limit or breakout entry, price conditions, timing)  \n"
            "🛑 Stop-loss level (specific price or %)  \n"
            "💰 Profit-taking strategy (target price, resistance level, or trailing stop)  \n"
            "🔍 Confidence level (High / Medium / Low)  \n"
            "🧠 Rationale (1–2 lines)\n\n"
            "Begin each ticker with a 💡 summary that integrates both technical and real-time context.\n\n"
            "If a setup is weak or conflicting, say 'No trade this week' and explain why.\n\n"
            "Assume:\n"
            "- A 1–2 week swing trade horizon\n"
            "- Technicals are important, but can be overridden by breaking news, macro conditions, or earnings catalysts\n"
            "- The investor is risk-aware but ready to act decisively on high-conviction short-term setups\n\n"
            "Prioritize quality over quantity. Be specific and tactical in your recommendations.\n\n"
            "---\n\n"
            "📊 Current Positions:\n\n"
            + (join_blocks_with_leading_blank(current_blocks) + "\n\n" if current_blocks else "")
            + "📊 New Technical Picks:\n\n"
            + (join_blocks_with_leading_blank(new_blocks) + "\n" if new_blocks else "")
        )
    else:
        # Standard prompt template
        prompt = (
            "You are a tactical swing trader managing a technical scan and open positions.\n"
            "Your task is to return an actionable 1–2 week trading playbook for each stock listed below.\n\n"
            "For each Current Position:\n"
            "- Recommend Hold, Sell, or Adjust\n"
            "- If \"Adjust\", provide a tactical move: e.g., raise stop, set trailing stop, scale out\n"
            "- Include a recommended stop-loss level and optional profit-taking level\n"
            "- Keep risk in mind—prioritize capital preservation if signals are weakening\n\n"
            "For each New Technical Pick:\n"
            "- Decide if it's a viable trade this week\n"
            "- If yes, provide:\n"
            "  - Entry strategy: buy now, wait for pullback, wait for breakout, etc.\n"
            "  - Stop-loss: price level or % below\n"
            "  - Profit target: based on analyst target, momentum, or resistance\n"
            "  - Confidence level: High / Medium / Low\n\n"
            "Assume:\n"
            "- A 1–2 week swing trade horizon\n"
            "- Technicals and analyst targets are the primary inputs\n"
            "- The investor is risk-aware but willing to act on strong short-term setups\n\n"
            "Be concise, tactical, and make clear, justified recommendations.\n\n"
            "Focus most attention on the 💡 summary line. Use the structured data only to support or refine the thesis.\n\n"
            "---\n\n"
            "## Current Positions\n\n"
            + (join_blocks_with_leading_blank(current_blocks) + "\n\n" if current_blocks else "")
            + "## New Technical Picks\n\n"
            + (join_blocks_with_leading_blank(new_blocks) + "\n" if new_blocks else "")
        )

    # Write prompt to text file
    with open(prompt_txt_path, "w") as f:
        f.write(prompt)
    typer.echo(f"Prompt written to {prompt_txt_path}")

    # Save to Parquet table for historical tracking
    # Store the prompt text and the top-N tickers/scores as a row
    if os.path.exists(prompt_parquet_path):
        prompt_df = pd.read_parquet(prompt_parquet_path)
        if 'deep_research' not in prompt_df.columns:
            prompt_df['deep_research'] = False
    else:
        prompt_df = pd.DataFrame()
    new_row = {
        "date": date,
        "model": model_name,
        "deep_research": deep_research,
        "prompt_txt": prompt
    }
    if not prompt_df.empty:
        # Remove any existing row for this date/model/deep_research
        prompt_df = prompt_df[~((prompt_df["date"] == date) & (prompt_df["model"] == model_name) & (prompt_df["deep_research"] == deep_research))]
        prompt_df = pd.concat([prompt_df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        prompt_df = pd.DataFrame([new_row])
    prompt_df = prompt_df.sort_values(["date", "model", "deep_research"]).reset_index(drop=True)
    prompt_df.to_parquet(prompt_parquet_path)
    typer.echo(f"Prompt row saved to {prompt_parquet_path}")

@app.command()
def update_data(
    tickers_input: Optional[Path] = typer.Argument(
        None,
        help="Path to file with tickers, or 'all' for S&P 500. If omitted, defaults to 'all' to update all S&P 500 tickers."
    ),
    days: int = typer.Option(60, help="Number of days of historical data to download"),
    features_dir: str = typer.Option("data/ticker_features", help="Directory to store feature files"),
    update_tickers: bool = typer.Option(
        True,
        "--update-tickers/--no-update-tickers",
        help="Update individual ticker features"
    ),
    update_market: bool = typer.Option(
        True,
        "--update-market/--no-update-market",
        help="Update market-wide features"
    ),
    update_sector_mapping: bool = typer.Option(
        False,
        "--update-sector-mapping/--no-update-sector-mapping",
        help="Force update sector mapping (default: false)"
    ),
    validate: bool = typer.Option(
        False,
        "--validate/--no-validate",
        help="Run data validation after updates (default: false)"
    )
):
    """Update ticker and market features.
    
    If no tickers file is provided, defaults to updating all S&P 500 tickers.
    TICKERS_INPUT can be:
    - A path to a file with tickers (one per line)
    - 'all' to update all S&P 500 tickers
    - Omitted to default to 'all'"""
    ticker_list = load_tickers(tickers_input)
    features_dir = Path(features_dir)
    features_dir.mkdir(exist_ok=True)
    data_path = Path("data")
    
    # Update ticker features if requested
    if update_tickers:
        logger.info("Updating ticker features...")
        failed_tickers = []
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn()) as progress:
            task = progress.add_task("Updating ticker features...", total=len(ticker_list))
            for ticker in ticker_list:
                norm_ticker = normalize_ticker(ticker)
                features_path = features_dir / f"{ticker}_features.parquet"
                df_before = pd.read_parquet(features_path) if features_path.exists() else pd.DataFrame()
                df_after = download_stock_data(norm_ticker, history_days=days, features_dir=str(features_dir))
                num_new_rows = len(df_after) - len(df_before)
                if df_after.empty:
                    logger.warning(f"No data for {ticker}")
                    failed_tickers.append(ticker)
                else:
                    if num_new_rows > 0:
                        logger.info(f"Downloaded {num_new_rows} rows for {ticker}")
                    else:
                        logger.info(f"No new rows downloaded for {ticker}")
                    logger.info(f"Computed features for {ticker} and saved to {features_dir}/{ticker}_features.parquet")
                progress.update(task, advance=1)
        if failed_tickers:
            print("\nThe following tickers failed to download data:")
            for t in failed_tickers:
                print(f"  - {t}")
    
    # Update market features if requested
    if update_market:
        logger.info("Updating market features...")
        market_features = MarketFeatures(data_path)
        market_features.generate_market_features(days=days, force_update_sector_mapping=update_sector_mapping)
    
    # Run validation if requested
    if validate:
        logger.info("Running data validation...")
        from scripts.validate_data import validate_all
        if not validate_all():
            logger.error("Data validation failed")
            raise typer.Exit(1)

@app.command()
def run_model(
    model_name: str = typer.Option(..., help="Name of the model to run (e.g., 'TechnicalScorer')"),
    tickers: str = typer.Option("all", help="Comma-separated tickers, path to file, or 'all' for all tickers"),
    date: str = typer.Option(None, help="Run model for this date (YYYY-MM-DD, defaults to latest)"),
    features_dir: str = typer.Option("data/ticker_features", help="Directory with feature Parquet files"),
    model_outputs_dir: str = typer.Option("model_outputs", help="Directory to save model outputs"),
    force: bool = typer.Option(False, help="Overwrite existing outputs if they exist")
):
    """Run a trading model on specified tickers and save outputs."""
    from trading_advisor.models import registry, ModelRunner
    from trading_advisor.data import load_ticker_features
    
    # Check if model exists
    if model_name not in registry.list_models():
        typer.echo(f"Error: Model '{model_name}' not found")
        typer.echo(f"Available models: {', '.join(registry.list_models().keys())}")
        raise typer.Exit(1)
    
    # Get tickers
    if tickers == "all":
        # Get all tickers from features directory
        tickers = [f.stem.replace("_features", "") for f in Path(features_dir).glob("*_features.parquet")]
    elif os.path.isfile(tickers):
        # Read tickers from file
        with open(tickers) as f:
            tickers = [line.strip() for line in f if line.strip()]
    else:
        # Parse comma-separated list
        tickers = [t.strip() for t in tickers.split(",")]
    
    # Load features
    features_df = load_ticker_features(tickers, features_dir)
    if features_df.empty:
        typer.echo("Error: No features found for any tickers")
        raise typer.Exit(1)
    
    # Create model runner
    runner = ModelRunner(output_dir=model_outputs_dir)
    
    # Run model
    typer.echo(f"Running {model_name} on {len(tickers)} tickers...")
    results = runner.run_model(model_name, tickers, features_df, date)
    
    # Print summary
    typer.echo(f"\nModel run complete:")
    typer.echo(f"- Processed {len(results)} tickers")
    typer.echo(f"- Outputs saved to {model_outputs_dir}/{model_name}/")
    
    # Print top 5 scores if available
    if results:
        scores = []
        for ticker, pred in results.items():
            if 'score' in pred:
                score_val = pred['score']
                # If score is a numpy array or list, use the last value as the most recent score
                if isinstance(score_val, (np.ndarray, list)):
                    score_val = float(score_val[-1])
                else:
                    score_val = float(score_val)
                scores.append((ticker, score_val))
        if scores:
            scores.sort(key=lambda x: x[1], reverse=True)
            typer.echo("\nTop 5 scores:")
            for ticker, score in scores[:5]:
                typer.echo(f"- {ticker}: {score:.2f}")

def to_serializable(val):
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    elif isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    elif isinstance(val, (np.ndarray,)):
        return val.tolist()
    elif isinstance(val, (pd.Timestamp, pd.Timedelta)):
        return str(val)
    elif isinstance(val, dict):
        return {k: to_serializable(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [to_serializable(v) for v in val]
    else:
        return val

@app.command()
def generate_report(
    model_outputs_dir: Path = typer.Argument(..., help="Directory containing model output Parquet files"),
    top_n: int = typer.Option(6, help="Number of top tickers to include in the report"),
    output: str = typer.Option("reports", help="Directory to save report files and Parquet table"),
    force: bool = typer.Option(False, help="Overwrite existing report if it exists"),
    positions_csv: str = typer.Option(None, help="CSV file of current positions to always include in the report")
):
    """
    Generate a Markdown report for the top-N tickers by score for a given date, using model outputs from a directory.
    The date is automatically determined from the directory name (e.g., 'model_outputs/technical/2025-06-03').
    Save as both a Markdown file and a row in a Parquet table for historical tracking.
    Optionally, always include tickers from a positions CSV, marked as current positions.
    """
    import os
    import pandas as pd
    import numpy as np
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    import json

    logger = logging.getLogger("trading_advisor.generate_report")

    # Get date from directory name
    try:
        date = model_outputs_dir.name
        # Validate date format
        pd.to_datetime(date)
    except (ValueError, AttributeError):
        typer.echo(f"Error: Directory name '{model_outputs_dir.name}' is not a valid date (YYYY-MM-DD)", err=True)
        raise typer.Exit(1)

    # Get model name from parent directory
    model_name = model_outputs_dir.parent.name

    # Parse positions CSV if provided
    positions_set = set()
    if positions_csv:
        try:
            pos_df = pd.read_csv(positions_csv, skiprows=2)
            if "Symbol" in pos_df.columns:
                # Only keep rows where Symbol looks like a ticker (A-Z, 0-9, no spaces)
                valid = pos_df["Symbol"].astype(str).str.match(r"^[A-Z0-9\.\-]+$")
                positions_set = set(pos_df.loc[valid, "Symbol"].dropna().astype(str).str.strip())
                print(f"[DEBUG] Parsed positions_set: {positions_set}")
            else:
                print("[DEBUG] 'Symbol' column not found in positions CSV.")
        except Exception as e:
            typer.echo(f"Error reading positions CSV: {e}", err=True)
            raise typer.Exit(1)

    # Create output directory
    os.makedirs(output, exist_ok=True)
    report_md_path = os.path.join(output, f"{model_name}_{date}.md")
    report_parquet_path = os.path.join(output, f"{model_name}.parquet")

    # Check if report already exists
    if os.path.exists(report_md_path) and not force:
        typer.echo(f"Report already exists: {report_md_path}. Use --force to overwrite.", err=True)
        raise typer.Exit(1)

    # Load existing report table if it exists
    if os.path.exists(report_parquet_path):
        report_df = pd.read_parquet(report_parquet_path)
    else:
        report_df = pd.DataFrame()

    # Helper function to extract scalar from numpy array
    def get_scalar(val):
        if isinstance(val, np.ndarray):
            return float(val.item())
        return float(val)

    # Collect data for the report
    rows = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn()) as progress:
        task = progress.add_task(f"Gathering data for {date}...", total=len(list(model_outputs_dir.glob("*.parquet"))))
        for parquet_file in model_outputs_dir.glob("*.parquet"):
            ticker = parquet_file.stem
            try:
                df = pd.read_parquet(parquet_file)
                if date not in df['date'].values:
                    logger.warning(f"No data for {ticker} on {date}")
                    progress.update(task, advance=1)
                    continue
                row = df[df['date'] == date].iloc[0]
                # Convert numpy arrays to scalars in details
                details = row.get("details", {})
                if details:
                    details = {
                        k: get_scalar(v) if isinstance(v, np.ndarray) else v
                        for k, v in details.items()
                    }
                rows.append({
                    "ticker": ticker,
                    "score": get_scalar(row.get("score", None)),
                    "details": details,
                    "is_position": ticker in positions_set
                })
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
            progress.update(task, advance=1)

    # Rank by score and select top N (positions do not count toward top_n)
    rows = [r for r in rows if r["score"] is not None]
    # Separate positions and non-positions
    positions_rows = [r for r in rows if r["is_position"]]
    non_position_rows = [r for r in rows if not r["is_position"]]
    # Sort non-positions by score and take top_n
    top_non_positions = sorted(non_position_rows, key=lambda x: x["score"], reverse=True)[:top_n]
    # Add all positions not already in top_n
    top_rows = top_non_positions.copy()
    top_tickers_set = {r["ticker"] for r in top_rows}
    for r in positions_rows:
        if r["ticker"] not in top_tickers_set:
            top_rows.append(r)

    # Generate Markdown report
    from datetime import datetime as dt
    md_lines = [
        f"# Trading Advisor Report\n",
        f"Generated on: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    ]
    # Current Positions section
    if positions_rows:
        md_lines.append(f"## Current Positions\n")
        for r in positions_rows:
            md_lines.append(f"### {r['ticker']}")
            md_lines.append(f"**Technical Score:** {r['score']:.2f}/10")
            if r['details']:
                md_lines.append(f"**Technical Details:**")
                for key, value in r['details'].items():
                    md_lines.append(f"- {key}: {value:.2f}")
            md_lines.append("")
    # New Technical Picks section
    if top_non_positions:
        md_lines.append(f"## New Technical Picks (Top {top_n}) for {date}\n")
        for r in top_non_positions:
            md_lines.append(f"### {r['ticker']}")
            md_lines.append(f"**Technical Score:** {r['score']:.2f}/10")
            if r['details']:
                md_lines.append(f"**Technical Details:**")
                for key, value in r['details'].items():
                    md_lines.append(f"- {key}: {value:.2f}")
            md_lines.append("")

    # Write Markdown report
    with open(report_md_path, "w") as f:
        f.write("\n".join(md_lines))
    typer.echo(f"Markdown report written to {report_md_path}")

    # Save to Parquet table for historical tracking
    new_row = {
        "date": date,
        "top_tickers": [r["ticker"] for r in top_rows],
        "scores": [r["score"] for r in top_rows],
        "report_md": "\n".join(md_lines)
    }
    if not report_df.empty:
        # Remove any existing row for this date
        report_df = report_df[report_df["date"] != date]
        report_df = pd.concat([report_df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        report_df = pd.DataFrame([new_row])
    report_df = report_df.sort_values("date").reset_index(drop=True)
    report_df.to_parquet(report_parquet_path)
    typer.echo(f"Report row saved to {report_parquet_path}")

@app.command()
def generate_dataset(
    tickers: str = typer.Option("all", help="Comma-separated tickers, path to file, or 'all' for all tickers"),
    start_date: str = typer.Option(..., help="Start date for dataset (YYYY-MM-DD)"),
    end_date: str = typer.Option(..., help="End date for dataset (YYYY-MM-DD)"),
    target_days: int = typer.Option(5, help="Number of days to look ahead for target"),
    target_return: float = typer.Option(0.02, help="Target return threshold"),
    train_months: int = typer.Option(3, help="Number of months for training"),
    val_months: int = typer.Option(1, help="Number of months for validation"),
    test_months: int = typer.Option(1, help="Number of months for testing"),
    min_samples: int = typer.Option(10, help="Minimum number of samples required"),
    output: str = typer.Option("data/ml_datasets", help="Directory to save output files"),
    feature_config: Optional[str] = typer.Option(None, help="Path to feature configuration JSON file"),
    force: bool = typer.Option(False, help="Overwrite existing files if they exist"),
    log_level: str = typer.Option("WARNING", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
):
    """Generate machine learning datasets for specified tickers."""
    from trading_advisor.models.dataset import DatasetGenerator
    from trading_advisor.data import load_tickers
    import logging
    
    # Set up logging
    logging.basicConfig(level=getattr(logging, log_level))
    
    # Parse tickers
    if tickers == "all":
        ticker_list = load_tickers("all")
    elif os.path.isfile(tickers):
        with open(tickers) as f:
            ticker_list = [line.strip() for line in f if line.strip()]
    else:
        ticker_list = [t.strip() for t in tickers.split(",")]
    
    # Initialize dataset generator
    generator = DatasetGenerator(
        market_features_dir="data/market_features",
        ticker_features_dir="data/ticker_features",
        output_dir=output,
        feature_config=feature_config
    )
    
    # Generate dataset
    generator.generate_dataset(
        tickers=ticker_list,
        start_date=start_date,
        end_date=end_date,
        target_days=target_days,
        target_return=target_return,
        train_months=train_months,
        val_months=val_months,
        test_months=test_months,
        min_samples=min_samples,
        output=output,
        force=force
    )

def run():
    """Run the CLI application."""
    app() 