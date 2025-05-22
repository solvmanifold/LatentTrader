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
from trading_advisor.analysis import analyze_stock, calculate_technical_indicators, calculate_score_history
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
    tickers: list[str] = typer.Argument(None, help="Stock ticker symbols (can specify multiple)", show_default=False),
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
    ),
    json_file: Optional[Path] = typer.Option(
        None,
        "--json", "-j",
        help="Path to analysis JSON file (from analyze command)"
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
                    df = download_stock_data(ticker, history_days=history_days)
                else:
                    df = download_stock_data(ticker, history_days=history_days)
                    score, score_details, analyst_targets = analyze_stock(ticker, df)

                # After downloading stock data, calculate technical indicators
                df = calculate_technical_indicators(df)

                # Ensure DataFrame has expected columns for overlays
                if 'SMA_20' in df.columns:
                    df['MA20'] = df['SMA_20']
                if 'SMA_50' in df.columns:
                    df['MA50'] = df['SMA_50']
                # BB columns are already named as expected in most cases, but add mapping if needed
                # (If your data source uses different names, add more mappings here)

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

@app.command()
def backtest(
    tickers: list[str] = typer.Argument(..., help="Stock ticker symbols (can specify multiple)"),
    start_date: str = typer.Option(..., help="Backtest start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(..., help="Backtest end date (YYYY-MM-DD)"),
    top_n: int = typer.Option(3, help="Number of top picks to buy each week"),
    hold_days: int = typer.Option(10, help="Max holding period in trading days"),
    stop_loss: float = typer.Option(-0.10, help="Stop-loss threshold (e.g., -0.10 for -10%)"),
    profit_target: float = typer.Option(0.10, help="Profit target threshold (e.g., 0.10 for +10%)")
):
    """Backtest the strategy using weekly top-N selection and fixed holding period with stop/profit exits."""
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    from trading_advisor.analysis import calculate_technical_indicators, calculate_score_history
    from trading_advisor.data import download_stock_data

    # Prepare date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    all_dates = pd.date_range(start=start_dt, end=end_dt, freq='B')
    week_starts = all_dates[all_dates.weekday == 0]  # Mondays
    if len(week_starts) == 0 or week_starts[0] > start_dt:
        week_starts = all_dates[all_dates.weekday == 0 | (all_dates == start_dt)]

    trade_log = []
    equity_curve = []
    portfolio = []  # List of open positions: dicts with ticker, entry_date, entry_price, etc.
    cash = 100000.0  # Start with $100k
    equity = cash
    for week_start in week_starts:
        week_str = week_start.strftime('%Y-%m-%d')
        # 1. For each ticker, get data up to this week
        scores = []
        for ticker in tickers:
            df = download_stock_data(ticker)
            df = df[df.index <= week_start]
            if len(df) < 50:
                continue
            df = calculate_technical_indicators(df)
            scored = calculate_score_history(df)
            if scored.empty:
                continue
            last_row = scored.iloc[-1]
            scores.append((ticker, last_row['score'], last_row['Close']))
        # 2. Select top N
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        picks = scores[:top_n]
        # 3. Simulate buying each pick (if not already held)
        for ticker, score, price in picks:
            if any(p['ticker'] == ticker and not p['closed'] for p in portfolio):
                continue  # Already held
            position = {
                'ticker': ticker,
                'entry_date': week_start,
                'entry_price': price,
                'max_price': price,
                'min_price': price,
                'holding_days': 0,
                'closed': False,
                'exit_date': None,
                'exit_price': None,
                'exit_reason': None
            }
            portfolio.append(position)
        # 4. Update open positions
        for pos in portfolio:
            if pos['closed']:
                continue
            df = download_stock_data(pos['ticker'])
            df = df[(df.index > pos['entry_date']) & (df.index <= pos['entry_date'] + pd.Timedelta(days=hold_days*2))]
            for i, (date, row) in enumerate(df.iterrows()):
                price = row['Close']
                pos['max_price'] = max(pos['max_price'], price)
                pos['min_price'] = min(pos['min_price'], price)
                ret = (price - pos['entry_price']) / pos['entry_price']
                pos['holding_days'] += 1
                if ret <= stop_loss:
                    pos['closed'] = True
                    pos['exit_date'] = date
                    pos['exit_price'] = price
                    pos['exit_reason'] = 'stop_loss'
                    trade_log.append({**pos})
                    break
                elif ret >= profit_target:
                    pos['closed'] = True
                    pos['exit_date'] = date
                    pos['exit_price'] = price
                    pos['exit_reason'] = 'profit_target'
                    trade_log.append({**pos})
                    break
                elif pos['holding_days'] >= hold_days:
                    pos['closed'] = True
                    pos['exit_date'] = date
                    pos['exit_price'] = price
                    pos['exit_reason'] = 'max_hold'
                    trade_log.append({**pos})
                    break
        # 5. Update equity
        open_equity = sum(
            (p['exit_price'] if p['closed'] else p['entry_price']) for p in portfolio if p['entry_date'] <= week_start
        )
        equity_curve.append({'date': week_start, 'equity': open_equity})
    # Output summary
    total_return = (
        sum(p['exit_price'] - p['entry_price'] for p in portfolio if p['closed']) /
        (len([p for p in portfolio if p['closed']]) * picks[0][2]) if picks else 0
    )
    typer.echo(f"Backtest complete. Total closed trades: {len([p for p in portfolio if p['closed']])}")
    typer.echo(f"Total return: {total_return*100:.2f}%")
    typer.echo("Trade log:")
    for trade in trade_log:
        typer.echo(trade)
    # (Plotting and more stats can be added later)

def run():
    """Run the CLI application."""
    app() 