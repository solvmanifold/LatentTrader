"""Command-line interface functionality."""

import logging
import sys
from pathlib import Path
from typing import Optional, List
import json
from datetime import datetime, timedelta
import pandas as pd
import os
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import typer
from rich.console import Console
import joblib

from trading_advisor import __version__
from trading_advisor.data import load_tickers, normalize_ticker, download_stock_data, load_ticker_features, ensure_data_dir, load_positions
from trading_advisor.output import generate_report, generate_research_prompt, generate_deep_research_prompt, generate_structured_data, generate_technical_summary, save_json_report
from trading_advisor.market_features import MarketFeatures
from trading_advisor.dataset_v2 import DatasetGeneratorV2
from trading_advisor.models import registry, ModelRunner
from trading_advisor.analysis import analyze_stock, calculate_technical_indicators, calculate_score
from trading_advisor.config import SCORE_WEIGHTS
from trading_advisor.backtest import run_backtest
from trading_advisor.features import update_features as update_stock_features
from trading_advisor.market_breadth import calculate_market_breadth
from trading_advisor.sector_performance import calculate_sector_performance
from trading_advisor.sentiment import MarketSentiment
from trading_advisor.utils import setup_logging
from trading_advisor.ml_data import prepare_ml_datasets, generate_swing_trade_labels
from trading_advisor.normalization import FeatureNormalizer
from .models.sklearn_models.logistic import LogisticRegressionModel

# Set up logging
setup_logging()
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

def get_available_models() -> List[str]:
    """Get list of available model names."""
    return list(registry.list_models().keys())

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
    model_name: str = typer.Option("logistic", help="Model to run (e.g., 'logistic')"),
    tickers: str = typer.Option("all", help="Comma-separated list of tickers or 'all' for all tickers"),
    date: str = typer.Option(None, help="Date to analyze (YYYY-MM-DD) or date range (YYYY-MM-DD:YYYY-MM-DD)"),
    output_dir: str = typer.Option("model_outputs", help="Directory to save model outputs"),
    model_dir: str = typer.Option("models", help="Directory containing trained models")
):
    """Run a model on specified tickers and date(s)."""
    try:
        # Parse tickers
        if tickers == "all":
            ticker_list = load_tickers("all")
        else:
            ticker_list = [t.strip() for t in tickers.split(",")]
            
        # Parse date(s)
        if date:
            if ":" in date:
                start_date, end_date = date.split(":")
                dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
            else:
                dates = [pd.to_datetime(date)]
        else:
            dates = [pd.to_datetime(datetime.now().date())]
            
        # Create output directory
        output_path = Path(output_dir) / model_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize dataset generator
        generator = DatasetGeneratorV2(
            market_features_dir="data/market_features",
            ticker_features_dir="data/ticker_features"
        )
        
        # Load model and metadata
        model_path = Path(model_dir) / model_name / "model"
        metadata_path = Path(model_dir) / model_name / "model.metadata"
        
        if not model_path.exists():
            raise ValueError(f"Model not found at {model_path}")
        if not metadata_path.exists():
            raise ValueError(f"Model metadata not found at {metadata_path}")
            
        # Load model and metadata
        model = joblib.load(model_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Load fitted normalizer
        normalizer = FeatureNormalizer.load(version="1.0.0")
        
        # Process each date
        for target_date in dates:
            logger.info(f"Processing date: {target_date.strftime('%Y-%m-%d')}")
            
            # Load features for all tickers
            features = []
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn()) as progress:
                task = progress.add_task("Loading features...", total=len(ticker_list))
                
                for ticker in ticker_list:
                    try:
                        # Prepare inference data
                        inference_data = generator.prepare_inference_data(ticker, target_date)
                        if inference_data is not None:
                            features.append(inference_data)
                    except Exception as e:
                        logger.warning(f"Error loading features for {ticker}: {e}")
                    progress.update(task, advance=1)
            
            if not features:
                logger.warning(f"No features available for {target_date.strftime('%Y-%m-%d')}")
                continue
                
            # Combine features into DataFrame
            df = pd.concat(features, axis=0)
            
            # Ensure all required features are present
            required_features = metadata['feature_columns']
            missing_features = set(required_features) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Select only the required features in the correct order
            X = df[required_features]
            
            # Normalize features
            X_normalized = normalizer.transform(X)
            
            # Make predictions
            predictions = model.predict(X_normalized)
            probabilities = model.predict_proba(X_normalized)
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'ticker': df['ticker'],
                'date': target_date.strftime('%Y-%m-%d'),
                'prediction': predictions,
                'probability_long': probabilities[:, 1],
                'probability_short': probabilities[:, 2],
                'probability_neutral': probabilities[:, 0]
            })
            
            # Map predictions to labels
            label_map = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}
            results_df['signal'] = results_df['prediction'].map(label_map)
            
            # Save to parquet
            output_file = output_path / f"{target_date.strftime('%Y-%m-%d')}.parquet"
            results_df.to_parquet(output_file)
            logger.info(f"Results saved to {output_file}")
            
            # Generate report
            report_path = output_path / f"{target_date.strftime('%Y-%m-%d')}.md"
            with open(report_path, "w") as f:
                f.write(f"# {model_name} Analysis Report\n")
                f.write(f"Date: {target_date.strftime('%Y-%m-%d')}\n\n")
                
                # Add model metadata
                f.write("## Model Information\n")
                f.write(f"Model Type: {metadata.get('model_name', 'Unknown')}\n")
                f.write(f"Target Column: {metadata.get('target_column', 'Unknown')}\n\n")
                
                # Add predictions
                f.write("## Predictions\n")
                
                # Long signals
                f.write("### Long Signals\n")
                long_signals = results_df[results_df['signal'] == 'LONG'].sort_values('probability_long', ascending=False)
                for _, row in long_signals.iterrows():
                    f.write(f"- {row['ticker']}: {row['probability_long']:.2%}\n")
                
                # Short signals
                f.write("\n### Short Signals\n")
                short_signals = results_df[results_df['signal'] == 'SHORT'].sort_values('probability_short', ascending=False)
                for _, row in short_signals.iterrows():
                    f.write(f"- {row['ticker']}: {row['probability_short']:.2%}\n")
                
                # Neutral signals
                f.write("\n### Neutral Signals\n")
                neutral_signals = results_df[results_df['signal'] == 'NEUTRAL'].sort_values('probability_neutral', ascending=False)
                for _, row in neutral_signals.iterrows():
                    f.write(f"- {row['ticker']}: {row['probability_neutral']:.2%}\n")
            
            logger.info(f"Report generated: {report_path}")
        
    except Exception as e:
        logger.error(f"Error running model: {str(e)}")
        raise typer.Exit(1)

@app.command()
def list_models():
    """List available models."""
    setup_logging()
    
    models = registry.list_models()
    for name, path in models.items():
        if path is not None:
            print(f"{name}: {path}")
        else:
            print(f"{name}: No saved model")

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
def generate_dataset(
    tickers: str = typer.Option("all", help="Comma-separated tickers, path to file, or 'all' for all tickers"),
    start_date: str = typer.Option(..., help="Start date for dataset (YYYY-MM-DD)"),
    end_date: str = typer.Option(..., help="End date for dataset (YYYY-MM-DD)"),
    output_dir: str = typer.Option("data/ml_datasets", help="Directory to save output files"),
    data_dir: str = typer.Option("data", help="Directory containing feature files"),
    train_months: int = typer.Option(6, help="Number of months to use for training"),
    val_months: int = typer.Option(2, help="Number of months to use for validation"),
    min_samples_per_ticker: int = typer.Option(30, help="Minimum number of trading days required per ticker in each split"),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Run data validation after generation (default: True). Validates ticker distribution, date ranges, missing values, and feature statistics."
    )
):
    """Generate ML datasets from ticker features."""
    # Parse tickers
    if tickers == "all":
        ticker_list = load_tickers("all")
    elif os.path.isfile(tickers):
        with open(tickers) as f:
            ticker_list = [line.strip() for line in f if line.strip()]
    else:
        ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize dataset generator
    generator = DatasetGeneratorV2(
        market_features_dir=os.path.join(data_dir, "market_features"),
        ticker_features_dir=os.path.join(data_dir, "ticker_features"),
        output_dir=output_dir,
        train_months=train_months,
        val_months=val_months,
        min_samples_per_ticker=min_samples_per_ticker
    )

    # Generate datasets with progress bar
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn()) as progress:
        task = progress.add_task("Generating dataset...", total=len(ticker_list))
        
        try:
            # Generate datasets
            generator.generate_dataset(
                tickers=ticker_list,
                start_date=start_date,
                end_date=end_date,
                output_dir=Path(output_dir),
                validate=validate,
                progress_callback=lambda x: progress.update(task, completed=x)
            )
            typer.echo(f"\nDataset generated successfully at {output_dir}")
            
        except ValueError as e:
            progress.stop()
            typer.echo(f"\nError: {str(e)}", err=True)
            if "Ticker distribution is not consistent" in str(e):
                typer.echo("\nThis error occurs when some tickers don't have enough data in all splits.")
                typer.echo("Try adjusting the split periods or minimum samples requirement:")
                typer.echo(f"- Current training period: {train_months} months")
                typer.echo(f"- Current validation period: {val_months} months")
                typer.echo(f"- Current minimum samples per ticker: {min_samples_per_ticker}")
            raise typer.Exit(1)
        except Exception as e:
            progress.stop()
            typer.echo(f"\nUnexpected error: {str(e)}", err=True)
            raise typer.Exit(1)

@app.command()
def generate_labels(
    input_dir: str = typer.Option("data/ml_datasets", help="Directory containing train/val/test parquet files from generate-dataset"),
    output_dir: str = typer.Option("data/ml_datasets", help="Directory to save labeled datasets"),
    lookback_days: int = typer.Option(3, help="Number of days to look back for trend confirmation"),
    forward_days: int = typer.Option(5, help="Number of days to look forward for profit target"),
    min_return: float = typer.Option(0.02, help="Minimum return required for a trade (e.g., 0.02 for 2%)"),
    max_drawdown: float = typer.Option(0.01, help="Maximum allowed drawdown (e.g., 0.01 for 1%)")
):
    """Generate swing trade labels for ML datasets.
    
    This command:
    1. Loads train/val/test datasets from the input directory (output of generate-dataset)
    2. Generates swing trade labels based on:
       - {lookback_days}-day lookback for trend confirmation
       - {forward_days}-day forward for profit target
       - {max_drawdown} maximum drawdown for risk management
    3. Saves the labeled datasets to the output directory
    
    Labels:
    - 1: Long signal (uptrend + {min_return} profit target + controlled drawdown)
    - -1: Short signal (downtrend + {min_return} profit target + controlled drawdown)
    - 0: No trade (default)
    """
    # Load datasets
    typer.echo("Loading datasets...")
    train_df = pd.read_parquet(Path(input_dir) / "train.parquet")
    val_df = pd.read_parquet(Path(input_dir) / "val.parquet")
    test_df = pd.read_parquet(Path(input_dir) / "test.parquet")
    
    # Generate labels for each dataset
    typer.echo("\nGenerating swing trade labels...")
    
    # Generate labels for each dataset with parameters
    train_labels = generate_swing_trade_labels(train_df, 
                                             lookback_days=lookback_days,
                                             forward_days=forward_days,
                                             min_return=min_return,
                                             max_drawdown=max_drawdown)
    val_labels = generate_swing_trade_labels(val_df,
                                           lookback_days=lookback_days,
                                           forward_days=forward_days,
                                           min_return=min_return,
                                           max_drawdown=max_drawdown)
    test_labels = generate_swing_trade_labels(test_df,
                                            lookback_days=lookback_days,
                                            forward_days=forward_days,
                                            min_return=min_return,
                                            max_drawdown=max_drawdown)
    
    # Save labels
    label_dir = Path(output_dir) / 'swing_trade'
    label_dir.mkdir(exist_ok=True)
    
    train_labels.to_parquet(label_dir / "train_labels.parquet")
    val_labels.to_parquet(label_dir / "val_labels.parquet")
    test_labels.to_parquet(label_dir / "test_labels.parquet")
    
    # Generate README
    readme_content = f"""# Swing Trade Labels

This directory contains swing trade labels generated using the following command:

```bash
trading-advisor generate-labels \\
    --input-dir {input_dir} \\
    --output-dir {output_dir} \\
    --lookback-days {lookback_days} \\
    --forward-days {forward_days} \\
    --min-return {min_return} \\
    --max-drawdown {max_drawdown}
```

## Label Generation Strategy

The labels are generated using a swing trading strategy with the following parameters:

- Lookback Period: {lookback_days} days (for trend confirmation)
- Forward Period: {forward_days} days (for profit target)
- Minimum Return: {min_return*100:.1f}% (required for a trade)
- Maximum Drawdown: {max_drawdown*100:.1f}% (risk management)

## Label Values

- 1: Long signal (uptrend + {min_return*100:.1f}% profit target + controlled drawdown)
- -1: Short signal (downtrend + {min_return*100:.1f}% profit target + controlled drawdown)
- 0: No trade (default)

## Files

- `train_labels.parquet`: Labels for training set
- `val_labels.parquet`: Labels for validation set
- `test_labels.parquet`: Labels for test set

## Label Distribution

"""
    
    # Add label distribution to README
    for name, labels in [("Train", train_labels), ("Validation", val_labels), ("Test", test_labels)]:
        dist = labels["label"].value_counts(normalize=True)
        readme_content += f"\n### {name} Set\n"
        for label, pct in dist.items():
            label_name = 'Long' if label == 1 else ('Short' if label == -1 else 'No Trade')
            readme_content += f"- {label_name}: {pct:.2%}\n"
    
    # Write README
    with open(label_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    # Print label distribution to stdout
    typer.echo("\nLabel Distribution:")
    for name, labels in [("Train", train_labels), ("Validation", val_labels), ("Test", test_labels)]:
        dist = labels["label"].value_counts(normalize=True)
        typer.echo(f"\n{name} set:")
        for label, pct in dist.items():
            label_name = 'Long' if label == 1 else ('Short' if label == -1 else 'No Trade')
            typer.echo(f"  {label_name}: {pct:.2%}")
    
    typer.echo(f"\nLabels generated successfully in {output_dir}")

def run():
    """Run the CLI application."""
    app() 