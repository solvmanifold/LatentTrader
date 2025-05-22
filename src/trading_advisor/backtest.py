import pandas as pd
from datetime import datetime
from trading_advisor.analysis import calculate_technical_indicators, calculate_score_history
from trading_advisor.data import download_stock_data

def run_backtest(
    tickers,
    start_date,
    end_date,
    top_n=3,
    hold_days=10,
    stop_loss=-0.10,
    profit_target=0.10
):
    """Backtest the strategy using weekly top-N selection and fixed holding period with stop/profit exits."""
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
            df = download_stock_data(ticker, end_date=week_start)
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
            df = download_stock_data(pos['ticker'], end_date=week_start + pd.Timedelta(days=hold_days*2))
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
    summary = {
        'total_closed_trades': len([p for p in portfolio if p['closed']]),
        'total_return': total_return,
        'trade_log': trade_log,
        'equity_curve': equity_curve
    }
    return summary 