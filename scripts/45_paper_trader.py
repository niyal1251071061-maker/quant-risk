import importlib.util
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / 'quant_risk.db'
INITIAL_CAPITAL = 100_000.0
ALLOCATION = 1 / 3
ASSETS = {
    'NIFTY50': {'label': 'NIFTY 50', 'yahoo': '^NSEI'},
    'SP500': {'label': 'S&P 500', 'yahoo': '^GSPC'},
    'FTSE100': {'label': 'FTSE 100', 'yahoo': '^FTSE'},
}


def _load_alert_sender():
    path = Path(__file__).with_name('44_alert_bot.py')
    spec = importlib.util.spec_from_file_location('quant_risk_alert_bot', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Unable to load alert bot from {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.send_discord_alert


def _latest_market_data(ticker, yahoo_symbol):
    history = yf.download(
        yahoo_symbol,
        period='5d',
        auto_adjust=False,
        multi_level_index=False,
        progress=False,
    )
    close = history['Close'].dropna() if 'Close' in history else pd.Series(dtype=float)
    if close.empty:
        fallback_path = ROOT / 'data' / f'{ticker}_features.parquet'
        if not fallback_path.exists():
            raise RuntimeError(f'No closing price available for {ticker}')
        print(f'Live price unavailable for {ticker}; using latest stored close.')
        fallback = pd.read_parquet(fallback_path)
        close = fallback['Close'].dropna()
        if close.empty:
            raise RuntimeError(f'No closing price available for {ticker}')
    latest_price = float(close.iloc[-1])
    daily_return = float(close.pct_change().dropna().iloc[-1]) if len(close) > 1 else 0.0
    return latest_price, daily_return


def _latest_prediction(connection, ticker):
    table = f'custom_{ticker}'
    row = connection.execute(
        f'SELECT Pred FROM {table} ORDER BY Date DESC LIMIT 1'
    ).fetchone()
    if row is None:
        raise RuntimeError(f'No custom prediction available for {ticker}')
    return float(row[0])


def run_daily_simulation():
    if not DB_PATH.exists():
        raise FileNotFoundError(f'{DB_PATH} does not exist. Run scripts\\init_db.py first.')

    send_alert = _load_alert_sender()
    simulation_date = date.today().isoformat()
    events = []
    with sqlite3.connect(DB_PATH) as connection:
        summary = connection.execute(
            'SELECT total_cash FROM portfolio_summary ORDER BY id DESC LIMIT 1'
        ).fetchone()
        cash = float(summary[0]) if summary else INITIAL_CAPITAL

        for ticker, asset in ASSETS.items():
            price, daily_return = _latest_market_data(ticker, asset['yahoo'])
            prediction = _latest_prediction(connection, ticker)
            signal = 'BUY' if prediction > 0 else 'SELL' if prediction < 0 else 'NEUTRAL'
            open_position = connection.execute(
                """
                SELECT id, shares FROM paper_portfolio
                WHERE ticker = ? AND status = 'OPEN'
                ORDER BY id DESC LIMIT 1
                """,
                (ticker,),
            ).fetchone()

            if signal == 'BUY' and open_position is None:
                allocation = min(INITIAL_CAPITAL * ALLOCATION, cash)
                if allocation > 0:
                    shares = allocation / price
                    connection.execute(
                        """
                        INSERT INTO paper_portfolio
                        (ticker, entry_date, entry_price, shares, capital_allocated, status)
                        VALUES (?, ?, ?, ?, ?, 'OPEN')
                        """,
                        (ticker, simulation_date, price, shares, allocation),
                    )
                    cash -= allocation
                    events.append(f'BUY {asset["label"]}: {shares:.4f} shares at {price:.2f}')
            elif signal == 'SELL' and open_position is not None:
                position_id, shares = open_position
                proceeds = float(shares) * price
                connection.execute(
                    'UPDATE paper_portfolio SET status = ? WHERE id = ?',
                    ('CLOSED', position_id),
                )
                cash += proceeds
                events.append(f'SELL {asset["label"]}: proceeds {proceeds:.2f}')

            if signal == 'BUY' or abs(daily_return) > 0.02:
                events.append(
                    f'{asset["label"]}: signal={signal}, daily return={daily_return:.2%}'
                )

        open_positions = connection.execute(
            "SELECT ticker, shares FROM paper_portfolio WHERE status = 'OPEN'"
        ).fetchall()
        prices = {
            ticker: _latest_market_data(ticker, asset['yahoo'])[0]
            for ticker, asset in ASSETS.items()
        }
        total_value = cash + sum(float(shares) * prices[ticker] for ticker, shares in open_positions)
        connection.execute(
            """
            INSERT INTO portfolio_summary (date, total_cash, total_value)
            VALUES (?, ?, ?)
            """,
            (simulation_date, cash, total_value),
        )
        connection.commit()

    for event in events:
        send_alert(f'QuantRisk AI paper-trading alert: {event}')
    return {
        'date': simulation_date,
        'total_cash': cash,
        'total_value': total_value,
        'events': events,
    }
