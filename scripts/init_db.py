import re
import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / 'quant_risk.db'
REPORTS = ROOT / 'reports'


def table_name(path):
    name = path.relative_to(REPORTS).with_suffix('').as_posix()
    return re.sub(r'[^A-Za-z0-9_]', '_', name)


def init_database():
    csv_files = sorted(REPORTS.rglob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No report CSV files found under {REPORTS}')

    with sqlite3.connect(DB_PATH) as connection:
        for path in csv_files:
            frame = pd.read_csv(path)
            frame.to_sql(table_name(path), connection, if_exists='replace', index=False)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_portfolio (
                id INTEGER PRIMARY KEY,
                ticker TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                entry_price REAL NOT NULL,
                shares REAL NOT NULL,
                capital_allocated REAL NOT NULL,
                status TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_summary (
                id INTEGER PRIMARY KEY,
                date TEXT NOT NULL,
                total_cash REAL NOT NULL,
                total_value REAL NOT NULL
            )
            """
        )
        connection.commit()
    print(f'Loaded {len(csv_files)} report files into {DB_PATH}')


if __name__ == '__main__':
    init_database()
