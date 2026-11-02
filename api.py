import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv('QUANT_RISK_DB', ROOT / 'quant_risk.db'))
MARKETS = {'NIFTY50', 'SP500'}

app = FastAPI(title='QuantRisk AI API', version='1.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['GET'],
    allow_headers=['*'],
)


class PredictionResponse(BaseModel):
    ticker: str
    signal: str = Field(pattern='^(BUY|SELL|NEUTRAL)$')
    latest_prediction: float
    baseline_mdd: float
    custom_mdd: float
    baseline_sortino: float
    custom_sortino: float


class ReturnPoint(BaseModel):
    date: str
    baseline_cumulative: float
    custom_cumulative: float


class PredictionPoint(BaseModel):
    date: str
    actual_return: float
    custom_prediction: float


class MetricsResponse(BaseModel):
    ticker: str
    baseline_mdd: float
    custom_mdd: float
    baseline_sortino: float
    custom_sortino: float
    baseline_sharpe: float
    custom_sharpe: float
    baseline_var_95: float
    custom_var_95: float
    baseline_cvar_95: float
    custom_cvar_95: float
    mdd_improvement_pct: float
    sortino_improvement_pct: float
    sharpe_improvement_pct: float


class PredictionsResponse(BaseModel):
    ticker: str
    predictions: list[PredictionPoint]


def validate_ticker(ticker):
    key = ticker.upper()
    if key not in MARKETS:
        raise HTTPException(status_code=404, detail=f'Unsupported ticker: {ticker}')
    return key


def load_market(ticker):
    if not DB_PATH.exists():
        raise HTTPException(status_code=503, detail=f'Database not found: {DB_PATH}')
    with sqlite3.connect(DB_PATH) as connection:
        base = pd.read_sql_query(
            f'SELECT Date, Actual, Pred FROM baseline_{ticker}', connection
        )
        custom = pd.read_sql_query(
            f'SELECT Date, Actual, Pred FROM custom_{ticker}', connection
        )
    if base.empty or custom.empty:
        raise HTTPException(status_code=404, detail=f'No predictions for {ticker}')
    return base, custom


def mdd(returns):
    cumulative = np.exp(pd.Series(returns).cumsum())
    return float(((cumulative - cumulative.cummax()) / cumulative.cummax()).min())


def sortino(returns, mar=0.0):
    values = pd.Series(returns)
    downside = np.minimum(0.0, values - mar)
    deviation = np.sqrt(np.mean(np.square(downside)))
    return 0.0 if deviation == 0 else float(
        (values.mean() - mar) / deviation * np.sqrt(252)
    )


def sharpe(returns):
    values = pd.Series(returns)
    deviation = values.std()
    return 0.0 if deviation == 0 else float(
        values.mean() / deviation * np.sqrt(252)
    )


def var_cvar(returns):
    values = np.asarray(returns, dtype=float)
    var_95 = float(np.percentile(values, 5))
    tail = values[values <= var_95]
    cvar_95 = float(tail.mean()) if tail.size else var_95
    return var_95, cvar_95


@app.get('/api/predict/{ticker}', response_model=PredictionResponse)
def predict(ticker):
    ticker = validate_ticker(ticker)
    base, custom = load_market(ticker)
    custom_strategy = np.where(custom['Pred'] > 0, custom['Actual'], 0.0)
    base_strategy = np.where(base['Pred'] > 0, base['Actual'], 0.0)
    latest = float(custom['Pred'].iloc[-1])
    signal = 'BUY' if latest > 0 else 'SELL' if latest < 0 else 'NEUTRAL'
    return PredictionResponse(
        ticker=ticker,
        signal=signal,
        latest_prediction=latest,
        baseline_mdd=mdd(base_strategy),
        custom_mdd=mdd(custom_strategy),
        baseline_sortino=sortino(base_strategy),
        custom_sortino=sortino(custom_strategy),
    )


@app.get('/api/metrics/{ticker}', response_model=MetricsResponse)
def metrics(ticker):
    ticker = validate_ticker(ticker)
    base, custom = load_market(ticker)
    merged = base[['Date', 'Actual', 'Pred']].merge(
        custom[['Date', 'Actual', 'Pred']],
        on='Date',
        suffixes=('_base', '_custom'),
    )
    base_returns = np.where(merged['Pred_base'] > 0, merged['Actual_base'], 0.0)
    custom_returns = np.where(merged['Pred_custom'] > 0, merged['Actual_custom'], 0.0)
    baseline_mdd = mdd(base_returns)
    custom_mdd = mdd(custom_returns)
    baseline_sortino = sortino(base_returns)
    custom_sortino = sortino(custom_returns)
    baseline_sharpe = sharpe(base_returns)
    custom_sharpe = sharpe(custom_returns)
    baseline_var_95, baseline_cvar_95 = var_cvar(base_returns)
    custom_var_95, custom_cvar_95 = var_cvar(custom_returns)
    return MetricsResponse(
        ticker=ticker,
        baseline_mdd=baseline_mdd,
        custom_mdd=custom_mdd,
        baseline_sortino=baseline_sortino,
        custom_sortino=custom_sortino,
        baseline_sharpe=baseline_sharpe,
        custom_sharpe=custom_sharpe,
        baseline_var_95=baseline_var_95,
        custom_var_95=custom_var_95,
        baseline_cvar_95=baseline_cvar_95,
        custom_cvar_95=custom_cvar_95,
        mdd_improvement_pct=(1 - custom_mdd / baseline_mdd) * 100,
        sortino_improvement_pct=(
            (custom_sortino / baseline_sortino - 1) * 100
            if baseline_sortino != 0 else 0.0
        ),
        sharpe_improvement_pct=(
            (custom_sharpe / baseline_sharpe - 1) * 100
            if baseline_sharpe != 0 else 0.0
        ),
    )


@app.get('/api/predictions/{ticker}', response_model=PredictionsResponse)
def predictions(ticker):
    ticker = validate_ticker(ticker)
    _, custom = load_market(ticker)
    points = [
        PredictionPoint(
            date=str(row.Date),
            actual_return=float(row.Actual),
            custom_prediction=float(row.Pred),
        )
        for row in custom.itertuples()
    ]
    return PredictionsResponse(ticker=ticker, predictions=points)
