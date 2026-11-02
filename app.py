import importlib.util
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api import metrics as api_metrics


ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv('QUANT_RISK_DB', ROOT / 'quant_risk.db'))
MARKETS = {'NIFTY 50': 'NIFTY50', 'S&P 500': 'SP500'}
MIDNIGHT = '#030407'
COBALT = '#1558C9'
LIME = '#D4FF24'

st.set_page_config(
    page_title='QuantRisk AI — Institutional Grade Analytics',
    page_icon='📈',
    layout='wide',
)
st.markdown(
    f"""
    <style>
    .stApp {{ background: {MIDNIGHT}; color: #F4F7FF; }}
    [data-testid="stMetric"] {{
        background: rgba(21, 88, 201, 0.14);
        border: 1px solid rgba(212, 255, 36, 0.28);
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 8px 24px rgba(0,0,0,.24);
    }}
    [data-testid="stMetricLabel"] {{ color: #AAB7D4; }}
    [data-testid="stMetricValue"] {{ color: {LIME}; }}
    .quant-nav {{
        border-bottom: 1px solid rgba(212,255,36,.25);
        padding: 10px 0 18px;
        margin-bottom: 24px;
        color: {LIME};
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: .03em;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def mdd(values):
    returns = pd.Series(values)
    cumulative = np.exp(returns.cumsum())
    peak = cumulative.cummax()
    return float(((cumulative - peak) / peak).min())


def sortino(values, mar=0.0):
    returns = pd.Series(values)
    downside = np.minimum(0.0, returns - mar)
    deviation = np.sqrt(np.mean(np.square(downside)))
    return 0.0 if deviation == 0 else float(
        (returns.mean() - mar) / deviation * np.sqrt(252)
    )


@st.cache_data
def load_market_data(ticker):
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f'{DB_PATH} does not exist. Run `python scripts\\init_db.py` first.'
        )
    with sqlite3.connect(DB_PATH) as connection:
        base = pd.read_sql_query(
            f'SELECT * FROM baseline_{ticker}', connection, parse_dates=['Date']
        )
        custom = pd.read_sql_query(
            f'SELECT * FROM custom_{ticker}', connection, parse_dates=['Date']
        )
    return base, custom


def load_paper_trader():
    path = ROOT / 'scripts' / '45_paper_trader.py'
    spec = importlib.util.spec_from_file_location('quant_risk_paper_trader', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Unable to load paper trader from {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_daily_simulation


def read_portfolio():
    with sqlite3.connect(DB_PATH) as connection:
        holdings = pd.read_sql_query(
            "SELECT * FROM paper_portfolio WHERE status = 'OPEN'",
            connection,
        )
        summary = pd.read_sql_query(
            'SELECT * FROM portfolio_summary ORDER BY id',
            connection,
        )
    return holdings, summary


def sentiment_summary(ticker_label):
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return 'Neutral — Gemini sentiment is unavailable because GEMINI_API_KEY is not configured.'
    try:
        import google.generativeai as genai
        import yfinance as yf
    except ImportError as error:
        return f'Neutral — optional news dependencies are unavailable: {error}'

    symbol = '^NSEI' if ticker_label == 'NIFTY 50' else '^GSPC'
    news = yf.Ticker(symbol).news[:3]
    headlines = [item.get('title', '') for item in news if item.get('title')]
    if not headlines:
        headlines = [
            f'{ticker_label} market outlook remains under analyst review.',
            'Investors are monitoring inflation and interest-rate expectations.',
            'Risk assets remain sensitive to global growth and earnings guidance.',
        ]
    genai.configure(api_key=api_key)
    prompt = (
        f'Classify {ticker_label} sentiment as Bullish, Bearish, or Neutral. '
        f'Return exactly two concise sentences using these headlines:\n'
        + '\n'.join(f'- {headline}' for headline in headlines)
    )
    model = genai.GenerativeModel('gemini-1.5-flash')
    return model.generate_content(prompt).text


market_label = st.sidebar.selectbox('Market', list(MARKETS))
ticker = MARKETS[market_label]
try:
    base, custom = load_market_data(ticker)
    risk = api_metrics(ticker)
except (FileNotFoundError, sqlite3.Error, pd.errors.DatabaseError) as error:
    st.error(str(error))
    st.stop()

base['Strategy'] = np.where(base['Pred'] > 0, base['Actual'], 0.0)
custom['Strategy'] = np.where(custom['Pred'] > 0, custom['Actual'], 0.0)
base['Cumulative'] = np.exp(base['Strategy'].cumsum())
custom['Cumulative'] = np.exp(custom['Strategy'].cumsum())

st.markdown(
    '<div class="quant-nav">QuantRisk AI — Institutional Grade Analytics</div>',
    unsafe_allow_html=True,
)
st.title(f'{market_label} Risk Terminal')
st.caption('Leakage-free AVP-Loss evaluation | Analytical decision-support only')

live_tab, paper_tab, risk_tab = st.tabs(
    ['Live Dashboard', 'Paper Trading', 'Risk Analytics']
)

with live_tab:
    cards = st.columns(5)
    cards[0].metric('Baseline MDD', f'{risk.baseline_mdd:.2%}')
    cards[1].metric('Custom MDD', f'{risk.custom_mdd:.2%}', f'{risk.mdd_improvement_pct:+.1f}%')
    cards[2].metric('Custom Sortino', f'{risk.custom_sortino:.3f}', f'{risk.sortino_improvement_pct:+.1f}%')
    cards[3].metric('Latest Signal', 'BUY' if custom['Pred'].iloc[-1] > 0 else 'SELL')
    cards[4].metric('Latest Prediction', f'{custom["Pred"].iloc[-1]:.6f}')

    st.subheader('AI Market Sentiment')
    st.info(sentiment_summary(market_label))

    st.subheader('Cumulative Strategy Returns')
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=base['Date'], y=base['Cumulative'], name='Baseline MSE', line={'color': COBALT}))
    figure.add_trace(go.Scatter(x=custom['Date'], y=custom['Cumulative'], name='Custom AVP-Loss', line={'color': LIME}))
    figure.update_layout(height=450, template='plotly_dark', paper_bgcolor=MIDNIGHT, plot_bgcolor=MIDNIGHT)
    st.plotly_chart(figure, width='stretch')

    st.subheader('Drawdown Comparison')
    base_drawdown = (base['Cumulative'] - base['Cumulative'].cummax()) / base['Cumulative'].cummax()
    custom_drawdown = (custom['Cumulative'] - custom['Cumulative'].cummax()) / custom['Cumulative'].cummax()
    drawdown = go.Figure()
    drawdown.add_trace(go.Scatter(x=base['Date'], y=base_drawdown, name='Baseline', line={'color': COBALT}))
    drawdown.add_trace(go.Scatter(x=custom['Date'], y=custom_drawdown, name='Custom AVP-Loss', line={'color': LIME}))
    drawdown.update_layout(height=350, template='plotly_dark', paper_bgcolor=MIDNIGHT, plot_bgcolor=MIDNIGHT, yaxis_tickformat='.1%')
    st.plotly_chart(drawdown, width='stretch')

with paper_tab:
    st.subheader('Live Paper Trading')
    st.caption('Virtual $100,000 portfolio; no broker or real-money orders are placed.')
    if st.button('▶️ Run Daily Simulation'):
        try:
            result = load_paper_trader()()
            st.success(f"Simulation complete: portfolio value ${result['total_value']:,.2f}")
            st.cache_data.clear()
        except (FileNotFoundError, RuntimeError, sqlite3.Error, ValueError) as error:
            st.error(f'Paper simulation failed: {error}')

    try:
        holdings, summary = read_portfolio()
        latest_summary = summary.iloc[-1] if not summary.empty else None
        total_value = latest_summary['total_value'] if latest_summary is not None else 100_000.0
        cash = latest_summary['total_cash'] if latest_summary is not None else 100_000.0
        paper_cards = st.columns(3)
        paper_cards[0].metric('Total Portfolio Value', f'${total_value:,.2f}')
        paper_cards[1].metric('Cash Available', f'${cash:,.2f}')
        paper_cards[2].metric('Active Positions', str(len(holdings)))
        st.dataframe(holdings, width='stretch', hide_index=True)
        if not summary.empty:
            portfolio_chart = go.Figure(go.Scatter(
                x=summary['date'], y=summary['total_value'],
                line={'color': LIME}, name='Portfolio Value',
            ))
            portfolio_chart.update_layout(
                height=350, template='plotly_dark',
                paper_bgcolor=MIDNIGHT, plot_bgcolor=MIDNIGHT,
            )
            st.plotly_chart(portfolio_chart, width='stretch')
    except (sqlite3.Error, pd.errors.DatabaseError) as error:
        st.info(f'Paper-trading tables are unavailable: {error}')

with risk_tab:
    st.subheader('Risk Analytics')
    risk_cards = st.columns(4)
    risk_cards[0].metric('Custom VaR 95%', f'{risk.custom_var_95:.2%}')
    risk_cards[1].metric('Custom CVaR 95%', f'{risk.custom_cvar_95:.2%}')
    risk_cards[2].metric('Custom MDD', f'{risk.custom_mdd:.2%}')
    risk_cards[3].metric('Custom Sharpe', f'{risk.custom_sharpe:.3f}')
    exposure = go.Figure(go.Bar(
        x=['VaR 95%', 'CVaR 95%', 'MDD'],
        y=[abs(risk.custom_var_95), abs(risk.custom_cvar_95), abs(risk.custom_mdd)],
        marker_color=[COBALT, LIME, COBALT],
    ))
    exposure.update_layout(
        title='Downside Risk Exposure',
        yaxis_tickformat='.1%',
        template='plotly_dark',
        paper_bgcolor=MIDNIGHT,
        plot_bgcolor=MIDNIGHT,
    )
    st.plotly_chart(exposure, width='stretch')
    st.caption('VaR and CVaR are calculated from the custom strategy log-return series.')

st.caption('⚠ Analytical decision-support tool only. Not financial advice.')
