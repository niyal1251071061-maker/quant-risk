import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Quantitative Risk Mitigation",
    page_icon="📈",
    layout="wide"
)

# ---------- Load Data ----------
@st.cache_data
def load_data():
    base = pd.read_csv('reports/baseline_results.csv', parse_dates=['Date'])
    cust = pd.read_csv('reports/custom_results.csv', parse_dates=['Date'])
    return base, cust

base, cust = load_data()

# ---------- Metric Functions ----------
def mdd(r):
    c = (1 + r).cumprod(); p = c.cummax()
    return ((c - p) / p).min()

def sortino(r):
    d = r[r < 0]
    return 0 if d.std() == 0 else r.mean() / d.std() * np.sqrt(252)

def rmse(a, p):
    return np.sqrt(((a - p) ** 2).mean())

base['Signal'] = (base['Pred'] > 0).astype(int)
cust['Signal'] = (cust['Pred'] > 0).astype(int)
base['Strategy'] = base['Signal'] * base['Actual']
cust['Strategy'] = cust['Signal'] * cust['Actual']
base['Cumulative'] = (1 + base['Strategy']).cumprod()
cust['Cumulative'] = (1 + cust['Strategy']).cumprod()

# ---------- Header ----------
st.title("Quantitative Risk Mitigation Dashboard")
st.caption("Custom Asymmetric Volatility-Penalized Loss vs Symmetric MSE Baseline — NIFTY 50, 2019–2023")

# ---------- KPI Cards ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Baseline MDD",
    f"{mdd(base['Strategy']):.2%}",
    help="Maximum Drawdown of baseline XGBoost MSE strategy"
)
c2.metric(
    "Custom MDD",
    f"{mdd(cust['Strategy']):.2%}",
    delta=f"{(mdd(cust['Strategy']) - mdd(base['Strategy'])):.2%}",
    delta_color="normal",
    help="Maximum Drawdown of custom AVP-Loss strategy (higher is better)"

)
c3.metric(
    "Custom Sortino",
    f"{sortino(cust['Strategy']):.3f}",
    delta=f"+{(sortino(cust['Strategy']) / sortino(base['Strategy']) - 1) * 100:.1f}%",
    help="Risk-adjusted return (higher is better)"
)
c4.metric(
    "RMSE",
    f"{rmse(cust['Actual'], cust['Pred']):.6f}",
    delta=f"{(rmse(cust['Actual'], cust['Pred']) / rmse(base['Actual'], base['Pred']) - 1) * 100:.1f}%",
    delta_color="inverse"
)

st.divider()

# ---------- Cumulative Returns Chart ----------
st.subheader("Cumulative Strategy Returns")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=base['Date'], y=base['Cumulative'],
    name='Baseline MSE', line=dict(color='#8888ff', width=2)
))
fig.add_trace(go.Scatter(
    x=cust['Date'], y=cust['Cumulative'],
    name='Custom AVP-Loss', line=dict(color='#00C853', width=2)
))
fig.update_layout(
    height=450,
    hovermode='x unified',
    legend=dict(orientation='h', yanchor='bottom', y=1.02),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis_title='Trading Date',
    yaxis_title='Cumulative Return'
)
st.plotly_chart(fig, use_container_width=True)

# ---------- Drawdown Chart ----------
st.subheader("Drawdown Comparison")
fig2 = go.Figure()
base_dd = (base['Cumulative'] - base['Cumulative'].cummax()) / base['Cumulative'].cummax()
cust_dd = (cust['Cumulative'] - cust['Cumulative'].cummax()) / cust['Cumulative'].cummax()
fig2.add_trace(go.Scatter(x=base['Date'], y=base_dd, name='Baseline', fill='tozeroy', line=dict(color='#8888ff')))
fig2.add_trace(go.Scatter(x=cust['Date'], y=cust_dd, name='Custom AVP', fill='tozeroy', line=dict(color='#00C853')))
fig2.update_layout(
    height=350,
    hovermode='x unified',
    yaxis_tickformat='.1%',
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis_title='Trading Date',
    yaxis_title='Drawdown'
)
st.plotly_chart(fig2, use_container_width=True)

# ---------- Sidebar ----------
st.sidebar.header("About This Dashboard")
st.sidebar.markdown("""
**Project:** Quantitative Market Intelligence Platform

**Group:** SY-H1

**Guide:** Prof. Manisha More

**Core Innovation:**
Custom Asymmetric Volatility-Penalized Loss (AVP-Loss) for XGBoost that penalizes over-predictions during high-volatility regimes.

**Key Results:**
- MDD reduced 47%
- Sortino improved 57%
- RMSE +1.8%

**Stack:** Python, XGBoost, Pandas, Streamlit, Plotly
""")

st.sidebar.divider()
st.sidebar.warning("⚠ Analytical decision-support tool only. Not financial advice.")

# ---------- Footer ----------
st.divider()
st.caption("Built with Streamlit and Plotly — Data: Yahoo Finance (daily OHLCV)")