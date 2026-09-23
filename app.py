import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Quantitative Risk Mitigation",
    page_icon="📈",
    layout="wide"
)

# ---------- Metric Functions ----------
def mdd(r):
    c = (1 + r).cumprod(); p = c.cummax()
    return ((c - p) / p).min()

def sortino(r):
    d = r[r < 0]
    return 0 if d.std() == 0 else r.mean() / d.std() * np.sqrt(252)

def rmse_fn(a, p):
    return np.sqrt(((a - p) ** 2).mean())

# ---------- Sidebar: Ticker Selector ----------
st.sidebar.header("Market Selection")
ticker_label = st.sidebar.selectbox("Choose Market", ["NIFTY 50", "S&P 500"])
ticker_key = "NIFTY50" if ticker_label == "NIFTY 50" else "SP500"

# ---------- Load Data ----------
@st.cache_data
def load_data(tk):
    base = pd.read_csv(f'reports/baseline_{tk}.csv', parse_dates=['Date'])
    cust = pd.read_csv(f'reports/custom_{tk}.csv', parse_dates=['Date'])
    return base, cust

try:
    base, cust = load_data(ticker_key)
except FileNotFoundError:
    st.error(f"Missing data for {ticker_label}. Run scripts/18_final_results.py first.")
    st.stop()

base['Signal'] = (base['Pred'] > 0).astype(int)
cust['Signal'] = (cust['Pred'] > 0).astype(int)
base['Strategy'] = base['Signal'] * base['Actual']
cust['Strategy'] = cust['Signal'] * cust['Actual']
base['Cumulative'] = (1 + base['Strategy']).cumprod()
cust['Cumulative'] = (1 + cust['Strategy']).cumprod()

# ---------- Compute Improvements ----------
mdd_imp  = (1 - mdd(cust['Strategy']) / mdd(base['Strategy'])) * 100
sort_imp = (sortino(cust['Strategy']) / sortino(base['Strategy']) - 1) * 100
rmse_imp = (1 - rmse_fn(cust['Actual'], cust['Pred']) / rmse_fn(base['Actual'], base['Pred'])) * 100

# ---------- Sidebar: About ----------
st.sidebar.header("About This Dashboard")
st.sidebar.markdown(f"""
**Project:** Quantitative Risk Mitigation Platform  
**Group:** SY-H1  
**Guide:** Prof. Manisha More  

**Core Innovation:**  
Custom Asymmetric Volatility-Penalized Loss (AVP-Loss) for XGBoost

**Method:**  
- Walk-forward validation (5-yr train, 1-yr test)  
- 8 engineered features  
- Evaluated across NIFTY 50 and S&P 500  

**{ticker_label} Results:**  
- MDD improved **{mdd_imp:+.1f}%**  
- Sortino improved **{sort_imp:+.1f}%**  
- RMSE improved **{rmse_imp:+.1f}%**

**Stack:** Python, XGBoost, Pandas, Streamlit, Plotly
""")

st.sidebar.divider()
st.sidebar.warning("⚠ Analytical decision-support tool only. Not financial advice.")

# ---------- Header ----------
st.title("Quantitative Risk Mitigation Dashboard")
st.caption(f"Custom AVP-Loss vs Symmetric MSE Baseline — {ticker_label}, 2019–2023")

# ---------- KPI Cards ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Baseline MDD", f"{mdd(base['Strategy']):.2%}")
c2.metric("Custom MDD", f"{mdd(cust['Strategy']):.2%}", delta=f"{mdd_imp:+.1f}%")
c3.metric("Custom Sortino", f"{sortino(cust['Strategy']):.3f}", delta=f"{sort_imp:+.1f}%")
c4.metric("RMSE", f"{rmse_fn(cust['Actual'], cust['Pred']):.6f}", delta=f"{rmse_imp:+.1f}%")

st.divider()

# ---------- Cumulative Returns ----------
st.subheader("Cumulative Strategy Returns")
fig = go.Figure()
fig.add_trace(go.Scatter(x=base['Date'], y=base['Cumulative'],
                         name='Baseline MSE', line=dict(color='#8888ff', width=2)))
fig.add_trace(go.Scatter(x=cust['Date'], y=cust['Cumulative'],
                         name='Custom AVP-Loss', line=dict(color='#00C853', width=2)))
fig.update_layout(height=450, hovermode='x unified',
                  legend=dict(orientation='h', yanchor='bottom', y=1.02),
                  margin=dict(l=20, r=20, t=40, b=20),
                  xaxis_title='Trading Date', yaxis_title='Cumulative Return')
st.plotly_chart(fig, use_container_width=True)

# ---------- Drawdown ----------
st.subheader("Drawdown Comparison")
base_dd = (base['Cumulative'] - base['Cumulative'].cummax()) / base['Cumulative'].cummax()
cust_dd = (cust['Cumulative'] - cust['Cumulative'].cummax()) / cust['Cumulative'].cummax()
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=base['Date'], y=base_dd, name='Baseline',
                          fill='tozeroy', line=dict(color='#8888ff')))
fig2.add_trace(go.Scatter(x=cust['Date'], y=cust_dd, name='Custom AVP',
                          fill='tozeroy', line=dict(color='#00C853')))
fig2.update_layout(height=350, hovermode='x unified', yaxis_tickformat='.1%',
                   margin=dict(l=20, r=20, t=20, b=20),
                   xaxis_title='Trading Date', yaxis_title='Drawdown')
st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.caption("Built with Streamlit and Plotly — Data: Yahoo Finance (daily OHLCV)")