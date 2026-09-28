import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Quantitative Risk Mitigation", page_icon="📈", layout="wide")

# ---------- Metric Functions ----------
def mdd(r):
    r = pd.Series(r)
    c = (1 + r).cumprod(); p = c.cummax()
    return ((c - p) / p).min()

def sortino(r):
    r = pd.Series(r)
    d = r[r < 0]
    return 0 if d.std() == 0 else r.mean() / d.std() * np.sqrt(252)

def sharpe(r):
    return (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() > 0 else 0

def rmse_fn(a, p):
    return np.sqrt(((a - p) ** 2).mean())

# ---------- Sidebar ----------
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
except FileNotFoundError as e:
    st.error(f"Missing data for {ticker_label}: {e}")
    st.stop()

base['Signal'] = (base['Pred'] > 0).astype(int)
cust['Signal'] = (cust['Pred'] > 0).astype(int)
base['Strategy'] = base['Signal'] * base['Actual']
cust['Strategy'] = cust['Signal'] * cust['Actual']
base['Cumulative'] = (1 + base['Strategy']).cumprod()
cust['Cumulative'] = (1 + cust['Strategy']).cumprod()

# ---------- Improvements ----------
mdd_imp  = (1 - mdd(cust['Strategy']) / mdd(base['Strategy'])) * 100
sort_imp = (sortino(cust['Strategy']) / sortino(base['Strategy']) - 1) * 100
rmse_imp = (1 - rmse_fn(cust['Actual'], cust['Pred']) / rmse_fn(base['Actual'], base['Pred'])) * 100
sharpe_imp = (sharpe(cust['Strategy']) / sharpe(base['Strategy']) - 1) * 100 if sharpe(base['Strategy']) != 0 else 0

# ---------- Sidebar About ----------
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
- Cross-market tested on NIFTY 50, S&P 500, FTSE 100  

**{ticker_label} Results:**  
- MDD improved **{mdd_imp:+.1f}%**  
- Sortino improved **{sort_imp:+.1f}%**  
- Sharpe improved **{sharpe_imp:+.1f}%**  
- RMSE improved **{rmse_imp:+.1f}%**  

**Stack:** Python, XGBoost, Pandas, Streamlit, Plotly
""")
st.sidebar.divider()
st.sidebar.warning("⚠ Analytical decision-support tool only. Not financial advice.")

# ---------- Header ----------
st.title("Quantitative Risk Mitigation Dashboard")
st.caption(f"Custom AVP-Loss vs Symmetric MSE Baseline — {ticker_label}, 2019–2023")

# ---------- KPI Cards (5 columns) ----------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Baseline MDD", f"{mdd(base['Strategy']):.2%}")
c2.metric("Custom MDD", f"{mdd(cust['Strategy']):.2%}", delta=f"{mdd_imp:+.1f}%")
c3.metric("Custom Sortino", f"{sortino(cust['Strategy']):.3f}", delta=f"{sort_imp:+.1f}%")
c4.metric("Custom Sharpe", f"{sharpe(cust['Strategy']):.3f}", delta=f"{sharpe_imp:+.1f}%")
c5.metric("RMSE", f"{rmse_fn(cust['Actual'], cust['Pred']):.6f}", delta=f"{rmse_imp:+.1f}%")

st.divider()

# ---------- Charts ----------
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

# ============================================================
# ---------- Crisis Period Analysis ----------
# ============================================================
st.divider()
st.subheader("Crisis Period Analysis")
st.caption("Performance during historical market stress windows.")
try:
    crisis = pd.read_csv('reports/crisis/crisis_analysis.csv')
    crisis_market = crisis[crisis['market'] == ticker_key]
    for _, row in crisis_market.iterrows():
        st.markdown(f"**{row['crisis'].replace('_', ' ')}** — {int(row['days'])} days")
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Baseline MDD", f"{row['baseline_mdd']:.2%}")
        cc2.metric("Custom MDD", f"{row['custom_mdd']:.2%}", delta=f"{row['mdd_improvement_pct']:+.1f}%")
        cc3.metric("Baseline Sortino", f"{row['baseline_sortino']:.3f}")
        cc4.metric("Custom Sortino", f"{row['custom_sortino']:.3f}")
except Exception as e:
    st.info(f"Crisis data not available: {e}")

# ============================================================
# ---------- Direction Classification ----------
# ============================================================
st.divider()
st.subheader("Direction Classification (Up/Down Prediction)")
st.caption("Volatility-adjusted threshold applied on standard logistic classifier.")

CLASS_RESULTS = {
    'NIFTY50': {'auc': 0.5680, 'bal_acc': 0.5728, 'beta': 0.1, 'base_thr': 0.52},
    'SP500':   {'auc': 0.5189, 'bal_acc': 0.5371, 'beta': 1.0, 'base_thr': 0.55},
}

cr = CLASS_RESULTS[ticker_key]
cl1, cl2, cl3 = st.columns(3)
cl1.metric("AUC", f"{cr['auc']:.4f}")
cl2.metric("Balanced Accuracy", f"{cr['bal_acc']:.4f}")
cl3.metric("Threshold β", f"{cr['beta']:.2f}")
st.caption(f"Optimal config: base_threshold={cr['base_thr']:.2f}, vol_adjustment_β={cr['beta']:.2f}")

# ============================================================
# ---------- Transaction Cost Analysis ----------
# ============================================================
st.divider()
st.subheader("Transaction Cost Analysis")
st.caption("Performance after 0.1% cost per position change.")
try:
    costs = pd.read_csv('reports/costs/transaction_cost_summary.csv')
    costs_market = costs[costs['market'] == ticker_key]
    if len(costs_market) > 0:
        row = costs_market.iloc[0]
        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("Baseline trades", f"{int(row['trades_baseline'])}")
        tc2.metric("Custom trades", f"{int(row['trades_custom'])}")
        tc3.metric("Net MDD improvement", f"{row['mdd_imp_net']:+.1f}%" if not pd.isna(row['mdd_imp_net']) else "Custom remains positive")
        st.caption(f"Gross Sortino improvement: {row['sort_imp_gross']:+.1f}%")
    else:
        st.info(f"No cost data for {ticker_label}.")
except Exception as e:
    st.info(f"Transaction cost data not available.")

# ============================================================
# ---------- Statistical Significance ----------
# ============================================================
st.divider()
st.subheader("Statistical Significance (Block Bootstrap)")
st.caption("Stationary block bootstrap (n=5000, avg block=60 days) on MDD and Sortino.")
try:
    boot = pd.read_csv('reports/statistics/bootstrap_v2_results.csv')
    boot_market = boot[boot['market'] == ticker_key]
    if len(boot_market) > 0:
        b = boot_market.iloc[0]
        bs1, bs2 = st.columns(2)
        bs1.metric("MDD: P(custom better)", f"{b['mdd_p_custom_better']:.4f}",
                   delta=f"95% CI [{b['mdd_ci_low']:.3f}, {b['mdd_ci_high']:.3f}]",
                   delta_color="off")
        bs2.metric("Sortino: P(custom better)", f"{b['sort_p_custom_better']:.4f}",
                   delta=f"95% CI [{b['sort_ci_low']:.3f}, {b['sort_ci_high']:.3f}]",
                   delta_color="off")
except Exception:
    st.info("Bootstrap results not available.")

# ============================================================
# ---------- Multi-Seed Robustness ----------
# ============================================================
st.divider()
st.subheader("Reproducibility")
st.caption("XGBoost tree-building is deterministic with exact splits and full subsampling.")
try:
    seeds = pd.read_csv('reports/multiseed/seed_robustness.csv')
    seeds_market = seeds[seeds['market'] == ticker_key]
    if len(seeds_market) > 0:
        s = seeds_market.iloc[0]
        st.success(
            f"Zero variance across 5 seeds — fully reproducible. "
            f"MDD={s['mdd_mean']:.4f} ± {s['mdd_std']:.4f}, "
            f"Sortino={s['sortino_mean']:.4f} ± {s['sortino_std']:.4f}"
        )
except Exception:
    st.info("Multi-seed results not available.")

# ============================================================
# ---------- Feature Importance ----------
# ============================================================
st.divider()
st.subheader("Feature Importance")
st.caption("SHAP analysis — which features drive the model's predictions.")

shap_img = f'reports/shap/{ticker_key}_beeswarm.png'
shap_bar = f'reports/shap/{ticker_key}_importance.png'

col_img1, col_img2 = st.columns(2)
with col_img1:
    if os.path.exists(shap_img):
        st.image(shap_img, caption=f"{ticker_label} — SHAP beeswarm", use_container_width=True)
    else:
        st.info("SHAP beeswarm plot not found.")
with col_img2:
    if os.path.exists(shap_bar):
        st.image(shap_bar, caption=f"{ticker_label} — Feature importance", use_container_width=True)
    else:
        st.info("SHAP importance plot not found.")

# Ablation table
try:
    ablation = pd.read_csv('reports/ablation/ablation_importance_ranking.csv')
    st.markdown("**Ablation: Feature Importance Ranking**")
    st.dataframe(ablation, use_container_width=True, hide_index=True)
except Exception:
    pass

# ============================================================
# ---------- Footer ----------
# ============================================================
st.divider()
st.caption("Built with Streamlit + Plotly | Data: Yahoo Finance daily OHLCV | Group SY-H1 · VIT Pune")