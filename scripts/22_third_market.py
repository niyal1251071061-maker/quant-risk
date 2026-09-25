"""
Third market validation — FTSE 100 (UK).
Trains baseline + custom AVP-Loss, computes metrics, compares to existing markets.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import yfinance as yf
import os

os.makedirs('reports', exist_ok=True)

# ---------- Metrics ----------
def mdd(r):
    r = pd.Series(r)
    c = (1 + r).cumprod(); p = c.cummax()
    return ((c - p) / p).min()

def sortino(r):
    r = pd.Series(r)
    d = r[r < 0]
    return 0 if d.std() == 0 else r.mean() / d.std() * np.sqrt(252)

def rmse_fn(a, p):
    return np.sqrt(mean_squared_error(a, p))

# ---------- Custom objective ----------
def make_obj(lam, sigma_train):
    def avp(preds, dtrain):
        y = dtrain.get_label()
        mult = np.where(preds > y, 1.0 + lam * sigma_train, 1.0)
        return mult * (preds - y), mult
    return avp

# ---------- Data preparation ----------
def prepare_dataset(ticker, name):
    print(f"\nDownloading {name} ({ticker})...")
    df = yf.download(ticker, start='2014-01-01', end='2024-01-01', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']

    df['Target'] = np.log(df['Close'] / df['Close'].shift(1)).shift(-1)
    df['Volatility'] = df['Target'].rolling(14).std() * np.sqrt(252)
    df['Log_Return_1d'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Log_Return_5d'] = np.log(df['Close'] / df['Close'].shift(5))

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    df['Vol_Ratio'] = df['Volatility'] / df['Volatility'].rolling(60).mean()
    df['Return_20d'] = np.log(df['Close'] / df['Close'].shift(20))
    df['HL_Range'] = (df['High'] - df['Low']) / df['Close']
    df['MA_Gap'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close']

    df.dropna(inplace=True)
    df.to_parquet(f'data/{name}_features.parquet')
    print(f"  Saved {len(df)} rows to data/{name}_features.parquet")
    return df

# ---------- Configuration ----------
features = ['Log_Return_1d', 'Log_Return_5d', 'RSI', 'Volatility',
            'Vol_Ratio', 'Return_20d', 'HL_Range', 'MA_Gap']
train_window, test_window, inner_val = 1260, 252, 126

# ---------- Training functions ----------
def run_baseline(df, name):
    print(f"\n  Baseline XGBoost (MSE)...")
    preds, acts, dates = [], [], []
    for i in range(train_window, len(df) - test_window, test_window):
        tr = df.iloc[:i]; te = df.iloc[i:i+test_window]
        dtr = xgb.DMatrix(tr[features], label=tr['Target'])
        dte = xgb.DMatrix(te[features])
        bst = xgb.train(
            {'max_depth': 4, 'learning_rate': 0.05, 'base_score': 0.0},
            dtr, num_boost_round=100
        )
        preds.extend(bst.predict(dte))
        acts.extend(te['Target'].values)
        dates.extend(te.index)

    out = pd.DataFrame({'Date': dates, 'Actual': acts, 'Pred': preds})
    out.to_csv(f'reports/baseline_{name}.csv', index=False)
    return out

def run_custom(df, name, depth, lam, lr, reg):
    print(f"\n  Custom AVP-Loss (depth={depth}, λ={lam})...")
    preds, acts, dates = [], [], []
    for i in range(train_window, len(df) - test_window, test_window):
        tr = df.iloc[:i]; te = df.iloc[i:i+test_window]
        tr_inner = tr.iloc[:-inner_val]
        val_inner = tr.iloc[-inner_val:]
        dtr = xgb.DMatrix(tr_inner[features], label=tr_inner['Target'])
        dval = xgb.DMatrix(val_inner[features], label=val_inner['Target'])
        dte = xgb.DMatrix(te[features])
        bst = xgb.train(
            {'max_depth': depth, 'learning_rate': lr,
             'base_score': 0.0, 'reg_lambda': reg},
            dtr, num_boost_round=500,
            obj=make_obj(lam, tr_inner['Volatility'].values),
            evals=[(dval, 'val')], early_stopping_rounds=30, verbose_eval=False
        )
        best_it = bst.best_iteration + 1 if bst.best_iteration is not None else 500
        preds.extend(bst.predict(dte, iteration_range=(0, best_it)))
        acts.extend(te['Target'].values)
        dates.extend(te.index)

    out = pd.DataFrame({'Date': dates, 'Actual': acts, 'Pred': preds})
    out.to_csv(f'reports/custom_{name}.csv', index=False)
    return out

# ---------- Main ----------
if __name__ == '__main__':
    NAME = 'FTSE100'
    TICKER = '^FTSE'

    # Step 1: Prepare data
    df = prepare_dataset(TICKER, NAME)

    # Step 2: Baseline
    base = run_baseline(df, NAME)
    base['Strategy'] = (base['Pred'] > 0).astype(int) * base['Actual']

    # Step 3: Custom (use same config as best found for NIFTY)
    cust = run_custom(df, NAME, depth=6, lam=0.5, lr=0.05, reg=1.0)
    cust['Strategy'] = (cust['Pred'] > 0).astype(int) * cust['Actual']

    # Step 4: Metrics
    b_rmse = rmse_fn(base['Actual'], base['Pred'])
    c_rmse = rmse_fn(cust['Actual'], cust['Pred'])
    b_mdd = mdd(base['Strategy'])
    c_mdd = mdd(cust['Strategy'])
    b_sort = sortino(base['Strategy'])
    c_sort = sortino(cust['Strategy'])

    print(f"\n{'='*72}")
    print(f"  THIRD MARKET RESULT — {NAME} ({TICKER})")
    print(f"{'='*72}")
    print(f"  {'Metric':<12} {'Baseline':>12} {'Custom':>12} {'Change':>12}")
    print(f"  {'-'*54}")
    print(f"  {'RMSE':<12} {b_rmse:>12.6f} {c_rmse:>12.6f} {(1 - c_rmse/b_rmse)*100:>+11.1f}%")
    print(f"  {'MDD':<12} {b_mdd:>12.4f} {c_mdd:>12.4f} {(1 - c_mdd/b_mdd)*100:>+11.1f}%")
    print(f"  {'Sortino':<12} {b_sort:>12.4f} {c_sort:>12.4f} {(c_sort/b_sort - 1)*100:>+11.1f}%")
    print(f"{'='*72}")

    # Save summary
    summary = pd.DataFrame([{
        'market': NAME, 'ticker': TICKER,
        'baseline_rmse': b_rmse, 'custom_rmse': c_rmse,
        'baseline_mdd': b_mdd, 'custom_mdd': c_mdd,
        'baseline_sortino': b_sort, 'custom_sortino': c_sort,
        'mdd_improvement_pct': (1 - c_mdd/b_mdd) * 100,
        'sortino_improvement_pct': (c_sort/b_sort - 1) * 100,
    }])
    summary.to_csv(f'reports/{NAME}_summary.csv', index=False)
    print(f"\nSaved -> reports/{NAME}_summary.csv")