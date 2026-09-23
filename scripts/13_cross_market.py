import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import yfinance as yf
import os

def prepare_dataset(ticker, name):
    print(f"\nLoading {name} ({ticker})...")
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
    return df

def mdd(r):
    c = (1 + r).cumprod(); p = c.cummax()
    return ((c - p) / p).min()

def sortino(r):
    d = r[r < 0]
    return 0 if d.std() == 0 else r.mean() / d.std() * np.sqrt(252)

def make_obj(lam, sigma_train):
    def avp(preds, dtrain):
        y = dtrain.get_label()
        mult = np.where(preds > y, 1.0 + lam * sigma_train, 1.0)
        return mult * (preds - y), mult
    return avp

features = ['Log_Return_1d', 'Log_Return_5d', 'RSI', 'Volatility',
            'Vol_Ratio', 'Return_20d', 'HL_Range', 'MA_Gap']
train_window, test_window, inner_val = 1260, 252, 126
LAM, DEPTH = 0.75, 6

def run(ticker, name):
    df = prepare_dataset(ticker, name)
    
    # Baseline
    preds_b, acts_b, dates_b = [], [], []
    for i in range(train_window, len(df) - test_window, test_window):
        tr = df.iloc[:i]; te = df.iloc[i:i+test_window]
        dtr = xgb.DMatrix(tr[features], label=tr['Target'])
        dte = xgb.DMatrix(te[features])
        bst = xgb.train({'max_depth': 4, 'learning_rate': 0.05, 'base_score': 0.0},
                        dtr, num_boost_round=100)
        preds_b.extend(bst.predict(dte)); acts_b.extend(te['Target'].values); dates_b.extend(te.index)
    rb = pd.DataFrame({'Date': dates_b, 'Actual': acts_b, 'Pred': preds_b})
    rb['Strategy'] = np.where(rb['Pred'] > 0, 1, 0) * rb['Actual']
    rb.to_csv(f'reports/baseline_{name}.csv', index=False)
    
    # Custom
    preds, acts, dates = [], [], []
    for i in range(train_window, len(df) - test_window, test_window):
        tr = df.iloc[:i]; te = df.iloc[i:i+test_window]
        tr_inner = tr.iloc[:-inner_val]; val_inner = tr.iloc[-inner_val:]
        dtr = xgb.DMatrix(tr_inner[features], label=tr_inner['Target'])
        dval = xgb.DMatrix(val_inner[features], label=val_inner['Target'])
        dte = xgb.DMatrix(te[features])
        bst = xgb.train({'max_depth': DEPTH, 'learning_rate': 0.05,
                         'base_score': 0.0, 'reg_lambda': 1.0},
                        dtr, num_boost_round=500,
                        obj=make_obj(LAM, tr_inner['Volatility'].values),
                        evals=[(dval, 'val')], early_stopping_rounds=30,
                        verbose_eval=False)
        preds.extend(bst.predict(dte, iteration_range=(0, bst.best_iteration+1)))
        acts.extend(te['Target'].values); dates.extend(te.index)
    rc = pd.DataFrame({'Date': dates, 'Actual': acts, 'Pred': preds})
    rc['Strategy'] = np.where(rc['Pred'] > 0, 1, 0) * rc['Actual']
    rc.to_csv(f'reports/custom_{name}.csv', index=False)
    
    print(f"\n{name}:")
    print(f"  Baseline: RMSE={np.sqrt(mean_squared_error(rb['Actual'], rb['Pred'])):.6f} | MDD={mdd(rb['Strategy']):.4f} | Sortino={sortino(rb['Strategy']):.4f}")
    print(f"  Custom:   RMSE={np.sqrt(mean_squared_error(rc['Actual'], rc['Pred'])):.6f} | MDD={mdd(rc['Strategy']):.4f} | Sortino={sortino(rc['Strategy']):.4f}")

run('^NSEI', 'NIFTY50')
run('^GSPC', 'SP500')