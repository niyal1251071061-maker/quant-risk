import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

df = pd.read_parquet('data/nifty_features.parquet')
features = ['Log_Return_1d', 'Log_Return_5d', 'RSI', 'Volatility']
train_window, test_window = 1260, 252
inner_val = 126  # last 6 months of training for early stopping

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

BASE = {'max_depth': 4, 'learning_rate': 0.05, 'base_score': 0.0}

print(f"{'lambda':>8} | {'RMSE':>10} | {'MDD':>10} | {'Sortino':>10}")
print("-" * 50)

for lam in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
    preds, acts = [], []
    for i in range(train_window, len(df) - test_window, test_window):
        tr = df.iloc[:i]; te = df.iloc[i:i+test_window]
        tr_inner = tr.iloc[:-inner_val]
        val_inner = tr.iloc[-inner_val:]
        
        dtr = xgb.DMatrix(tr_inner[features], label=tr_inner['Target'])
        dval = xgb.DMatrix(val_inner[features], label=val_inner['Target'])
        dte = xgb.DMatrix(te[features])
        
        bst = xgb.train(
            BASE, dtr, num_boost_round=500,
            obj=make_obj(lam, tr_inner['Volatility'].values),
            evals=[(dval, 'val')],
            early_stopping_rounds=30,
            verbose_eval=False
        )
        preds.extend(bst.predict(dte, iteration_range=(0, bst.best_iteration+1)))
        acts.extend(te['Target'].values)
    
    r = pd.DataFrame({'Actual': acts, 'Pred': preds})
    r['Strategy'] = np.where(r['Pred'] > 0, 1, 0) * r['Actual']
    rmse = np.sqrt(mean_squared_error(r['Actual'], r['Pred']))
    print(f"{lam:>8.2f} | {rmse:>10.6f} | {mdd(r['Strategy']):>10.4f} | {sortino(r['Strategy']):>10.4f}")