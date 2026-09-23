import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

df = pd.read_parquet('data/nifty_features.parquet')
features = ['Log_Return_1d', 'Log_Return_5d', 'RSI', 'Volatility']
train_window, test_window = 1260, 252

def mdd(r):
    c = (1 + r).cumprod()
    p = c.cummax()
    return ((c - p) / p).min()

def sortino(r):
    d = r[r < 0]
    return 0 if d.std() == 0 else r.mean() / d.std() * np.sqrt(252)

def make_obj(lam, sigma_train):
    def avp(preds, dtrain):
        y = dtrain.get_label()
        # preds > y means the model over-predicted the return
        mult = np.where(preds > y, 1.0 + lam * sigma_train, 1.0)
        # Match XGBoost's internal 0.5*(y-yhat)^2 scale exactly
        grad = mult * (preds - y)
        hess = mult
        return grad, hess
    return avp

BASE_PARAMS = {'max_depth': 4, 'learning_rate': 0.05, 'base_score': 0.0}

print("Running Unweighted Baseline...")
preds_b, acts_b = [], []
for i in range(train_window, len(df) - test_window, test_window):
    tr = df.iloc[:i]
    te = df.iloc[i:i+test_window]
    dtr = xgb.DMatrix(tr[features], label=tr['Target'])
    dte = xgb.DMatrix(te[features])
    bst = xgb.train(BASE_PARAMS, dtr, num_boost_round=100)
    preds_b.extend(bst.predict(dte))
    acts_b.extend(te['Target'].values)

rb = pd.DataFrame({'Actual': acts_b, 'Pred': preds_b})
rb['Strategy'] = np.where(rb['Pred'] > 0, 1, 0) * rb['Actual']
print(f"BASELINE UNWEIGHTED | RMSE={np.sqrt(mean_squared_error(rb['Actual'], rb['Pred'])):.6f} | MDD={mdd(rb['Strategy']):.4f} | Sortino={sortino(rb['Strategy']):.4f}")
print("-" * 70)
print(f"{'lambda':>8} | {'RMSE':>10} | {'MDD':>10} | {'Sortino':>10}")
print("-" * 70)

for lam in [0.0, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
    preds, acts = [], []
    for i in range(train_window, len(df) - test_window, test_window):
        tr = df.iloc[:i]
        te = df.iloc[i:i+test_window]
        dtr = xgb.DMatrix(tr[features], label=tr['Target'])
        dte = xgb.DMatrix(te[features])
        sigma_tr = tr['Volatility'].values
        bst = xgb.train(BASE_PARAMS, dtr, num_boost_round=100,
                        obj=make_obj(lam, sigma_tr))
        preds.extend(bst.predict(dte))
        acts.extend(te['Target'].values)
    r = pd.DataFrame({'Actual': acts, 'Pred': preds})
    r['Strategy'] = np.where(r['Pred'] > 0, 1, 0) * r['Actual']
    rmse = np.sqrt(mean_squared_error(r['Actual'], r['Pred']))
    print(f"{lam:>8.1f} | {rmse:>10.6f} | {mdd(r['Strategy']):>10.4f} | {sortino(r['Strategy']):>10.4f}")