import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

df = pd.read_parquet('data/nifty_features.parquet')
features = ['Log_Return_1d', 'Log_Return_5d', 'RSI', 'Volatility']
train_window, test_window = 1260, 252

def mdd(returns):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    return ((cum - peak) / peak).min()

def sortino(returns):
    d = returns[returns < 0]
    if d.std() == 0: return 0
    return returns.mean() / d.std() * np.sqrt(252)

def make_obj(lam):
    def avp(preds, dtrain):
        y = dtrain.get_label()
        sigma = dtrain.get_weight()
        err = y - preds
        over = preds > y
        mult = np.where(over, 1.0 + lam * sigma, 1.0)
        return -2.0 * mult * err, 2.0 * mult
    return avp

print(f"{'lambda':>8} | {'RMSE':>10} | {'MDD':>10} | {'Sortino':>10}")
print("-" * 50)

for lam in [0.0, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]:
    preds, acts = [], []
    for i in range(train_window, len(df) - test_window, test_window):
        tr = df.iloc[:i]; te = df.iloc[i:i+test_window]
        dtr = xgb.DMatrix(tr[features], label=tr['Target'], weight=tr['Volatility'])
        dte = xgb.DMatrix(te[features])
        bst = xgb.train({'max_depth': 4, 'learning_rate': 0.05}, dtr,
                        num_boost_round=100, obj=make_obj(lam))
        preds.extend(bst.predict(dte))
        acts.extend(te['Target'].values)
    r = pd.DataFrame({'Actual': acts, 'Pred': preds})
    r['Strategy'] = np.where(r['Pred'] > 0, 1, 0) * r['Actual']
    rmse = np.sqrt(mean_squared_error(r['Actual'], r['Pred']))
    print(f"{lam:>8.1f} | {rmse:>10.6f} | {mdd(r['Strategy']):>10.4f} | {sortino(r['Strategy']):>10.4f}")

print("\nNote: lambda=0.0 is equivalent to symmetric MSE (baseline).")