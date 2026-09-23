import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

df = pd.read_parquet('data/nifty_features.parquet')

# Add more features
df['Vol_Ratio'] = df['Volatility'] / df['Volatility'].rolling(60).mean()
df['Return_20d'] = np.log(df['Close'] / df['Close'].shift(20))
df['HL_Range'] = (df['High'] - df['Low']) / df['Close']
df['MA_Gap'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close']
df.dropna(inplace=True)

features = ['Log_Return_1d', 'Log_Return_5d', 'RSI', 'Volatility',
            'Vol_Ratio', 'Return_20d', 'HL_Range', 'MA_Gap']
train_window, test_window = 1260, 252
inner_val = 126

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

print(f"{'depth':>6} | {'lambda':>8} | {'RMSE':>10} | {'MDD':>10} | {'Sortino':>10}")
print("-" * 65)

for depth in [4, 5, 6]:
    for lam in [0.75, 1.0, 1.25]:
        preds, acts = [], []
        for i in range(train_window, len(df) - test_window, test_window):
            tr = df.iloc[:i]; te = df.iloc[i:i+test_window]
            tr_inner = tr.iloc[:-inner_val]
            val_inner = tr.iloc[-inner_val:]
            
            dtr = xgb.DMatrix(tr_inner[features], label=tr_inner['Target'])
            dval = xgb.DMatrix(val_inner[features], label=val_inner['Target'])
            dte = xgb.DMatrix(te[features])
            
            params = {'max_depth': depth, 'learning_rate': 0.05,
                      'base_score': 0.0, 'reg_lambda': 1.0}
            bst = xgb.train(
                params, dtr, num_boost_round=500,
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
        print(f"{depth:>6} | {lam:>8.2f} | {rmse:>10.6f} | {mdd(r['Strategy']):>10.4f} | {sortino(r['Strategy']):>10.4f}")