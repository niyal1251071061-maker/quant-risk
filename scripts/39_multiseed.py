"""
Multi-seed robustness test.
Runs winning configs with 5 different seeds to compute error bars.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import os

os.makedirs('reports/multiseed', exist_ok=True)

features = ['Log_Return_1d', 'Log_Return_5d', 'RSI', 'Volatility',
            'Vol_Ratio', 'Return_20d', 'HL_Range', 'MA_Gap']
train_window, test_window, inner_val = 1260, 252, 126

CONFIGS = {
    'NIFTY50': {'depth': 6, 'lam': 0.5, 'lr': 0.05, 'reg': 1.0},
    'SP500':   {'depth': 8, 'lam': 0.5, 'lr': 0.10, 'reg': 1.0},
}
SEEDS = [0, 42, 123, 456, 789]

def mdd(r):
    r = pd.Series(r); c = (1 + r).cumprod(); p = c.cummax()
    return ((c - p) / p).min()

def sortino(r):
    r = pd.Series(r); d = r[r < 0]
    return 0 if d.std() == 0 else r.mean() / d.std() * np.sqrt(252)

def make_obj(lam, sigma_train):
    def avp(preds, dtrain):
        y = dtrain.get_label()
        mult = np.where(preds > y, 1.0 + lam * sigma_train, 1.0)
        return mult * (preds - y), mult
    return avp

def run_seed(market, cfg, seed):
    df = pd.read_parquet(f'data/{market}_features.parquet')
    preds, acts = [], []
    for i in range(train_window, len(df) - test_window, test_window):
        tr = df.iloc[:i]; te = df.iloc[i:i+test_window]
        tr_inner = tr.iloc[:-inner_val]; val_inner = tr.iloc[-inner_val:]
        dtr = xgb.DMatrix(tr_inner[features], label=tr_inner['Target'])
        dval = xgb.DMatrix(val_inner[features], label=val_inner['Target'])
        dte = xgb.DMatrix(te[features])
        bst = xgb.train(
            {'max_depth': cfg['depth'], 'learning_rate': cfg['lr'],
             'base_score': 0.0, 'reg_lambda': cfg['reg'], 'seed': seed},
            dtr, num_boost_round=500,
            obj=make_obj(cfg['lam'], tr_inner['Volatility'].values),
            evals=[(dval, 'val')], early_stopping_rounds=30, verbose_eval=False
        )
        best_it = bst.best_iteration + 1 if bst.best_iteration is not None else 500
        preds.extend(bst.predict(dte, iteration_range=(0, best_it)))
        acts.extend(te['Target'].values)
    r = pd.DataFrame({'Actual': acts, 'Pred': preds})
    r['Strategy'] = (r['Pred'] > 0).astype(int) * r['Actual']
    return {
        'rmse': np.sqrt(mean_squared_error(r['Actual'], r['Pred'])),
        'mdd': mdd(r['Strategy']),
        'sortino': sortino(r['Strategy']),
    }

print(f"{'='*80}")
print(f"  MULTI-SEED ROBUSTNESS")
print(f"{'='*80}")

all_results = []
for market, cfg in CONFIGS.items():
    print(f"\n  {market} — Config: depth={cfg['depth']}, λ={cfg['lam']}, lr={cfg['lr']}")
    print(f"  {'Seed':>6} | {'RMSE':>12} | {'MDD':>10} | {'Sortino':>10}")
    print(f"  {'-'*50}")
    seed_results = []
    for seed in SEEDS:
        r = run_seed(market, cfg, seed)
        seed_results.append(r)
        print(f"  {seed:>6} | {r['rmse']:>12.6f} | {r['mdd']:>10.4f} | {r['sortino']:>10.4f}")
    
    # Compute mean ± std
    rmse_arr = np.array([r['rmse'] for r in seed_results])
    mdd_arr = np.array([r['mdd'] for r in seed_results])
    sort_arr = np.array([r['sortino'] for r in seed_results])
    
    print(f"  {'-'*50}")
    print(f"  {'Mean':>6} | {rmse_arr.mean():>12.6f} | {mdd_arr.mean():>10.4f} | {sort_arr.mean():>10.4f}")
    print(f"  {'Std':>6} | {rmse_arr.std():>12.6f} | {mdd_arr.std():>10.4f} | {sort_arr.std():>10.4f}")
    
    all_results.append({
        'market': market,
        'rmse_mean': rmse_arr.mean(), 'rmse_std': rmse_arr.std(),
        'mdd_mean': mdd_arr.mean(), 'mdd_std': mdd_arr.std(),
        'sortino_mean': sort_arr.mean(), 'sortino_std': sort_arr.std(),
    })

pd.DataFrame(all_results).to_csv('reports/multiseed/seed_robustness.csv', index=False)
print(f"\n  Saved -> reports/multiseed/seed_robustness.csv")