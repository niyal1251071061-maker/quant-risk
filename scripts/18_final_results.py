"""
Final Review 1 lock-in script.
Trains NIFTY 50 and S&P 500 with the optimal hyperparameters found
by the exhaustive grid search (scripts/17_best_search.py).

Winning configs:
  NIFTY 50 : depth=6, lambda=0.50, lr=0.05, reg=1.0
  S&P 500  : depth=8, lambda=0.50, lr=0.10, reg=1.0

Outputs:
  reports/custom_NIFTY50.csv
  reports/custom_SP500.csv
  reports/FINAL_REVIEW1_SUMMARY.csv
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import os

os.makedirs('reports', exist_ok=True)

# ---------- Metrics ----------
def mdd(r):
    c = (1 + r).cumprod(); p = c.cummax()
    return ((c - p) / p).min()

def sortino(r):
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

# ---------- Config ----------
features = ['Log_Return_1d', 'Log_Return_5d', 'RSI', 'Volatility',
            'Vol_Ratio', 'Return_20d', 'HL_Range', 'MA_Gap']
train_window, test_window, inner_val = 1260, 252, 126

CONFIGS = {
    'NIFTY50': {'depth': 6, 'lambda': 0.50, 'lr': 0.05, 'reg': 1.0,
                'baseline_rmse': 0.013218, 'baseline_mdd': -0.3183, 'baseline_sortino': 1.0415},
    'SP500':   {'depth': 8, 'lambda': 0.50, 'lr': 0.10, 'reg': 1.0,
                'baseline_rmse': 0.015253, 'baseline_mdd': -0.3247, 'baseline_sortino': 0.2964},
}

# ---------- Training ----------
def train_market(name, cfg):
    print(f"\n{'='*72}")
    print(f"  Training {name}  |  depth={cfg['depth']}, λ={cfg['lambda']}, lr={cfg['lr']}, reg={cfg['reg']}")
    print(f"{'='*72}")
    df = pd.read_parquet(f'data/{name}_features.parquet')

    preds, acts, dates = [], [], []
    for i in range(train_window, len(df) - test_window, test_window):
        tr = df.iloc[:i]; te = df.iloc[i:i+test_window]
        tr_inner = tr.iloc[:-inner_val]
        val_inner = tr.iloc[-inner_val:]

        dtr = xgb.DMatrix(tr_inner[features], label=tr_inner['Target'])
        dval = xgb.DMatrix(val_inner[features], label=val_inner['Target'])
        dte = xgb.DMatrix(te[features])

        bst = xgb.train(
            {'max_depth': cfg['depth'], 'learning_rate': cfg['lr'],
             'base_score': 0.0, 'reg_lambda': cfg['reg']},
            dtr, num_boost_round=500,
            obj=make_obj(cfg['lambda'], tr_inner['Volatility'].values),
            evals=[(dval, 'val')],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        best_it = bst.best_iteration + 1 if bst.best_iteration is not None else 500
        preds.extend(bst.predict(dte, iteration_range=(0, best_it)))
        acts.extend(te['Target'].values)
        dates.extend(te.index)

    r = pd.DataFrame({'Date': dates, 'Actual': acts, 'Pred': preds})
    r.to_csv(f'reports/custom_{name}.csv', index=False)

    r['Strategy'] = np.where(r['Pred'] > 0, 1, 0) * r['Actual']
    m = mdd(r['Strategy'])
    s = sortino(r['Strategy'])
    rm = rmse_fn(r['Actual'], r['Pred'])

    bl_mdd = cfg['baseline_mdd']
    bl_sort = cfg['baseline_sortino']

    print(f"  RMSE    : {rm:.6f}  (baseline {cfg['baseline_rmse']:.6f})")
    print(f"  MDD     : {m:.4f}    (baseline {bl_mdd:.4f})   -> {(1 - m/bl_mdd)*100:+.1f}%")
    print(f"  Sortino : {s:.4f}    (baseline {bl_sort:.4f})   -> {(s/bl_sort - 1)*100:+.1f}%")
    print(f"  Saved -> reports/custom_{name}.csv")

    return {
        'market': name, 'depth': cfg['depth'], 'lambda': cfg['lambda'],
        'lr': cfg['lr'], 'reg': cfg['reg'],
        'rmse': rm, 'mdd': m, 'sortino': s,
        'baseline_rmse': cfg['baseline_rmse'],
        'baseline_mdd': bl_mdd, 'baseline_sortino': bl_sort,
        'mdd_improvement_pct': (1 - m / bl_mdd) * 100,
        'sortino_improvement_pct': (s / bl_sort - 1) * 100,
    }

# ---------- Run ----------
if __name__ == '__main__':
    results = [train_market(name, cfg) for name, cfg in CONFIGS.items()]

    summary = pd.DataFrame(results)
    summary.to_csv('reports/FINAL_REVIEW1_SUMMARY.csv', index=False)

    print(f"\n{'='*72}")
    print("  FINAL REVIEW 1 SUMMARY")
    print(f"{'='*72}")
    print(f"{'Market':>10} | {'Base MDD':>10} | {'Cust MDD':>10} | {'Δ MDD':>8} | {'Base Sort':>10} | {'Cust Sort':>10} | {'Δ Sort':>8}")
    print("-" * 90)
    for row in results:
        print(f"{row['market']:>10} | {row['baseline_mdd']:>10.4f} | {row['mdd']:>10.4f} | "
              f"{row['mdd_improvement_pct']:>+7.1f}% | {row['baseline_sortino']:>10.4f} | "
              f"{row['sortino']:>10.4f} | {row['sortino_improvement_pct']:>+7.1f}%")
    print(f"{'='*72}")
    print("\nSaved -> reports/FINAL_REVIEW1_SUMMARY.csv")