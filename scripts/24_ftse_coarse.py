"""Stage 1: Coarse lambda search on FTSE 100. Log-spaced range."""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import os

os.makedirs('reports/optimization_logs', exist_ok=True)

def mdd(r):
    r = pd.Series(r)
    c = (1 + r).cumprod(); p = c.cummax()
    return ((c - p) / p).min()

def sortino(r):
    r = pd.Series(r)
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

df = pd.read_parquet('data/FTSE100_features.parquet')

# Baseline
print("Running baseline...")
preds_b, acts_b = [], []
for i in range(train_window, len(df) - test_window, test_window):
    tr = df.iloc[:i]; te = df.iloc[i:i+test_window]
    dtr = xgb.DMatrix(tr[features], label=tr['Target'])
    dte = xgb.DMatrix(te[features])
    bst = xgb.train({'max_depth': 4, 'learning_rate': 0.05, 'base_score': 0.0},
                    dtr, num_boost_round=100)
    preds_b.extend(bst.predict(dte)); acts_b.extend(te['Target'].values)

rb = pd.DataFrame({'Actual': acts_b, 'Pred': preds_b})
rb['Strategy'] = (rb['Pred'] > 0).astype(int) * rb['Actual']
b_mdd = mdd(rb['Strategy'])
b_sort = sortino(rb['Strategy'])
print(f"BASELINE: MDD={b_mdd:.4f} | Sortino={b_sort:.4f}")

# Log-spaced lambdas
LAMBDAS = np.logspace(np.log10(0.01), np.log10(1000), 40)  # 40 values
DEPTHS = [4, 6]

results = []
print(f"\nRunning {len(LAMBDAS)*len(DEPTHS)} configs...\n")
print(f"{'depth':>6} | {'lambda':>10} | {'MDD':>10} | {'Sortino':>10} | {'MDD Δ':>9} | {'Sort Δ':>9}")
print("-" * 80)

for depth in DEPTHS:
    for lam in LAMBDAS:
        preds, acts = [], []
        for i in range(train_window, len(df) - test_window, test_window):
            tr = df.iloc[:i]; te = df.iloc[i:i+test_window]
            tr_inner = tr.iloc[:-inner_val]; val_inner = tr.iloc[-inner_val:]
            dtr = xgb.DMatrix(tr_inner[features], label=tr_inner['Target'])
            dval = xgb.DMatrix(val_inner[features], label=val_inner['Target'])
            dte = xgb.DMatrix(te[features])
            bst = xgb.train(
                {'max_depth': depth, 'learning_rate': 0.05,
                 'base_score': 0.0, 'reg_lambda': 1.0},
                dtr, num_boost_round=500,
                obj=make_obj(lam, tr_inner['Volatility'].values),
                evals=[(dval, 'val')], early_stopping_rounds=30, verbose_eval=False
            )
            best_it = bst.best_iteration + 1 if bst.best_iteration is not None else 500
            preds.extend(bst.predict(dte, iteration_range=(0, best_it)))
            acts.extend(te['Target'].values)

        r = pd.DataFrame({'Actual': acts, 'Pred': preds})
        r['Strategy'] = (r['Pred'] > 0).astype(int) * r['Actual']
        m = mdd(r['Strategy'])
        s = sortino(r['Strategy'])
        mdd_imp = (1 - m/b_mdd) * 100
        sort_imp = (s/b_sort - 1) * 100
        score = 0.7 * mdd_imp + 0.3 * sort_imp

        results.append({
            'depth': depth, 'lambda': lam, 'mdd': m, 'sortino': s,
            'mdd_imp': mdd_imp, 'sort_imp': sort_imp, 'score': score
        })
        print(f"{depth:>6} | {lam:>10.4f} | {m:>10.4f} | {s:>10.4f} | {mdd_imp:>+8.1f}% | {sort_imp:>+8.1f}%")

results_df = pd.DataFrame(results).sort_values('score', ascending=False)
results_df.to_csv('reports/optimization_logs/FTSE100_coarse.csv', index=False)

print(f"\n{'='*80}")
print(f"  TOP 10 COARSE CONFIGS")
print(f"{'='*80}")
for i, row in results_df.head(10).iterrows():
    print(f"  depth={int(row['depth'])}  λ={row['lambda']:.4f}  MDD={row['mdd']:.4f} ({row['mdd_imp']:+.1f}%)  Sortino={row['sortino']:.4f} ({row['sort_imp']:+.1f}%)")

best = results_df.iloc[0]
print(f"\n  Best λ: {best['lambda']:.4f} (depth={int(best['depth'])})")
print(f"  Next stage should search around this value.")
print(f"\nSaved -> reports/optimization_logs/FTSE100_coarse.csv")