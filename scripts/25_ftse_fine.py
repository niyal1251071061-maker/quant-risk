"""Stage 2: Fine lambda search around the best coarse value."""
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

# =========== EDIT THIS ===========
CENTER_LAMBDA = 1.0      # <-- paste best λ from coarse run
DEPTH = 6                # <-- paste best depth from coarse run
WIDTH_FACTOR = 3.0       # search 1/3x to 3x around center
N_POINTS = 40            # fine points
# =================================

LAMBDAS = np.logspace(np.log10(CENTER_LAMBDA/WIDTH_FACTOR), 
                       np.log10(CENTER_LAMBDA*WIDTH_FACTOR), N_POINTS)

print(f"Fine search: {N_POINTS} lambdas from {LAMBDAS[0]:.4f} to {LAMBDAS[-1]:.4f}")
print(f"Depth: {DEPTH}")
print(f"BASELINE: MDD={b_mdd:.4f} | Sortino={b_sort:.4f}\n")

results = []
print(f"{'lambda':>10} | {'MDD':>10} | {'Sortino':>10} | {'MDD Δ':>9} | {'Sort Δ':>9} | {'Score':>8}")
print("-" * 75)

for lam in LAMBDAS:
    preds, acts = [], []
    for i in range(train_window, len(df) - test_window, test_window):
        tr = df.iloc[:i]; te = df.iloc[i:i+test_window]
        tr_inner = tr.iloc[:-inner_val]; val_inner = tr.iloc[-inner_val:]
        dtr = xgb.DMatrix(tr_inner[features], label=tr_inner['Target'])
        dval = xgb.DMatrix(val_inner[features], label=val_inner['Target'])
        dte = xgb.DMatrix(te[features])
        bst = xgb.train(
            {'max_depth': DEPTH, 'learning_rate': 0.05,
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

    results.append({'lambda': lam, 'mdd': m, 'sortino': s,
                    'mdd_imp': mdd_imp, 'sort_imp': sort_imp, 'score': score})
    print(f"{lam:>10.4f} | {m:>10.4f} | {s:>10.4f} | {mdd_imp:>+8.1f}% | {sort_imp:>+8.1f}% | {score:>+8.2f}")

results_df = pd.DataFrame(results).sort_values('score', ascending=False)
results_df.to_csv('reports/optimization_logs/FTSE100_fine.csv', index=False)

best = results_df.iloc[0]
print(f"\n{'='*75}")
print(f"  GLOBAL WINNER")
print(f"{'='*75}")
print(f"  Lambda:  {best['lambda']:.4f}")
print(f"  Depth:   {DEPTH}")
print(f"  MDD:     {best['mdd']:.4f}  ({best['mdd_imp']:+.1f}% vs baseline)")
print(f"  Sortino: {best['sortino']:.4f}  ({best['sort_imp']:+.1f}% vs baseline)")

if best['mdd_imp'] > 0 and best['sort_imp'] > 0:
    print(f"\n  ✅ WINNER beats baseline on BOTH metrics")
elif best['mdd_imp'] > 0:
    print(f"\n  ⚠️  MDD improves, Sortino doesn't")
elif best['sort_imp'] > 0:
    print(f"\n  ⚠️  Sortino improves, MDD doesn't")
else:
    print(f"\n  ❌ NO λ beats baseline")
    print(f"     Best MDD: {results_df['mdd_imp'].max():+.1f}%")
    print(f"     Best Sortino: {results_df['sort_imp'].max():+.1f}%")
    print(f"     Conclusion: Framework does not transfer to FTSE 100.")