"""
Feature ablation study.
Removes one feature at a time from the best config and measures impact.
Best config: NIFTY 50, depth=6, lambda=0.5.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import os

os.makedirs('reports/ablation', exist_ok=True)

# ---------- Metric functions ----------
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

def make_obj(lam, sigma_train):
    def avp(preds, dtrain):
        y = dtrain.get_label()
        mult = np.where(preds > y, 1.0 + lam * sigma_train, 1.0)
        return mult * (preds - y), mult
    return avp

# ---------- Config ----------
ALL_FEATURES = ['Log_Return_1d', 'Log_Return_5d', 'RSI', 'Volatility',
                'Vol_Ratio', 'Return_20d', 'HL_Range', 'MA_Gap']
train_window, test_window, inner_val = 1260, 252, 126
LAM = 0.5
DEPTH = 6

df = pd.read_parquet('data/NIFTY50_features.parquet')

# ---------- Full model baseline ----------
def train_and_evaluate(features, label):
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
            obj=make_obj(LAM, tr_inner['Volatility'].values),
            evals=[(dval, 'val')], early_stopping_rounds=30, verbose_eval=False
        )
        best_it = bst.best_iteration + 1 if bst.best_iteration is not None else 500
        preds.extend(bst.predict(dte, iteration_range=(0, best_it)))
        acts.extend(te['Target'].values)

    r = pd.DataFrame({'Actual': acts, 'Pred': preds})
    r['Strategy'] = (r['Pred'] > 0).astype(int) * r['Actual']
    return {
        'label': label,
        'n_features': len(features),
        'rmse': rmse_fn(r['Actual'], r['Pred']),
        'mdd': mdd(r['Strategy']),
        'sortino': sortino(r['Strategy']),
    }

print(f"{'='*80}")
print(f"  ABLATION STUDY — NIFTY 50, depth={DEPTH}, lambda={LAM}")
print(f"{'='*80}\n")

# Full model
print("Training full model (8 features)...")
full = train_and_evaluate(ALL_FEATURES, 'FULL')
print(f"  RMSE={full['rmse']:.6f}  MDD={full['mdd']:.4f}  Sortino={full['sortino']:.4f}\n")

# Remove one at a time
results = [full]
for feature in ALL_FEATURES:
    reduced = [f for f in ALL_FEATURES if f != feature]
    print(f"Training without '{feature}'...")
    r = train_and_evaluate(reduced, f'NO_{feature}')
    results.append(r)
    print(f"  RMSE={r['rmse']:.6f}  MDD={r['mdd']:.4f}  Sortino={r['sortino']:.4f}")

# ---------- Analysis ----------
print(f"\n{'='*80}")
print(f"  ABLATION RESULTS TABLE")
print(f"{'='*80}")
print(f"  {'Config':<28} {'n_feat':>6} | {'RMSE':>10} {'MDD':>10} {'Sortino':>10}")
print(f"  {'-'*76}")
for r in results:
    print(f"  {r['label']:<28} {r['n_features']:>6} | {r['rmse']:>10.6f} {r['mdd']:>10.4f} {r['sortino']:>10.4f}")

# ---------- Feature importance ----------
print(f"\n{'='*80}")
print(f"  FEATURE IMPORTANCE (based on MDD degradation when removed)")
print(f"{'='*80}")
print(f"  {'Feature Removed':<22} {'Δ MDD':>12} {'Δ Sortino':>12} {'Δ RMSE':>12} {'Importance':>12}")
print(f"  {'-'*76}")

importances = []
for r in results[1:]:
    feature = r['label'].replace('NO_', '')
    delta_mdd = r['mdd'] - full['mdd']  # positive = removing hurt (feature valuable)
    delta_sort = r['sortino'] - full['sortino']
    delta_rmse = r['rmse'] - full['rmse']  # positive = removing hurt

    # Composite importance score (weighted toward MDD)
    importance = 0.6 * (-delta_mdd) + 0.3 * (-delta_sort) + 0.1 * delta_rmse
    importances.append({
        'feature': feature,
        'delta_mdd': delta_mdd,
        'delta_sortino': delta_sort,
        'delta_rmse': delta_rmse,
        'importance': importance,
    })

importances.sort(key=lambda x: x['importance'], reverse=True)

for imp in importances:
    print(f"  {imp['feature']:<22} {imp['delta_mdd']:>+12.4f} {imp['delta_sortino']:>+12.4f} "
          f"{imp['delta_rmse']:>+12.6f} {imp['importance']:>+12.4f}")

# ---------- Summary ----------
print(f"\n{'='*80}")
print(f"  TOP 3 MOST IMPORTANT FEATURES")
print(f"{'='*80}")
for i, imp in enumerate(importances[:3]):
    print(f"  {i+1}. {imp['feature']:<20} (importance score: {imp['importance']:+.4f})")

print(f"\n  BOTTOM 2 LEAST IMPORTANT FEATURES (candidates for removal)")
for imp in importances[-2:]:
    print(f"     - {imp['feature']:<20} (importance score: {imp['importance']:+.4f})")

# ---------- Save ----------
df_results = pd.DataFrame(results)
df_results.to_csv('reports/ablation/ablation_full_results.csv', index=False)

df_importance = pd.DataFrame(importances)
df_importance.to_csv('reports/ablation/ablation_importance_ranking.csv', index=False)

print(f"\n  Saved -> reports/ablation/ablation_full_results.csv")
print(f"  Saved -> reports/ablation/ablation_importance_ranking.csv")