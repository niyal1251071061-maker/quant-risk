"""
Corrected feature selection — backward elimination.
Score = MDD * 100 + Sortino * 10 (higher = better).
We KEEP features that improve this score, drop those that hurt.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import os

os.makedirs('reports/feature_selection_v2', exist_ok=True)

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

ALL_FEATURES = ['Log_Return_1d', 'Log_Return_5d', 'RSI', 'Volatility',
                'Vol_Ratio', 'Return_20d', 'HL_Range', 'MA_Gap']
train_window, test_window, inner_val = 1260, 252, 126
LAM, DEPTH = 0.5, 6

def score(mdd_val, sort_val):
    """Higher = better. MDD is negative; Sortino positive."""
    return mdd_val * 100 + sort_val * 10

def evaluate(features, df):
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
        'rmse': rmse_fn(r['Actual'], r['Pred']),
        'mdd': mdd(r['Strategy']),
        'sortino': sortino(r['Strategy']),
    }

def backward_elimination(market):
    print(f"\n{'='*76}")
    print(f"  BACKWARD ELIMINATION (CORRECTED) — {market}")
    print(f"{'='*76}\n")

    df = pd.read_parquet(f'data/{market}_features.parquet')
    current = list(ALL_FEATURES)
    history = []

    # Full model baseline
    r = evaluate(current, df)
    r['score'] = score(r['mdd'], r['sortino'])
    history.append({'step': 0, 'features': '+'.join(current), 'n': len(current), **r})
    print(f"Step 0 — Full 8 features")
    print(f"  RMSE={r['rmse']:.6f}  MDD={r['mdd']:.4f}  Sortino={r['sortino']:.4f}  Score={r['score']:.3f}\n")

    step = 1
    while len(current) > 3:
        print(f"Step {step} — Trying to remove one feature from {len(current)}")
        best_removal = None
        best_score = history[-1]['score']
        best_r = None

        for f in current:
            trial = [x for x in current if x != f]
            res = evaluate(trial, df)
            res['score'] = score(res['mdd'], res['sortino'])
            marker = ""
            if res['score'] > best_score:
                best_score = res['score']
                best_removal = f
                best_r = res
                marker = " ← BEST"
            print(f"  Remove '{f:<16}': MDD={res['mdd']:.4f}  Sort={res['sortino']:.4f}  Score={res['score']:.3f}{marker}")

        if best_removal is not None:
            print(f"\n  ✅ Removing '{best_removal}' improves score. Keeping it removed.")
            current.remove(best_removal)
            history.append({
                'step': step, 'features': '+'.join(current), 'n': len(current),
                'removed': best_removal, **best_r
            })
            print(f"  New set ({len(current)}): {current}\n")
            step += 1
        else:
            print(f"\n  ❌ No removal improves score. Stopping.")
            break

    best = history[-1]
    print(f"{'='*76}")
    print(f"  FINAL for {market}")
    print(f"{'='*76}")
    print(f"  Features: {best['features']}")
    print(f"  Count:    {best['n']}")
    print(f"  RMSE:     {best['rmse']:.6f}")
    print(f"  MDD:      {best['mdd']:.4f}")
    print(f"  Sortino:  {best['sortino']:.4f}")
    removed = [f for f in ALL_FEATURES if f not in best['features'].split('+')]
    print(f"  Removed:  {removed if removed else 'none'}")

    return history, best

nifty_history, nifty_best = backward_elimination('NIFTY50')
sp500_history, sp500_best = backward_elimination('SP500')

# Save
pd.DataFrame(nifty_history).to_csv('reports/feature_selection_v2/NIFTY50_selection.csv', index=False)
pd.DataFrame(sp500_history).to_csv('reports/feature_selection_v2/SP500_selection.csv', index=False)

print(f"\n{'='*76}")
print(f"  CROSS-MARKET SUMMARY")
print(f"{'='*76}")
print(f"\n  NIFTY 50: {nifty_best['features']}")
print(f"    RMSE={nifty_best['rmse']:.6f}  MDD={nifty_best['mdd']:.4f}  Sortino={nifty_best['sortino']:.4f}")
print(f"\n  S&P 500: {sp500_best['features']}")
print(f"    RMSE={sp500_best['rmse']:.6f}  MDD={sp500_best['mdd']:.4f}  Sortino={sp500_best['sortino']:.4f}")
print(f"\n  Saved -> reports/feature_selection_v2/")