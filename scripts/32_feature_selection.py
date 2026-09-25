"""
Feature selection via backward elimination.
Starts with all 8 features, removes one at a time, keeps the best.
Runs on both NIFTY 50 and S&P 500. Finds optimal subset per market.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import os

os.makedirs('reports/feature_selection', exist_ok=True)

def mdd(r):
    r = pd.Series(r); c = (1 + r).cumprod(); p = c.cummax()
    return ((c - p) / p).min()

def sortino(r):
    r = pd.Series(r); d = r[r < 0]
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

def evaluate(features, df):
    """Train and return MDD, Sortino, RMSE."""
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
    print(f"\n{'='*90}")
    print(f"  BACKWARD ELIMINATION — {market}")
    print(f"{'='*90}\n")

    df = pd.read_parquet(f'data/{market}_features.parquet')
    current = list(ALL_FEATURES)
    history = []

    # Step 0: full 8-feature model
    print(f"Step 0 — Full 8-feature baseline")
    r = evaluate(current, df)
    history.append({'step': 0, 'features': '+'.join(current), 'n': len(current), **r})
    print(f"  n=8  RMSE={r['rmse']:.6f}  MDD={r['mdd']:.4f}  Sortino={r['sortino']:.4f}\n")

    step = 1
    while len(current) > 2:
        print(f"Step {step} — trying to remove one feature from {len(current)}")
        best_removal = None
        best_score = -np.inf

        for feature in current:
            trial = [f for f in current if f != feature]
            r = evaluate(trial, df)
            # Score: higher MDD improvement (less negative) + higher Sortino
            score = (-r['mdd']) * 1000 + r['sortino'] * 10 + (1 - r['rmse']) * 100
            print(f"    Remove '{feature}': RMSE={r['rmse']:.6f}  MDD={r['mdd']:.4f}  Sortino={r['sortino']:.4f}  score={score:.3f}")

            if score > best_score:
                best_score = score
                best_removal = feature
                best_r = r

        # Compare best removal to current
        current_r = history[-1]
        current_score = (-current_r['mdd']) * 1000 + current_r['sortino'] * 10 + (1 - current_r['rmse']) * 100

        if best_score > current_score:
            print(f"\n  ✅ Removing '{best_removal}' improves score. Keeping it removed.")
            current.remove(best_removal)
            history.append({
                'step': step, 'features': '+'.join(current), 'n': len(current),
                'removed': best_removal, **best_r
            })
            print(f"  New set ({len(current)} features): {current}\n")
        else:
            print(f"\n  ❌ No removal improves score. Stopping.")
            break
        step += 1

    # Final result
    best = history[-1]
    print(f"{'='*90}")
    print(f"  FINAL RESULT FOR {market}")
    print(f"{'='*90}")
    print(f"  Best feature set: {best['features']}")
    print(f"  Number of features: {best['n']}")
    print(f"  RMSE:     {best['rmse']:.6f}")
    print(f"  MDD:      {best['mdd']:.4f}")
    print(f"  Sortino:  {best['sortino']:.4f}")
    print(f"  Removed:  {ALL_FEATURES or 'none'}")
    removed = [f for f in ALL_FEATURES if f not in best['features'].split('+')]
    print(f"  Removed features: {removed if removed else 'none'}")

    return history, best

# Run for both markets
nifty_history, nifty_best = backward_elimination('NIFTY50')
sp500_history, sp500_best = backward_elimination('SP500')

# Save results
nifty_df = pd.DataFrame(nifty_history)
sp500_df = pd.DataFrame(sp500_history)
nifty_df.to_csv('reports/feature_selection/NIFTY50_selection.csv', index=False)
sp500_df.to_csv('reports/feature_selection/SP500_selection.csv', index=False)

# Summary
print(f"\n{'='*90}")
print(f"  CROSS-MARKET SUMMARY")
print(f"{'='*90}")
print(f"\n  NIFTY 50:")
print(f"    Best features: {nifty_best['features']}")
print(f"    RMSE={nifty_best['rmse']:.6f}  MDD={nifty_best['mdd']:.4f}  Sortino={nifty_best['sortino']:.4f}")

print(f"\n  S&P 500:")
print(f"    Best features: {sp500_best['features']}")
print(f"    RMSE={sp500_best['rmse']:.6f}  MDD={sp500_best['mdd']:.4f}  Sortino={sp500_best['sortino']:.4f}")

print(f"\n  Saved results to reports/feature_selection/")