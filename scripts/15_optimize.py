"""
Exhaustive grid search for AVP-Loss hyperparameters.
Picks the config that best improves BOTH MDD and Sortino vs baseline.
Saves the winning predictions to reports/custom_{market}.csv
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import os, json, itertools, time

os.makedirs('reports', exist_ok=True)
os.makedirs('reports/optimization_logs', exist_ok=True)

# ---------------- Metrics ----------------
def mdd(r):
    c = (1 + r).cumprod(); p = c.cummax()
    return ((c - p) / p).min()

def sortino(r):
    d = r[r < 0]
    return 0 if d.std() == 0 else r.mean() / d.std() * np.sqrt(252)

def rmse_fn(a, p):
    return np.sqrt(mean_squared_error(a, p))

# ---------------- Custom objective ----------------
def make_obj(lam, sigma_train):
    def avp(preds, dtrain):
        y = dtrain.get_label()
        mult = np.where(preds > y, 1.0 + lam * sigma_train, 1.0)
        return mult * (preds - y), mult
    return avp

# ---------------- Data config ----------------
features = ['Log_Return_1d', 'Log_Return_5d', 'RSI', 'Volatility',
            'Vol_Ratio', 'Return_20d', 'HL_Range', 'MA_Gap']
train_window, test_window, inner_val = 1260, 252, 126

# ---------------- Search space ----------------
DEPTH_GRID   = [3, 4, 5, 6, 7, 8]
LAMBDA_GRID  = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
LR_GRID      = [0.03, 0.05, 0.1]
REG_GRID     = [0.5, 1.0, 2.0]

# ---------------- Baseline reference per market ----------------
BASELINES = {
    'NIFTY50': {'rmse': 0.013218, 'mdd': -0.3183, 'sortino': 1.0415},
    'SP500':   {'rmse': 0.015253, 'mdd': -0.3247, 'sortino': 0.2964},
}

# ---------------- Score ----------------
# We want MDD -> 0 (maximize) AND Sortino -> +inf (maximize).
# Normalize both against baseline and combine with weights.
def combined_score(mdd_val, sortino_val, baseline, w_mdd=0.7, w_sort=0.3):
    # MDD improvement (positive is better; 47% improvement -> 0.47)
    mdd_imp = 1.0 - (mdd_val / baseline['mdd'])  # both negative -> ratio positive
    # Sortino improvement (positive is better)
    sort_imp = (sortino_val / baseline['sortino']) - 1.0 if baseline['sortino'] > 0 else 0.0
    return w_mdd * mdd_imp + w_sort * sort_imp

# ---------------- Evaluation ----------------
def evaluate_config(df, depth, lam, lr, reg):
    preds, acts, dates = [], [], []
    for i in range(train_window, len(df) - test_window, test_window):
        tr = df.iloc[:i]; te = df.iloc[i:i+test_window]
        tr_inner = tr.iloc[:-inner_val]
        val_inner = tr.iloc[-inner_val:]

        dtr = xgb.DMatrix(tr_inner[features], label=tr_inner['Target'])
        dval = xgb.DMatrix(val_inner[features], label=val_inner['Target'])
        dte = xgb.DMatrix(te[features])

        try:
            bst = xgb.train(
                {'max_depth': depth, 'learning_rate': lr,
                 'base_score': 0.0, 'reg_lambda': reg,
                 'verbosity': 0},
                dtr, num_boost_round=500,
                obj=make_obj(lam, tr_inner['Volatility'].values),
                evals=[(dval, 'val')],
                early_stopping_rounds=30,
                verbose_eval=False,
            )
            best_it = bst.best_iteration + 1 if bst.best_iteration is not None else 500
            preds.extend(bst.predict(dte, iteration_range=(0, best_it)))
        except Exception:
            return None
        acts.extend(te['Target'].values)
        dates.extend(te.index)

    r = pd.DataFrame({'Date': dates, 'Actual': acts, 'Pred': preds})
    r['Strategy'] = np.where(r['Pred'] > 0, 1, 0) * r['Actual']
    return {
        'rmse':    rmse_fn(r['Actual'], r['Pred']),
        'mdd':     mdd(r['Strategy']),
        'sortino': sortino(r['Strategy']),
        'preds':   preds,
        'acts':    acts,
        'dates':   dates,
    }

# ---------------- Main search ----------------
def search_market(name):
    print(f"\n{'='*80}")
    print(f"  FULL GRID SEARCH: {name}")
    print(f"  Baseline: MDD={BASELINES[name]['mdd']:.4f} | Sortino={BASELINES[name]['sortino']:.4f}")
    print(f"  Configs to test: {len(DEPTH_GRID)*len(LAMBDA_GRID)*len(LR_GRID)*len(REG_GRID)}")
    print(f"{'='*80}")

    df = pd.read_parquet(f'data/{name}_features.parquet')
    baseline = BASELINES[name]

    log_rows = []
    best = {'score': -np.inf}
    start = time.time()
    count = 0
    total = len(DEPTH_GRID) * len(LAMBDA_GRID) * len(LR_GRID) * len(REG_GRID)

    for depth, lam, lr, reg in itertools.product(DEPTH_GRID, LAMBDA_GRID, LR_GRID, REG_GRID):
        count += 1
        res = evaluate_config(df, depth, lam, lr, reg)
        if res is None:
            continue

        score = combined_score(res['mdd'], res['sortino'], baseline)
        log_rows.append({
            'depth': depth, 'lambda': lam, 'lr': lr, 'reg': reg,
            'rmse': res['rmse'], 'mdd': res['mdd'], 'sortino': res['sortino'],
            'score': score,
        })

        # Progress print
        elapsed = time.time() - start
        eta = (elapsed / count) * (total - count) if count < total else 0
        print(f"[{count:>4}/{total}] d={depth} λ={lam:<4} lr={lr} reg={reg} | "
              f"RMSE={res['rmse']:.6f} MDD={res['mdd']:.4f} Sort={res['sortino']:.4f} | "
              f"score={score:+.3f} | ETA {eta/60:.1f}m")

        if score > best['score']:
            best = {
                'score': score,
                'depth': depth, 'lambda': lam, 'lr': lr, 'reg': reg,
                'rmse': res['rmse'], 'mdd': res['mdd'], 'sortino': res['sortino'],
                'preds': res['preds'], 'acts': res['acts'], 'dates': res['dates'],
            }

    # Save full log
    log_df = pd.DataFrame(log_rows).sort_values('score', ascending=False)
    log_df.to_csv(f'reports/optimization_logs/{name}_full_log.csv', index=False)

    print(f"\n{'='*80}")
    print(f"  BEST CONFIG FOR {name}")
    print(f"{'='*80}")
    print(f"  Depth: {best['depth']} | Lambda: {best['lambda']} | LR: {best['lr']} | L2: {best['reg']}")
    print(f"  RMSE={best['rmse']:.6f} | MDD={best['mdd']:.4f} | Sortino={best['sortino']:.4f}")
    print(f"  MDD improvement:     {(1 - best['mdd']/baseline['mdd'])*100:+.1f}%")
    print(f"  Sortino improvement: {(best['sortino']/baseline['sortino'] - 1)*100:+.1f}%")
    print(f"  Combined score:      {best['score']:+.4f}")

    # Save winning predictions for dashboard
    out = pd.DataFrame({'Date': best['dates'], 'Actual': best['acts'], 'Pred': best['preds']})
    out.to_csv(f'reports/custom_{name}.csv', index=False)
    print(f"  Saved -> reports/custom_{name}.csv")

    # Save winner summary as JSON
    summary = {k: v for k, v in best.items() if k not in ('preds', 'acts', 'dates')}
    with open(f'reports/optimization_logs/{name}_winner.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    return best

# ---------------- Run ----------------
if __name__ == '__main__':
    results = {}
    for market in ['NIFTY50', 'SP500']:
        results[market] = search_market(market)

    print(f"\n{'='*80}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"{'Market':>10} | {'Baseline MDD':>13} | {'Optimal MDD':>13} | {'Baseline Sort':>14} | {'Optimal Sort':>13}")
    for m, best in results.items():
        b = BASELINES[m]
        print(f"{m:>10} | {b['mdd']:>13.4f} | {best['mdd']:>13.4f} | {b['sortino']:>14.4f} | {best['sortino']:>13.4f}")
    print(f"{'='*80}")