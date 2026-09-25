"""
Volatility-adaptive lambda heuristic.
Derives optimal lambda for each market from its average volatility.
Formula: lambda_market = lambda_ref * (sigma_ref / sigma_market)

Tests whether the heuristic produces comparable results to grid-searched lambda.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import os

os.makedirs('reports/adaptive', exist_ok=True)

# ---------- Reference market ----------
# NIFTY 50 was found to have optimal lambda = 0.5
LAMBDA_REF = 0.5

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
features = ['Log_Return_1d', 'Log_Return_5d', 'RSI', 'Volatility',
            'Vol_Ratio', 'Return_20d', 'HL_Range', 'MA_Gap']
train_window, test_window, inner_val = 1260, 252, 126

# ---------- Step 1: Compute avg volatility per market ----------
print(f"{'='*72}")
print(f"  STEP 1: COMPUTE AVERAGE VOLATILITY PER MARKET")
print(f"{'='*72}")

markets = ['NIFTY50', 'SP500', 'FTSE100']
vol_avg = {}

for name in markets:
    df = pd.read_parquet(f'data/{name}_features.parquet')
    avg_vol = df['Volatility'].mean()
    vol_avg[name] = avg_vol
    print(f"  {name:>10}: average annualized volatility = {avg_vol:.4f} ({avg_vol*100:.2f}%)")

sigma_ref = vol_avg['NIFTY50']
print(f"\n  Reference market: NIFTY50")
print(f"  sigma_ref = {sigma_ref:.4f}  |  lambda_ref = {LAMBDA_REF}")

# ---------- Step 2: Derive lambda per market ----------
print(f"\n{'='*72}")
print(f"  STEP 2: DERIVE ADAPTIVE LAMBDA")
print(f"{'='*72}")
print(f"  Formula: lambda_market = lambda_ref × (sigma_ref / sigma_market)\n")

lambda_adaptive = {}
for name in markets:
    lam = LAMBDA_REF * (sigma_ref / vol_avg[name])
    lambda_adaptive[name] = lam
    print(f"  {name:>10}: sigma={vol_avg[name]:.4f}  ->  lambda = {lam:.4f}")

# ---------- Step 3: Train each market with adaptive lambda ----------
print(f"\n{'='*72}")
print(f"  STEP 3: TRAIN WITH ADAPTIVE LAMBDA")
print(f"{'='*72}")

results = []

for name in markets:
    lam = lambda_adaptive[name]
    print(f"\n  Training {name} with lambda = {lam:.4f} ...")

    df = pd.read_parquet(f'data/{name}_features.parquet')

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

    # Custom
    preds, acts = [], []
    for i in range(train_window, len(df) - test_window, test_window):
        tr = df.iloc[:i]; te = df.iloc[i:i+test_window]
        tr_inner = tr.iloc[:-inner_val]; val_inner = tr.iloc[-inner_val:]
        dtr = xgb.DMatrix(tr_inner[features], label=tr_inner['Target'])
        dval = xgb.DMatrix(val_inner[features], label=val_inner['Target'])
        dte = xgb.DMatrix(te[features])
        bst = xgb.train(
            {'max_depth': 6, 'learning_rate': 0.05,
             'base_score': 0.0, 'reg_lambda': 1.0},
            dtr, num_boost_round=500,
            obj=make_obj(lam, tr_inner['Volatility'].values),
            evals=[(dval, 'val')], early_stopping_rounds=30, verbose_eval=False
        )
        best_it = bst.best_iteration + 1 if bst.best_iteration is not None else 500
        preds.extend(bst.predict(dte, iteration_range=(0, best_it)))
        acts.extend(te['Target'].values)

    rc = pd.DataFrame({'Actual': acts, 'Pred': preds})
    rc['Strategy'] = (rc['Pred'] > 0).astype(int) * rc['Actual']

    b_mdd = mdd(rb['Strategy'])
    c_mdd = mdd(rc['Strategy'])
    b_sort = sortino(rb['Strategy'])
    c_sort = sortino(rc['Strategy'])
    b_rmse = rmse_fn(rb['Actual'], rb['Pred'])
    c_rmse = rmse_fn(rc['Actual'], rc['Pred'])

    mdd_abs = c_mdd - b_mdd  # negative means custom is better (less negative)
    sort_abs = c_sort - b_sort  # positive means custom is better

    results.append({
        'market': name,
        'lambda': lam,
        'avg_vol': vol_avg[name],
        'baseline_mdd': b_mdd, 'custom_mdd': c_mdd,
        'baseline_sortino': b_sort, 'custom_sortino': c_sort,
        'baseline_rmse': b_rmse, 'custom_rmse': c_rmse,
        'mdd_abs_delta': mdd_abs, 'sort_abs_delta': sort_abs,
    })

# ---------- Step 4: Comparison table ----------
print(f"\n{'='*72}")
print(f"  STEP 4: ADAPTIVE LAMBDA RESULTS")
print(f"{'='*72}")
print(f"  {'Market':>10} | {'lambda':>8} | {'Avg Vol':>8} | {'Base MDD':>10} | {'Cust MDD':>10} | {'Base Sort':>10} | {'Cust Sort':>10}")
print(f"  {'-'*82}")

for r in results:
    print(f"  {r['market']:>10} | {r['lambda']:>8.4f} | {r['avg_vol']:>8.4f} | "
          f"{r['baseline_mdd']:>10.4f} | {r['custom_mdd']:>10.4f} | "
          f"{r['baseline_sortino']:>10.4f} | {r['custom_sortino']:>10.4f}")

# ---------- Step 5: Verdict per market ----------
print(f"\n{'='*72}")
print(f"  STEP 5: VERDICT")
print(f"{'='*72}")

for r in results:
    mdd_better = r['custom_mdd'] > r['baseline_mdd']  # less negative
    sort_better = r['custom_sortino'] > r['baseline_sortino']
    print(f"\n  {r['market']} (lambda={r['lambda']:.4f}):")
    print(f"    MDD     : {r['baseline_mdd']:+.4f} -> {r['custom_mdd']:+.4f}  |  "
          f"Delta = {r['mdd_abs_delta']:+.4f}  ({'BETTER' if mdd_better else 'WORSE'})")
    print(f"    Sortino : {r['baseline_sortino']:+.4f} -> {r['custom_sortino']:+.4f}  |  "
          f"Delta = {r['sort_abs_delta']:+.4f}  ({'BETTER' if sort_better else 'WORSE'})")

    if mdd_better and sort_better:
        print(f"    -> ✅ ADAPTIVE LAMBDA WORKS on both metrics")
    elif mdd_better or sort_better:
        print(f"    -> ⚠️  Partial improvement")
    else:
        print(f"    -> ❌ Framework does not transfer with adaptive lambda")

# ---------- Summary comparison with grid search ----------
print(f"\n{'='*72}")
print(f"  COMPARISON: ADAPTIVE vs GRID-SEARCHED")
print(f"{'='*72}")
print(f"\n  NIFTY 50:  grid λ=0.50  |  adaptive λ={lambda_adaptive['NIFTY50']:.4f}")
print(f"  S&P 500 :  grid λ=0.50  |  adaptive λ={lambda_adaptive['SP500']:.4f}")
print(f"  FTSE 100:  grid λ=2.68  |  adaptive λ={lambda_adaptive['FTSE100']:.4f}")

print(f"\n  Note: If adaptive lambda produces similar results to grid search,")
print(f"  the heuristic is validated and replaces 100+ config grid search.")

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv('reports/adaptive/adaptive_lambda_results.csv', index=False)
print(f"\n  Saved -> reports/adaptive/adaptive_lambda_results.csv")