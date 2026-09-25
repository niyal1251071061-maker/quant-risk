import pandas as pd
import numpy as np
import os

os.makedirs('reports/statistics', exist_ok=True)

def mdd(r):
    r = pd.Series(r)
    c = (1 + r).cumprod(); p = c.cummax()
    return ((c - p) / p).min()

def sortino(r):
    r = pd.Series(r)
    d = r[r < 0]
    return 0 if d.std() == 0 else r.mean() / d.std() * np.sqrt(252)

def stationary_bootstrap_indices(n, avg_block=60, seed=None):
    if seed is not None:
        np.random.seed(seed)
    indices = []
    while len(indices) < n:
        block_len = np.random.geometric(1.0 / avg_block)
        start = np.random.randint(0, n)
        for i in range(block_len):
            indices.append((start + i) % n)
            if len(indices) >= n:
                break
    return np.array(indices[:n])

def run_market(name, n_boot=5000, avg_block=60):
    print(f"\n{'='*72}\n  BOOTSTRAP v2 — {name} (n={n_boot}, block={avg_block})\n{'='*72}")
    base = pd.read_csv(f'reports/baseline_{name}.csv', parse_dates=['Date'])
    cust = pd.read_csv(f'reports/custom_{name}.csv', parse_dates=['Date'])
    merged = base.merge(cust, on='Date', suffixes=('_base', '_cust'))
    actual = merged['Actual_base'].values
    ret_b = np.where(merged['Pred_base'].values > 0, 1, 0) * actual
    ret_c = np.where(merged['Pred_cust'].values > 0, 1, 0) * actual

    b_mdd, c_mdd = mdd(ret_b), mdd(ret_c)
    b_sort, c_sort = sortino(ret_b), sortino(ret_c)
    print(f"\n  MDD     — baseline: {b_mdd:+.4f}  custom: {c_mdd:+.4f}")
    print(f"  Sortino — baseline: {b_sort:+.4f}  custom: {c_sort:+.4f}")

    mdd_diffs, sort_diffs = [], []
    n = len(ret_b)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(n, avg_block=avg_block, seed=1000 + b)
        mdd_diffs.append(mdd(ret_b[idx]) - mdd(ret_c[idx]))
        sort_diffs.append(sortino(ret_c[idx]) - sortino(ret_b[idx]))
    mdd_diffs = np.array(mdd_diffs)
    sort_diffs = np.array(sort_diffs)

    # Custom is better for MDD when diff < 0 (b - c < 0)
    mdd_p_better = (mdd_diffs < 0).mean()
    mdd_ci = (np.percentile(mdd_diffs, 2.5), np.percentile(mdd_diffs, 97.5))
    # Custom is better for Sortino when diff > 0 (c - b > 0)
    sort_p_better = (sort_diffs > 0).mean()
    sort_ci = (np.percentile(sort_diffs, 2.5), np.percentile(sort_diffs, 97.5))

    print(f"\n  MDD diff (baseline − custom):")
    print(f"    Mean: {mdd_diffs.mean():+.4f}")
    print(f"    95% CI: [{mdd_ci[0]:+.4f}, {mdd_ci[1]:+.4f}]")
    print(f"    P(custom better): {mdd_p_better:.4f}")

    print(f"\n  Sortino diff (custom − baseline):")
    print(f"    Mean: {sort_diffs.mean():+.4f}")
    print(f"    95% CI: [{sort_ci[0]:+.4f}, {sort_ci[1]:+.4f}]")
    print(f"    P(custom better): {sort_p_better:.4f}")

    return {
        'market': name,
        'mdd_diff_mean': mdd_diffs.mean(),
        'mdd_ci_low': mdd_ci[0], 'mdd_ci_high': mdd_ci[1],
        'mdd_p_custom_better': mdd_p_better,
        'sort_diff_mean': sort_diffs.mean(),
        'sort_ci_low': sort_ci[0], 'sort_ci_high': sort_ci[1],
        'sort_p_custom_better': sort_p_better,
    }

results = [run_market(n) for n in ['NIFTY50', 'SP500']]
pd.DataFrame(results).to_csv('reports/statistics/bootstrap_v2_results.csv', index=False)
print(f"\nSaved -> reports/statistics/bootstrap_v2_results.csv")