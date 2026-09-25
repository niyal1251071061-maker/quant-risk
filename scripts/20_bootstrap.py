"""
Block bootstrap confidence intervals for MDD and Sortino.
Tests the actual claim: is the improvement in drawdown / risk-adjusted return
statistically distinguishable from noise?

Method: stationary block bootstrap on the daily strategy returns.
Resamples 1000 times with random block lengths averaging 20 days.
"""
import pandas as pd
import numpy as np
import os

os.makedirs('reports/statistics', exist_ok=True)
np.random.seed(42)

def mdd(r):
    r = pd.Series(r)
    c = (1 + r).cumprod(); p = c.cummax()
    return ((c - p) / p).min()

def sortino(r):
    r = pd.Series(r)
    d = r[r < 0]
    return 0 if d.std() == 0 else r.mean() / d.std() * np.sqrt(252)

def stationary_bootstrap_indices(n, avg_block=20, seed=None):
    """Generate stationary bootstrap sample indices."""
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

def bootstrap_metric(returns, metric_fn, n_boot=1000, avg_block=20):
    n = len(returns)
    samples = []
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(n, avg_block=avg_block, seed=42 + b)
        samples.append(metric_fn(returns[idx]))
    samples = np.array(samples)
    return {
        'mean': samples.mean(),
        'ci_lower': np.percentile(samples, 2.5),
        'ci_upper': np.percentile(samples, 97.5),
        'std': samples.std(),
    }

def run_market(name):
    print(f"\n{'='*72}")
    print(f"  BLOCK BOOTSTRAP — {name}")
    print(f"{'='*72}")

    base = pd.read_csv(f'reports/baseline_{name}.csv', parse_dates=['Date'])
    cust = pd.read_csv(f'reports/custom_{name}.csv', parse_dates=['Date'])
    merged = base.merge(cust, on='Date', suffixes=('_base', '_cust'))

    actual = merged['Actual_base'].values
    ret_b = np.where(merged['Pred_base'].values > 0, 1, 0) * actual
    ret_c = np.where(merged['Pred_cust'].values > 0, 1, 0) * actual

    # Point estimates
    b_mdd = mdd(ret_b)
    c_mdd = mdd(ret_c)
    b_sort = sortino(ret_b)
    c_sort = sortino(ret_c)

    print(f"\n  Point estimates:")
    print(f"    MDD     — baseline: {b_mdd:+.4f}   custom: {c_mdd:+.4f}")
    print(f"    Sortino — baseline: {b_sort:+.4f}   custom: {c_sort:+.4f}")

    # Bootstrap MDD difference
    print(f"\n  Bootstrapping MDD difference (n=1000, avg block=20 days)...")
    mdd_diffs = []
    sort_diffs = []
    for b in range(1000):
        idx = stationary_bootstrap_indices(len(ret_b), avg_block=20, seed=1000 + b)
        mdd_diffs.append(mdd(ret_b[idx]) - mdd(ret_c[idx]))
        sort_diffs.append(sortino(ret_c[idx]) - sortino(ret_b[idx]))
    mdd_diffs = np.array(mdd_diffs)
    sort_diffs = np.array(sort_diffs)

    # MDD is negative; a positive difference (B - C) means custom is LESS negative = better
    print(f"\n  MDD improvement (baseline minus custom):")
    print(f"    Mean:     {mdd_diffs.mean():+.4f}")
    print(f"    95% CI:   [{np.percentile(mdd_diffs, 2.5):+.4f}, {np.percentile(mdd_diffs, 97.5):+.4f}]")
    print(f"    p-value:  {(mdd_diffs <= 0).mean():.4f}  (prob. custom is worse)")

    print(f"\n  Sortino improvement (custom minus baseline):")
    print(f"    Mean:     {sort_diffs.mean():+.4f}")
    print(f"    95% CI:   [{np.percentile(sort_diffs, 2.5):+.4f}, {np.percentile(sort_diffs, 97.5):+.4f}]")
    print(f"    p-value:  {(sort_diffs <= 0).mean():.4f}  (prob. custom is worse)")

    # Interpretation
    mdd_sig = (mdd_diffs > 0).mean() >= 0.95
    sort_sig = (sort_diffs > 0).mean() >= 0.95

    print(f"\n  MDD improvement significant at 5%?    {'YES' if mdd_sig else 'NO'}")
    print(f"  Sortino improvement significant at 5%? {'YES' if sort_sig else 'NO'}")

    return {
        'market': name,
        'baseline_mdd': b_mdd, 'custom_mdd': c_mdd,
        'baseline_sortino': b_sort, 'custom_sortino': c_sort,
        'mdd_diff_mean': mdd_diffs.mean(),
        'mdd_ci_low': np.percentile(mdd_diffs, 2.5),
        'mdd_ci_high': np.percentile(mdd_diffs, 97.5),
        'mdd_p': (mdd_diffs <= 0).mean(),
        'sort_diff_mean': sort_diffs.mean(),
        'sort_ci_low': np.percentile(sort_diffs, 2.5),
        'sort_ci_high': np.percentile(sort_diffs, 97.5),
        'sort_p': (sort_diffs <= 0).mean(),
    }

results = [run_market(n) for n in ['NIFTY50', 'SP500']]

print(f"\n{'='*72}")
print(f"  BOOTSTRAP SUMMARY — 95% CONFIDENCE INTERVALS")
print(f"{'='*72}")
for r in results:
    print(f"\n  {r['market']}:")
    print(f"    MDD diff:     {r['mdd_diff_mean']:+.4f}  CI [{r['mdd_ci_low']:+.4f}, {r['mdd_ci_high']:+.4f}]")
    print(f"    Sortino diff: {r['sort_diff_mean']:+.4f}  CI [{r['sort_ci_low']:+.4f}, {r['sort_ci_high']:+.4f}]")

pd.DataFrame(results).to_csv('reports/statistics/bootstrap_results.csv', index=False)
print(f"\nSaved -> reports/statistics/bootstrap_results.csv")