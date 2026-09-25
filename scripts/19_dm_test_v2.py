"""
Diebold-Mariano test v2 — Correct specification.
Tests whether custom model's STRATEGY RETURNS are statistically better
than baseline strategy returns (not whether RMSE improved).
"""
import pandas as pd
import numpy as np
from scipy import stats
import os

os.makedirs('reports/statistics', exist_ok=True)

def dm_test(loss_base, loss_custom, h=1):
    """DM test — Harvey-Leybourne-Newbold corrected."""
    d = loss_base - loss_custom
    n = len(d)
    d_mean = d.mean()
    d_var = d.var(ddof=1)
    
    gamma0 = d_var
    gamma_sum = 0
    for k in range(1, h):
        gamma_k = ((d[k:] - d_mean) * (d[:-k] - d_mean)).mean()
        gamma_sum += (1 - k / h) * gamma_k * 2
    var_d = (gamma0 + gamma_sum) / n
    
    dm_stat = d_mean / np.sqrt(var_d)
    correction = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    dm_stat_corrected = dm_stat * correction
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat_corrected), df=n-1))
    
    return {'dm_stat': dm_stat_corrected, 'p_value': p_value, 'n': n, 'd_mean': d_mean}

def run_market(name):
    print(f"\n{'='*70}")
    print(f"  DM TEST — {name} — Strategy Returns")
    print(f"{'='*70}")
    
    base = pd.read_csv(f'reports/baseline_{name}.csv', parse_dates=['Date'])
    cust = pd.read_csv(f'reports/custom_{name}.csv', parse_dates=['Date'])
    
    merged = base.merge(cust, on='Date', suffixes=('_base', '_cust'))
    
    actual = merged['Actual_base'].values
    ret_b = np.where(merged['Pred_base'].values > 0, 1, 0) * actual
    ret_c = np.where(merged['Pred_cust'].values > 0, 1, 0) * actual
    
    # Loss = negative return (so lower loss = higher return)
    loss_b = -ret_b
    loss_c = -ret_c
    
    result = dm_test(loss_b, loss_c)
    
    print(f"  Observations:              {result['n']}")
    print(f"  Mean daily return (baseline): {ret_b.mean():+.6f}")
    print(f"  Mean daily return (custom):   {ret_c.mean():+.6f}")
    print(f"  Mean loss diff (B - C):    {result['d_mean']:+.8f}")
    print(f"  DM statistic:              {result['dm_stat']:+.4f}")
    print(f"  P-value (2-tailed):        {result['p_value']:.4f}")
    
    if result['p_value'] < 0.01:
        sig = "*** p < 0.01 — Highly significant"
    elif result['p_value'] < 0.05:
        sig = "**  p < 0.05 — Significant"
    elif result['p_value'] < 0.10:
        sig = "*   p < 0.10 — Weakly significant"
    else:
        sig = "    p >= 0.10 — Not significant"
    print(f"  Significance:              {sig}")
    
    if result['d_mean'] > 0 and result['p_value'] < 0.05:
        print(f"  Interpretation: Custom returns are SIGNIFICANTLY HIGHER.")
    elif result['d_mean'] > 0:
        print(f"  Interpretation: Custom returns higher but not significant.")
    else:
        print(f"  Interpretation: Baseline returns higher.")
    
    return {'market': name, **result, 
            'mean_ret_base': ret_b.mean(), 'mean_ret_cust': ret_c.mean()}

results = [run_market(n) for n in ['NIFTY50', 'SP500']]

print(f"\n{'='*70}")
print(f"  DM TEST SUMMARY — STRATEGY RETURNS")
print(f"{'='*70}")
print(f"{'Market':>10} | {'DM Stat':>10} | {'P-Value':>10} | {'Significant?':>15}")
print("-" * 60)
for r in results:
    sig = "YES" if r['p_value'] < 0.05 else ("WEAK" if r['p_value'] < 0.10 else "NO")
    print(f"{r['market']:>10} | {r['dm_stat']:>+10.4f} | {r['p_value']:>10.4f} | {sig:>15}")

pd.DataFrame(results).to_csv('reports/statistics/dm_test_v2_results.csv', index=False)
print(f"\nSaved -> reports/statistics/dm_test_v2_results.csv")