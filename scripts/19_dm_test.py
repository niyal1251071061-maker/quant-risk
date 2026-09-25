"""
Diebold-Mariano test — statistical significance of custom vs baseline.
Tests whether the custom model's loss improvement is real or by chance.
"""
import pandas as pd
import numpy as np
from scipy import stats
import os

os.makedirs('reports/statistics', exist_ok=True)

def dm_test(loss_base, loss_custom, h=1):
    """
    Diebold-Mariano test.
    H0: E[loss_base] = E[loss_custom]  (no difference)
    H1: E[loss_base] != E[loss_custom] (two-tailed)
    h = forecast horizon (1 for 1-step-ahead)
    """
    d = loss_base - loss_custom
    n = len(d)
    d_mean = d.mean()
    d_var = d.var(ddof=1)
    
    # Newey-West style adjustment (Harvey, Leybourne, Newbold 1997)
    gamma0 = d_var
    gamma_sum = 0
    for k in range(1, h):
        gamma_k = ((d[k:] - d_mean) * (d[:-k] - d_mean)).mean()
        gamma_sum += (1 - k / h) * gamma_k * 2
    var_d = (gamma0 + gamma_sum) / n
    
    dm_stat = d_mean / np.sqrt(var_d)
    
    # Harvey et al. small-sample correction
    correction = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    dm_stat_corrected = dm_stat * correction
    
    # Two-tailed p-value from t-distribution with n-1 df
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat_corrected), df=n-1))
    
    return {
        'dm_stat': dm_stat_corrected,
        'p_value': p_value,
        'n': n,
        'd_mean': d_mean,
    }

def run_market(name):
    print(f"\n{'='*70}")
    print(f"  DIEBOLD-MARIANO TEST — {name}")
    print(f"{'='*70}")
    
    base = pd.read_csv(f'reports/baseline_{name}.csv', parse_dates=['Date'])
    cust = pd.read_csv(f'reports/custom_{name}.csv', parse_dates=['Date'])
    
    # Align on dates
    merged = base.merge(cust, on='Date', suffixes=('_base', '_cust'))
    if len(merged) == 0:
        print(f"  ERROR: No aligned dates for {name}")
        return None
    
    actual = merged['Actual_base'].values
    pred_b = merged['Pred_base'].values
    pred_c = merged['Pred_cust'].values
    
    # Squared losses
    loss_b = (actual - pred_b) ** 2
    loss_c = (actual - pred_c) ** 2
    
    result = dm_test(loss_b, loss_c)
    
    print(f"  Observations:        {result['n']}")
    print(f"  Mean loss diff:      {result['d_mean']:+.8f}")
    print(f"  DM statistic:        {result['dm_stat']:+.4f}")
    print(f"  P-value (2-tailed):  {result['p_value']:.4f}")
    print(f"  Significance:")
    if result['p_value'] < 0.01:
        print(f"    *** p < 0.01 — Highly significant")
    elif result['p_value'] < 0.05:
        print(f"    **  p < 0.05 — Statistically significant")
    elif result['p_value'] < 0.10:
        print(f"    *   p < 0.10 — Weakly significant")
    else:
        print(f"        p >= 0.10 — Not statistically significant")
    
    # Interpretation
    if result['d_mean'] > 0 and result['p_value'] < 0.05:
        print(f"  Interpretation: Custom model has SIGNIFICANTLY LOWER loss.")
    elif result['d_mean'] < 0 and result['p_value'] < 0.05:
        print(f"  Interpretation: Baseline has SIGNIFICANTLY LOWER loss.")
    else:
        print(f"  Interpretation: No significant difference.")
    
    return {'market': name, **result}

# Run both markets
results = []
for name in ['NIFTY50', 'SP500']:
    r = run_market(name)
    if r:
        results.append(r)

print(f"\n{'='*70}")
print(f"  DM TEST SUMMARY")
print(f"{'='*70}")
print(f"{'Market':>10} | {'DM Stat':>10} | {'P-Value':>10} | {'Significant?':>15}")
print("-" * 60)
for r in results:
    sig = "YES (p<0.05)" if r['p_value'] < 0.05 else "NO"
    print(f"{r['market']:>10} | {r['dm_stat']:>+10.4f} | {r['p_value']:>10.4f} | {sig:>15}")

# Save
pd.DataFrame(results).to_csv('reports/statistics/dm_test_results.csv', index=False)
print(f"\nSaved -> reports/statistics/dm_test_results.csv")
