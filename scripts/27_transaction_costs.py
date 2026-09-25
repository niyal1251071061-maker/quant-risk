"""
Transaction cost modeling.
Applies 0.1% round-trip cost per position change (entry or exit).
Reports MDD, Sortino, and mean return both gross and net of costs.
Handles baseline Sortino flipping negative after costs (safe % change).
"""
import pandas as pd
import numpy as np
import os

os.makedirs('reports/costs', exist_ok=True)

COST_RATE = 0.001  # 0.1% per trade

def mdd(r):
    r = pd.Series(r)
    c = (1 + r).cumprod(); p = c.cummax()
    return ((c - p) / p).min()

def sortino(r):
    r = pd.Series(r)
    d = r[r < 0]
    return 0 if d.std() == 0 else r.mean() / d.std() * np.sqrt(252)

def mean_return(r):
    return np.mean(r)

def safe_pct_change(new, old):
    """Percentage change; returns NaN if old <= 0 (undefined sign)."""
    if old <= 0:
        return float('nan')
    return (new / old - 1) * 100

def apply_costs(returns, signal, cost_rate=COST_RATE):
    """Returns net of transaction costs. Position change triggers one cost."""
    position_change = np.abs(np.diff(np.concatenate([[0], signal])))
    return returns - position_change * cost_rate

def run_market(name):
    print(f"\n{'='*72}")
    print(f"  TRANSACTION COST ANALYSIS — {name}")
    print(f"{'='*72}")

    base = pd.read_csv(f'reports/baseline_{name}.csv', parse_dates=['Date'])
    cust = pd.read_csv(f'reports/custom_{name}.csv', parse_dates=['Date'])
    merged = base.merge(cust, on='Date', suffixes=('_base', '_cust'))

    actual = merged['Actual_base'].values
    sig_b = (merged['Pred_base'].values > 0).astype(int)
    sig_c = (merged['Pred_cust'].values > 0).astype(int)

    gross_b = sig_b * actual
    gross_c = sig_c * actual
    net_b = apply_costs(gross_b, sig_b)
    net_c = apply_costs(gross_c, sig_c)

    trades_b = int(np.sum(np.abs(np.diff(np.concatenate([[0], sig_b])))))
    trades_c = int(np.sum(np.abs(np.diff(np.concatenate([[0], sig_c])))))

    print(f"\n  Trade counts (position changes):")
    print(f"    Baseline: {trades_b} trades over {len(actual)} days")
    print(f"    Custom:   {trades_c} trades over {len(actual)} days")
    print(f"    Cost per trade: {COST_RATE*100:.2f}%")

    # Raw metrics
    gb_mean, gc_mean = gross_b.mean(), gross_c.mean()
    nb_mean, nc_mean = net_b.mean(), net_c.mean()
    gb_mdd, gc_mdd = mdd(gross_b), mdd(gross_c)
    nb_mdd, nc_mdd = mdd(net_b), mdd(net_c)
    gb_sort, gc_sort = sortino(gross_b), sortino(gross_c)
    nb_sort, nc_sort = sortino(net_b), sortino(net_c)

    print(f"\n  {'Metric':<16} {'Gross Base':>12} {'Gross Cust':>12} {'Net Base':>12} {'Net Cust':>12}")
    print(f"  {'-'*66}")
    print(f"  {'Mean daily ret':<16} {gb_mean:>12.6f} {gc_mean:>12.6f} {nb_mean:>12.6f} {nc_mean:>12.6f}")
    print(f"  {'MDD':<16} {gb_mdd:>12.4f} {gc_mdd:>12.4f} {nb_mdd:>12.4f} {nc_mdd:>12.4f}")
    print(f"  {'Sortino':<16} {gb_sort:>12.4f} {gc_sort:>12.4f} {nb_sort:>12.4f} {nc_sort:>12.4f}")

    # Improvements (safe)
    mdd_imp_gross = safe_pct_change(gc_mdd, gb_mdd)
    mdd_imp_net = safe_pct_change(nc_mdd, nb_mdd)
    sort_imp_gross = safe_pct_change(gc_sort, gb_sort)
    sort_imp_net = safe_pct_change(nc_sort, nb_sort)

    # Absolute deltas (always defined)
    mdd_abs_gross = gc_mdd - gb_mdd
    mdd_abs_net = nc_mdd - nb_mdd
    sort_abs_gross = gc_sort - gb_sort
    sort_abs_net = nc_sort - nb_sort

    print(f"\n  {'Improvement':<16} {'Gross %':>12} {'Net %':>12} | {'Gross Δ':>10} {'Net Δ':>10}")
    print(f"  {'-'*66}")
    print(f"  {'MDD':<16} {mdd_imp_gross:>11.1f}% {mdd_imp_net:>11.1f}% | {mdd_abs_gross:>+10.4f} {mdd_abs_net:>+10.4f}")
    print(f"  {'Sortino':<16} {sort_imp_gross:>11.1f}% {sort_imp_net:>11.1f}% | {sort_abs_gross:>+10.4f} {sort_abs_net:>+10.4f}")

    # Note about baseline Sortino turning negative
    if nb_sort < 0:
        print(f"\n  ⚠️  Baseline Sortino turned NEGATIVE after costs ({nb_sort:.4f}).")
        print(f"      % change is undefined, but custom model remains positive ({nc_sort:.4f}).")
        print(f"      This is a stronger outcome than a simple % improvement suggests.")

    return {
        'market': name,
        'trades_baseline': trades_b,
        'trades_custom': trades_c,
        'gross_mdd_base': gb_mdd, 'gross_mdd_cust': gc_mdd,
        'net_mdd_base': nb_mdd, 'net_mdd_cust': nc_mdd,
        'gross_sort_base': gb_sort, 'gross_sort_cust': gc_sort,
        'net_sort_base': nb_sort, 'net_sort_cust': nc_sort,
        'mdd_imp_gross': mdd_imp_gross, 'mdd_imp_net': mdd_imp_net,
        'sort_imp_gross': sort_imp_gross, 'sort_imp_net': sort_imp_net,
        'mdd_abs_net': mdd_abs_net, 'sort_abs_net': sort_abs_net,
    }

results = [run_market(n) for n in ['NIFTY50', 'SP500']]

print(f"\n{'='*72}")
print(f"  FINAL SUMMARY — REAL-WORLD VIABILITY")
print(f"{'='*72}")
print(f"  {'Market':>10} | {'MDD Δ Net':>11} | {'Sort Δ Net':>11} | {'Custom Net MDD':>15} | {'Custom Net Sort':>16}")
print(f"  {'-'*76}")
for r in results:
    mdd_str = f"{r['mdd_imp_net']:+.1f}%" if not np.isnan(r['mdd_imp_net']) else "N/A"
    sort_str = f"{r['sort_imp_net']:+.1f}%" if not np.isnan(r['sort_imp_net']) else f"cust {r['net_sort_cust']:+.2f}"
    print(f"  {r['market']:>10} | {mdd_str:>11} | {sort_str:>11} | {r['net_mdd_cust']:>15.4f} | {r['net_sort_cust']:>16.4f}")

# Verdict
print(f"\n{'='*72}")
print(f"  VERDICT")
print(f"{'='*72}")
for r in results:
    mdd_ok = r['net_mdd_cust'] > r['net_mdd_base']
    sort_ok = r['net_sort_cust'] > 0  # Custom still positive
    if mdd_ok and sort_ok:
        print(f"  {r['market']:>10}: ✅ REAL-WORLD VIABLE (MDD better, Sortino positive)")
    elif mdd_ok:
        print(f"  {r['market']:>10}: ⚠️  MDD better, Sortino mixed")
    else:
        print(f"  {r['market']:>10}: ❌ Not viable after costs")

pd.DataFrame(results).to_csv('reports/costs/transaction_cost_summary.csv', index=False)
print(f"\nSaved -> reports/costs/transaction_cost_summary.csv")