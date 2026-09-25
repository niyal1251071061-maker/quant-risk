"""
Transaction cost modeling.
Applies 0.1% round-trip cost per position change (entry or exit).
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

def apply_costs(returns, signal, cost_rate=COST_RATE):
    """Returns net of transaction costs."""
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

    # Gross returns
    gross_b = sig_b * actual
    gross_c = sig_c * actual

    # Net returns after 0.1% per position change
    net_b = apply_costs(gross_b, sig_b)
    net_c = apply_costs(gross_c, sig_c)

    # Count trades
    trades_b = np.sum(np.abs(np.diff(np.concatenate([[0], sig_b]))))
    trades_c = np.sum(np.abs(np.diff(np.concatenate([[0], sig_c]))))

    print(f"\n  Trade counts (position changes):")
    print(f"    Baseline: {int(trades_b)} trades over {len(actual)} days")
    print(f"    Custom:   {int(trades_c)} trades over {len(actual)} days")
    print(f"    Cost per trade: {COST_RATE*100:.2f}%")

    print(f"\n  {'Metric':<18} {'Gross Base':>12} {'Gross Cust':>12} {'Net Base':>12} {'Net Cust':>12}")
    print(f"  {'-'*68}")
    print(f"  {'Mean daily ret':<18} {gross_b.mean():>12.6f} {gross_c.mean():>12.6f} {net_b.mean():>12.6f} {net_c.mean():>12.6f}")
    print(f"  {'MDD':<18} {mdd(gross_b):>12.4f} {mdd(gross_c):>12.4f} {mdd(net_b):>12.4f} {mdd(net_c):>12.4f}")
    print(f"  {'Sortino':<18} {sortino(gross_b):>12.4f} {sortino(gross_c):>12.4f} {sortino(net_b):>12.4f} {sortino(net_c):>12.4f}")

    # Improvement after costs
    mdd_imp_gross = (1 - mdd(gross_c) / mdd(gross_b)) * 100
    mdd_imp_net = (1 - mdd(net_c) / mdd(net_b)) * 100
    sort_imp_gross = (sortino(gross_c) / sortino(gross_b) - 1) * 100
    sort_imp_net = (sortino(net_c) / sortino(net_b) - 1) * 100

    print(f"\n  {'Improvement':<18} {'Gross':>12} {'Net (after cost)':>20}")
    print(f"  {'-'*52}")
    print(f"  {'MDD Δ':<18} {mdd_imp_gross:>11.1f}% {mdd_imp_net:>19.1f}%")
    print(f"  {'Sortino Δ':<18} {sort_imp_gross:>11.1f}% {sort_imp_net:>19.1f}%")

    return {
        'market': name,
        'trades_baseline': int(trades_b),
        'trades_custom': int(trades_c),
        'gross_mdd_base': mdd(gross_b), 'gross_mdd_cust': mdd(gross_c),
        'net_mdd_base': mdd(net_b), 'net_mdd_cust': mdd(net_c),
        'gross_sort_base': sortino(gross_b), 'gross_sort_cust': sortino(gross_c),
        'net_sort_base': sortino(net_b), 'net_sort_cust': sortino(net_c),
        'mdd_imp_gross': mdd_imp_gross, 'mdd_imp_net': mdd_imp_net,
        'sort_imp_gross': sort_imp_gross, 'sort_imp_net': sort_imp_net,
    }

results = [run_market(n) for n in ['NIFTY50', 'SP500']]

print(f"\n{'='*72}")
print(f"  FINAL SUMMARY — REAL-WORLD VIABILITY")
print(f"{'='*72}")
print(f"{'Market':>10} | {'MDD Δ Gross':>12} | {'MDD Δ Net':>12} | {'Sort Δ Gross':>13} | {'Sort Δ Net':>13} | {'Viable?':>10}")
print("-" * 90)
for r in results:
    viable = "YES" if r['mdd_imp_net'] > 0 and r['sort_imp_net'] > 0 else "PARTIAL"
    print(f"{r['market']:>10} | {r['mdd_imp_gross']:>11.1f}% | {r['mdd_imp_net']:>11.1f}% | {r['sort_imp_gross']:>12.1f}% | {r['sort_imp_net']:>12.1f}% | {viable:>10}")

pd.DataFrame(results).to_csv('reports/costs/transaction_cost_summary.csv', index=False)
print(f"\nSaved -> reports/costs/transaction_cost_summary.csv")