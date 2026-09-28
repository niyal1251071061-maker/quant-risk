"""
Crisis period analysis.
Isolates COVID-2020 crash and 2022 rate hikes. Computes MDD, Sortino,
and directional accuracy on those windows only.
"""
import pandas as pd
import numpy as np
import os

os.makedirs('reports/crisis', exist_ok=True)

def mdd(r):
    r = pd.Series(r); c = (1 + r).cumprod(); p = c.cummax()
    return ((c - p) / p).min()

def sortino(r):
    r = pd.Series(r); d = r[r < 0]
    return 0 if d.std() == 0 else r.mean() / d.std() * np.sqrt(252)

# Crisis windows (dates inclusive)
CRISES = {
    'COVID_2020': ('2020-02-01', '2020-05-31'),
    'RATE_HIKES_2022': ('2022-01-01', '2022-10-31'),
}

print(f"{'='*90}")
print(f"  CRISIS PERIOD ANALYSIS")
print(f"{'='*90}")

all_rows = []
for market in ['NIFTY50', 'SP500']:
    base = pd.read_csv(f'reports/baseline_{market}.csv', parse_dates=['Date'])
    cust = pd.read_csv(f'reports/custom_{market}.csv', parse_dates=['Date'])
    base['Strategy'] = (base['Pred'] > 0).astype(int) * base['Actual']
    cust['Strategy'] = (cust['Pred'] > 0).astype(int) * cust['Actual']
    
    print(f"\n{'='*90}")
    print(f"  {market}")
    print(f"{'='*90}")
    print(f"  {'Crisis':>16} | {'Days':>5} | {'Base MDD':>10} | {'Cust MDD':>10} | {'Base Sort':>10} | {'Cust Sort':>10} | {'MDD Δ':>8}")
    print(f"  {'-'*84}")
    
    for crisis_name, (start, end) in CRISES.items():
        b_window = base[(base['Date'] >= start) & (base['Date'] <= end)]
        c_window = cust[(cust['Date'] >= start) & (cust['Date'] <= end)]
        
        if len(b_window) < 5 or len(c_window) < 5:
            print(f"  {crisis_name:>16} | (no data)")
            continue
        
        b_mdd = mdd(b_window['Strategy'])
        c_mdd = mdd(c_window['Strategy'])
        b_sort = sortino(b_window['Strategy'])
        c_sort = sortino(c_window['Strategy'])
        mdd_imp = (1 - c_mdd / b_mdd) * 100 if b_mdd != 0 else 0
        
        print(f"  {crisis_name:>16} | {len(b_window):>5} | {b_mdd:>10.4f} | {c_mdd:>10.4f} | "
              f"{b_sort:>10.4f} | {c_sort:>10.4f} | {mdd_imp:>+7.1f}%")
        
        all_rows.append({
            'market': market, 'crisis': crisis_name, 'days': len(b_window),
            'baseline_mdd': b_mdd, 'custom_mdd': c_mdd,
            'baseline_sortino': b_sort, 'custom_sortino': c_sort,
            'mdd_improvement_pct': mdd_imp,
        })

pd.DataFrame(all_rows).to_csv('reports/crisis/crisis_analysis.csv', index=False)
print(f"\n  Saved -> reports/crisis/crisis_analysis.csv")