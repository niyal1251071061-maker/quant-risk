import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def mdd(returns):
    returns = pd.Series(returns)
    cum = np.exp(returns.cumsum())
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()

def sortino(returns, mar=0.0):
    returns = pd.Series(returns)
    downside_diff = np.minimum(0.0, returns - mar)
    downside_dev = np.sqrt(np.mean(np.square(downside_diff)))
    if downside_dev == 0:
        return 0
    return (returns.mean() - mar) / downside_dev * np.sqrt(252)

for market in ('NIFTY50', 'SP500'):
    base = pd.read_csv(f'reports/baseline_{market}.csv')
    cust = pd.read_csv(f'reports/custom_{market}.csv')

    base['Signal'] = np.where(base['Pred'] > 0, 1, 0)
    cust['Signal'] = np.where(cust['Pred'] > 0, 1, 0)
    base['Strategy'] = base['Signal'] * base['Actual']
    cust['Strategy'] = cust['Signal'] * cust['Actual']

    print("=" * 50)
    print(f"RISK METRICS COMPARISON — {market}")
    print("=" * 50)
    print(f"Baseline MDD:      {mdd(base['Strategy']):.4f}")
    print(f"Custom MDD:        {mdd(cust['Strategy']):.4f}")
    print(f"Baseline Sortino:  {sortino(base['Strategy']):.4f}")
    print(f"Custom Sortino:    {sortino(cust['Strategy']):.4f}")
    print("=" * 50)

    plt.figure(figsize=(12, 5))
    plt.plot(np.exp(base['Strategy'].cumsum()), label='Baseline XGBoost (MSE)')
    plt.plot(np.exp(cust['Strategy'].cumsum()), label='Custom AVP-Loss XGBoost')
    plt.legend()
    plt.title(f'Cumulative Returns: Baseline vs Custom AVP-Loss ({market})')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Return')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'reports/cumulative_returns_{market}.png')
    plt.close()
    print(f"\nPlot saved to reports/cumulative_returns_{market}.png")