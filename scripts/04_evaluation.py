import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = pd.read_csv('reports/baseline_results.csv')
cust = pd.read_csv('reports/custom_results.csv')

def mdd(returns):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()

def sortino(returns):
    downside = returns[returns < 0]
    if downside.std() == 0:
        return 0
    return returns.mean() / downside.std() * np.sqrt(252)

base['Signal'] = np.where(base['Pred'] > 0, 1, 0)
cust['Signal'] = np.where(cust['Pred'] > 0, 1, 0)

base['Strategy'] = base['Signal'] * base['Actual']
cust['Strategy'] = cust['Signal'] * cust['Actual']

print("="*50)
print("RISK METRICS COMPARISON")
print("="*50)
print(f"Baseline MDD:      {mdd(base['Strategy']):.4f}")
print(f"Custom MDD:        {mdd(cust['Strategy']):.4f}")
print(f"Baseline Sortino:  {sortino(base['Strategy']):.4f}")
print(f"Custom Sortino:    {sortino(cust['Strategy']):.4f}")
print("="*50)

plt.figure(figsize=(12,5))
plt.plot((1+base['Strategy']).cumprod(), label='Baseline XGBoost (MSE)')
plt.plot((1+cust['Strategy']).cumprod(), label='Custom AVP-Loss XGBoost')
plt.legend()
plt.title('Cumulative Returns: Baseline vs Custom AVP-Loss')
plt.xlabel('Days')
plt.ylabel('Cumulative Return')
plt.grid(True, alpha=0.3)
plt.savefig('reports/cumulative_returns.png')
print("\nPlot saved to reports/cumulative_returns.png")