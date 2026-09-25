"""Characterize market regimes via volatility and autocorrelation."""
import pandas as pd
import numpy as np

markets = ['NIFTY50', 'SP500', 'FTSE100']

print(f"{'='*72}")
print(f"  REGIME CHARACTERIZATION")
print(f"{'='*72}")
print(f"  {'Market':>10} | {'Avg Vol':>10} | {'Autocorr(1)':>12} | {'Trend Persist':>14} | {'Regime':>18}")
print(f"  {'-'*82}")

for name in markets:
    df = pd.read_parquet(f'data/{name}_features.parquet')
    returns = df['Target'].dropna()
    
    avg_vol = df['Volatility'].mean()
    autocorr_1 = returns.autocorr(lag=1)
    
    # Compute fraction of days where return sign is same as previous (trend persistence)
    sign_persist = (np.sign(returns) == np.sign(returns.shift(1))).mean()
    
    # Classify regime
    if sign_persist > 0.52:
        regime = "Trending"
    elif sign_persist < 0.48:
        regime = "Mean-reverting"
    else:
        regime = "Neutral"
    
    print(f"  {name:>10} | {avg_vol:>10.4f} | {autocorr_1:>+12.4f} | {sign_persist:>13.4f} | {regime:>18}")

print(f"\n  Interpretation:")
print(f"  - Autocorr(1) > 0: returns tend to follow previous direction (trending)")
print(f"  - Autocorr(1) < 0: returns reverse previous direction (mean-reverting)")
print(f"  - Sign persistence > 0.52: strong trend")
print(f"  - Sign persistence < 0.48: strong mean-reversion")