## Final Review 1 Deliverables (24/09/2026)

### Winning Configs
- NIFTY 50 : depth=6, λ=0.50, lr=0.05, reg=1.0
- S&P 500  : depth=8, λ=0.50, lr=0.10, reg=1.0

### Headline Results
| Market  | Base MDD | Cust MDD | Δ MDD  | Δ Sortino |
|---------|---------:|---------:|-------:|----------:|
| NIFTY50 |  -0.3183 |  -0.1693 | +46.8% |  +74.5%   |
| SP500   |  -0.3247 |  -0.2809 | +13.5% | +133.1%   |

### Cross-Market Generalization
The framework reduces Maximum Drawdown on BOTH major indices. Optimal
lambda and tree depth must be market-adaptive due to differing volatility
distributions (NIFTY σ ≈ 20%, S&P σ ≈ 16%).

### Reproducible via
`python scripts/18_final_results.py`