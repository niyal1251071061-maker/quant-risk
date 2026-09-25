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
## FTSE 100 Validation (Third Market) - COMPLETE

- Coarse sweep: λ ∈ [0.01, 1000], 80 configs, depths 4 & 6
- Fine sweep: λ ∈ [0.33, 3.0], 40 configs, depth 6
- Best config: λ=2.68, depth=6
- Result: MDD -0.3436 → -0.2907 (+15.4%), Sortino 0.2043 → 0.1341 (-34.4%)
- Verdict: PARTIAL transfer. Framework works on trending markets but not mean-reverting.
- Boundary condition: AVP-Loss assumes volatility signals danger. FTSE is mean-reverting, so this assumption fails.

## Current Level 2 Status
- [x] DM test
- [x] Block bootstrap
- [x] Third market (FTSE 100)
- [ ] Transaction cost modeling
- [ ] Volatility-adaptive λ formula
- [ ] Ablation study
- [ ] Deployment (Streamlit Cloud)
- [ ] IEEE paper draft
## Transaction Cost Analysis — FINAL (0.1% per trade)

### NIFTY 50
- MDD: -0.3183 (base) → -0.1693 (custom) gross; -0.3267 → -0.2069 net
- Sortino: 1.0415 → 1.8114 gross; 0.7389 → 1.3435 net (+81.8%)
- Trades: 339 (base) vs 323 (custom)
- Verdict: ✅ REAL-WORLD VIABLE

### S&P 500
- MDD: -0.3247 → -0.2809 gross; -0.3301 → -0.2823 net
- Sortino: 0.2964 → 0.6909 gross; baseline flips to -0.0594 net, custom stays +0.5279
- Trades: 297 (base) vs 151 (custom) — custom trades 50% less
- Verdict: ✅ REAL-WORLD VIABLE (custom beats a now-negative baseline)

### Key insight
The custom model's reduced trading frequency (especially on S&P 500) naturally lowers cost drag. This is a design advantage of the asymmetric penalty: it avoids chasing noise in volatile periods.
## Volatility-Adaptive Lambda — Results

### Formula
λ_market = λ_ref × (σ_ref / σ_market)

### Average volatilities (nearly identical across markets)
- NIFTY 50: 14.25%
- S&P 500: 14.61%
- FTSE 100: 13.84%

### Adaptive λ values
- NIFTY 50: 0.5000
- S&P 500: 0.4876
- FTSE 100: 0.5148

### Results
| Market | Adaptive λ | MDD | Sortino | Verdict |
|---|---|---|---|---|
| NIFTY 50 | 0.5000 | Better | Better | ✅ Works |
| S&P 500 | 0.4876 | Better | Better | ✅ Works |
| FTSE 100 | 0.5148 | Worse | Worse | ❌ Fails |

### Key research insight
The adaptive λ formula produces near-identical values because all three markets
have similar volatility (~14%). FTSE's failure is NOT volatility-driven — it
is regime-driven (mean-reverting vs trending). Volatility alone cannot
distinguish market regimes.

### Paper contribution
Propose "regime-aware loss" as future work. The volatility-adaptive λ heuristic
works for trending markets; mean-reverting markets require a different penalty
structure incorporating autocorrelation.