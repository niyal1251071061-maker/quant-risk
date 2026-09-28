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
## Regime Characterization (Final Boundary Condition Analysis)

### Metrics per market
| Market | Avg Vol | Autocorr(1) | Sign Persist | AVP-Loss Result |
|---|---|---|---|---|
| NIFTY 50 | 14.25% | -0.0066 | 0.5450 | ✅ Works |
| S&P 500 | 14.61% | -0.1406 | 0.4748 | ✅ Works |
| FTSE 100 | 13.84% | -0.0141 | 0.4994 | ❌ Fails |

### Key insight
The framework works when the market has DIRECTIONAL PREDICTABILITY
(sign persistence ≠ 0.5), regardless of whether that predictability is
trending (NIFTY: 0.545) or mean-reverting (S&P: 0.475).

FTSE 100 (sign persist = 0.4994) is essentially a random walk. There is no
exploitable pattern for the model to learn. Any asymmetric penalty adds
noise on top of noise.

### Boundary condition (for paper)
The AVP-Loss requires sign persistence ≠ 0.5. This is a testable
precondition that can be computed in one line before training.

### Adaptive λ formula status
- Works for trending and mean-reverting markets
- Fails for random-walk markets
- Formula: λ_market = λ_ref × (σ_ref / σ_market)

## Ablation Study (NIFTY 50, depth=6, λ=0.5)

### Feature importance ranking (by MDD contribution)
1. [Top feature] — importance score X
2. [Second] — importance score Y
3. [Third] — importance score Z
...
8. [Least important] — importance score W

### Key findings
- [Summary of which features matter]
- [Whether the model can be simplified]
- [Confirms/refutes the volatility-penalty hypothesis]

## Feature Selection (Corrected)

### NIFTY 50
- All 8 features contribute positively
- Removing any single feature degrades score
- **FINAL: 8 features**

### S&P 500
- RSI removal marginally improves composite score (Sortino 0.32→0.63)
- But MDD slightly worse (-0.2809 → -0.2875)
- Note: analysis ran at depth=6, not optimal depth=8
- **FINAL: Use 8 features for consistency with NIFTY, paper simplicity**
## SHAP Explainability (Level 3.2)

### NIFTY 50 (top 3)
1. Log_Return_1d (0.001374)
2. Volatility (0.001193)
3. Vol_Ratio (0.001155)

### S&P 500 (top 3)
1. Return_20d (0.001607)
2. Log_Return_5d (0.001130)
3. Vol_Ratio (0.001121)

### Key findings
- Vol_Ratio is top-3 in BOTH markets → consistent volatility signal
- NIFTY uses short-term features; S&P uses 20-day momentum
- RSI is bottom-2 in both → redundant
- Confirms volatility-driven nature of the loss
- Figures saved: reports/shap/*.
## Classification Grid Search — FINAL RESULTS (Level 3.3)

### NIFTY 50
- Best AUC: 0.5680
- Best config: max_depth=5, lr=0.03, n_estimators=200, min_child_weight=5, subsample=1.0
- Best threshold: β=0.1, base_thr=0.52
- Balanced Accuracy: 0.5728
- **Verdict: PUBLISHABLE (AUC > 0.55, BalAcc > 0.55)**

### S&P 500
- Best AUC: 0.5189
- Best config: max_depth=7, lr=0.1, n_estimators=200, min_child_weight=5, subsample=1.0
- Best threshold: β=1.0, base_thr=0.55
- Balanced Accuracy: 0.5371
- **Verdict: WEAK BUT POSITIVE**

### Key insight
The AVP-Loss principle transfers to classification as a DECISION-BOUNDARY
adjustment (vol-adjusted threshold), not as a loss-function modification.
NIFTY responds strongly; S&P weakly.

### Files
- reports/classification_grid/NIFTY50_stage1.csv
- reports/classification_grid/NIFTY50_stage2.csv
- reports/classification_grid/SP500_stage1.csv
- reports/classification_grid/SP500_stage2.csv
- reports/classification_grid/final_summary.json
## Multi-Seed Robustness (Level 3.6)

Zero variance across 5 seeds (0, 42, 123, 456, 789).
Reason: XGBoost deterministic tree building with exact splits.
Result: Fully reproducible.

## Crisis Period Analysis (Level 3.5)

### NIFTY 50
- COVID-2020 (77 days):  MDD -31.83% → -15.30% (+51.9% improvement)
- 2022 rate hikes (205 days): MDD -19.45% → -11.84% (+39.2% improvement)

### S&P 500
- COVID-2020 (82 days):  MDD -31.69% → -28.09% (+11.4% improvement)
- 2022 rate hikes (209 days): MDD -22.00% → -15.66% (+28.8% improvement)

### Key finding: Sortino during crises
- NIFTY COVID:    Baseline -0.52 → Custom +1.54
- NIFTY 2022:     Baseline -0.35 → Custom +2.06
- S&P COVID:      Baseline -0.59 → Custom +0.15
- S&P 2022:       Baseline -0.84 → Custom -0.44

**Custom model produces POSITIVE risk-adjusted returns during 3 of 4 crisis windows where baseline is deeply negative.**