## Final Review 1 Results (24/09/2026)

### NIFTY 50
- Baseline: RMSE=0.013218 | MDD=-0.3183 | Sortino=1.0415
- Custom (d=6, λ=0.5): RMSE=0.012970 | MDD=-0.1693 | Sortino=1.8172
- MDD -46.8% | Sortino +74.5%

### S&P 500
- Baseline: RMSE=0.015253 | MDD=-0.3247 | Sortino=0.2964
- Custom (d=8, λ=0.5, lr=0.1, reg=1.0): RMSE=0.014718 | MDD=-0.2809 | Sortino=0.6909
- MDD -13.5% | Sortino +133.1%

### Alt S&P config (MDD-focused)
- d=6, λ=5.0, lr=0.03, reg=0.5: MDD=-0.2123 (-34.6%) | Sortino=0.3429

### Optimization method
Exhaustive grid search: depth ∈ [3-8], λ ∈ [0.1-10], lr ∈ [0.03-0.1], reg ∈ [0.5-2.0]
Total configs evaluated: 1080 (540 per market)