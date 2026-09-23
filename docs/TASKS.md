# Project Tasks

## Phase 1: Setup & Data
- [x] Initialize Git repository
- [x] Fetch 10-year NIFTY 50 data
- [x] Engineer log returns, volatility, RSI features
- [x] Save processed data to Parquet

## Phase 2: Baseline & Walk-Forward
- [ ] Implement expanding walk-forward loop
- [ ] Train XGBoost baseline (MSE)
- [ ] Save baseline predictions
- [ ] Compute baseline RMSE

## Phase 3: Custom AVP-Loss
- [ ] Derive Gradient (g) and Hessian (h)
- [ ] Implement custom objective in XGBoost
- [ ] Numerically check gradients
- [ ] Train custom AVP-Loss model

## Phase 4: Evaluation
- [ ] Compute Maximum Drawdown (MDD)
- [ ] Compute Sortino Ratio
- [ ] Plot Cumulative Returns
- [ ] Plot Drawdown Curves

## Phase 5: Dashboard & API
- [ ] Build FastAPI `/predict` endpoint
- [ ] Build Streamlit dashboard
- [ ] Integrate Plotly chart