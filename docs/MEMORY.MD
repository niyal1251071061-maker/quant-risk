# Project Memory

## Current Status
Review 1 Preparation. Data pipeline complete. Baseline and custom loss in development.

## Completed
- Project setup & Git initialization.
- 10-year NIFTY 50 data fetched (2014-2024).
- Feature engineering (Log_Return_1d, Log_Return_5d, RSI, Volatility).
- Data saved to `data/nifty_features.parquet`.

## In Progress
- Implementing walk-forward cross-validation loop.
- Deriving mathematical gradient and Hessian for AVP-Loss.

## Known Issues
- Custom XGBoost objective function requires numerical gradient checking.
- Data is currently limited to NIFTY 50. S&P 500 to be added later.

## Next Step
- Complete baseline model and save `reports/baseline_results.csv` for Review 1 submission.