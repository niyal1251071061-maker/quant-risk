# QuantRisk AI — System Architecture

## High-Level Architecture
Data Ingestion (yfinance)
        ↓
Feature Engineering (Pandas/NumPy, past-only)
        ↓
Walk-Forward Validation + AVP-Loss (XGBoost)
        ↓
Report CSVs → SQLite (`quant_risk.db`)
                    ↓
             FastAPI + Pydantic
                    ↓
             Streamlit + Plotly
                    ↑
       Optional Gemini news sentiment

## Technology Stack
- **Language:** Python 3.11
- **Data Processing:** Pandas, NumPy, yfinance
- **Machine Learning:** XGBoost, LightGBM, Scikit-learn
- **Backend:** FastAPI, Uvicorn, SQLite (v1)
- **Frontend:** Streamlit, Plotly
- **Version Control:** Git + GitHub

## Folder Structure
```text
quant-risk/
├── docs/                  # Project documentation
├── data/                  # Parquet datasets
├── notebooks/             # Jupyter notebooks for EDA & prototyping
├── src/                   # Source code
│   ├── data_loader.py
│   ├── features.py
│   ├── walkforward.py
│   ├── loss.py            # Custom AVP-Loss implementation
│   └── metrics.py
├── app.py                 # Streamlit app
├── api.py                 # FastAPI endpoints
└── requirements.txt
```

## API contract

- `GET /api/predict/{ticker}` returns the latest signal and risk metrics.
- `GET /api/metrics/{ticker}` returns baseline/custom MDD, Sortino, Sharpe, and
  improvement percentages.
- `GET /api/predictions/{ticker}` returns dates, actual returns, and custom
  predictions for charting.

The allow-listed tickers are `NIFTY50` and `SP500`. API responses are validated
with Pydantic models before serialization.

## Operational flow

After ML scripts regenerate reports, run `python scripts\init_db.py` to replace
the SQLite report tables. Start the API with `uvicorn api:app --reload` and the
dashboard with `streamlit run app.py`.