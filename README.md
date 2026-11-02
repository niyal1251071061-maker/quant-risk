# Quantitative Risk Mitigation Platform

**Custom Asymmetric Volatility-Penalized Loss (AVP-Loss) for XGBoost**

🌐 **Live Dashboard:** [quant-risk.streamlit.app](https://quant-risk-lhg3hl5ww4utbsagetzapn.streamlit.app)

## Project

Asymmetric loss function that penalizes over-predictions during high-volatility regimes. Tested on NIFTY 50 and S&P 500 with 10 years of daily data.

## Key Results

| Market | MDD Improvement | Sortino Improvement |
|---|---:|---:|
| NIFTY 50 | **+39.3%** | **+13.5%** |
| S&P 500 | **+16.4%** | **+990.0%** |

These results use leakage-free volatility features and clean-data hyperparameter tuning.

- Statistically significant via block bootstrap (n=5000)
- Survives 0.1% transaction costs
- Zero variance across 5 seeds (fully reproducible)
- Direction classification AUC: 0.568 (NIFTY), 0.519 (S&P)

## Stack

Python · XGBoost · Pandas · Streamlit · Plotly · FastAPI · SQLite

## Local application

Initialize the SQLite report database after regenerating reports:

```powershell
python scripts\init_db.py
```

Run the API and dashboard in separate terminals:

```powershell
uvicorn api:app --reload
streamlit run app.py
```

Set `GEMINI_API_KEY` to enable the optional AI Market Sentiment panel. Without a
key, the dashboard displays a neutral availability message and remains usable.

## Documentation

- [PRD](docs/PRD.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Development Rules](docs/RULES.md)
- [Project Memory](docs/MEMORY.md)

## Team

Group SY-H1 — Niyal Lawhale, Divya Natkar, Om Padwal, Nishant Patil  
Guide: Prof. Manisha More  
Vishwakarma Institute of Technology, Pune