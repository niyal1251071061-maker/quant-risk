# System Architecture

## High-Level Architecture
Data Ingestion (yfinance) 
↓
Feature Engineering (Pandas/NumPy)
↓
Walk-Forward Validation Engine
↓
Custom AVP-Loss (XGBoost)
↓
FastAPI Backend (Caching layer)
↓
Streamlit Dashboard (Plotly)

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

### 3. `docs/DESIGN.md`
```markdown
# Design System

## Style
Modern, Quantitative, Data-Dense

## Dashboard Layout
1. **Header:** Ticker selection (NIFTY 50, S&P 500)
2. **Main Panel:**
   - Price Chart (Plotly Candlestick)
   - Volatility Overlay (14-day rolling)
3. **Risk Metrics Card:**
   - Current Regime (Low/High Volatility)
   - Model Signal (Buy/Neutral/Sell)
   - Confidence Interval
   - Maximum Drawdown
4. **Warning Banner:** Displayed if Volatility > 35% or Confidence < 55%.

## Color Palette
- Primary: #2962FF (Blue)
- Success: #00C853 (Green)
- Warning: #FF6D00 (Orange)
- Danger: #D50000 (Red)
- Background: #0E1117 (Dark, Streamlit default)
- Text: #FAFAFA

## UX Requirements
- Interactive tooltips on charts.
- Clear "Not Financial Advice" disclaimer in footer.
- Responsive layout.
- Loading spinners during model inference.