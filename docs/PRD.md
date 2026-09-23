# Product Requirements Document (PRD)

## Product
Quantitative Risk Mitigation Platform

## Problem
Standard machine learning models in finance use symmetric loss functions (like Mean Squared Error). During high-volatility market crashes, these models generate catastrophic false-positive "buy" signals, leading to severe capital drawdowns. There is no lightweight, mathematically robust loss function that dynamically penalizes optimistic predictions based on real-time market volatility.

## Target Users
Retail learners, finance students, and quantitative analysts studying market risk.

## Goal
Create a centralized analytics engine that reduces downside risk by applying a custom Asymmetric Volatility-Penalized Loss (AVP-Loss) to gradient-boosted models.

## Core Features
1. Automated Data Ingestion (10-year OHLCV)
2. Feature Engineering (Log returns, volatility, RSI)
3. Walk-Forward Validation (No data leakage)
4. Custom AVP-Loss XGBoost Model
5. Financial Risk Metrics (MDD, Sortino, Downside Hit Rate)
6. Interactive Dashboard (Streamlit/FastAPI)

## MVP (Minimum Viable Product)
- Data pipeline for NIFTY 50.
- Baseline XGBoost model.
- Custom AVP-Loss implementation.
- Evaluation plots.

## Out of Scope (v1)
- PostgreSQL & Redis (use Parquet/SQLite)
- User Authentication
- Live streaming data
- News sentiment analysis
- Mobile application

## Success Criteria
1. Data cleaned and features engineered.
2. Custom loss implemented in XGBoost.
3. Walk-forward validation loop working.
4. Maximum Drawdown (MDD) of custom model is lower than baseline MSE model.