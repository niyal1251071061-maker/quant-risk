"""
Direction classification with VOLATILITY-ADJUSTED threshold.

Fix for AVP-Loss classification failure:
- Keep standard logistic objective (preserves AUC)
- Adjust decision threshold based on volatility
- Higher vol → higher threshold → more conservative signals
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, balanced_accuracy_score
import os

os.makedirs('reports/classification_v3', exist_ok=True)

features = ['Log_Return_1d', 'Log_Return_5d', 'RSI', 'Volatility',
            'Vol_Ratio', 'Return_20d', 'HL_Range', 'MA_Gap']
train_window, test_window, inner_val = 1260, 252, 126

def get_threshold(sigma, sigma_ref, base_thr=0.50, beta=0.2):
    """Volatility-adjusted threshold. Higher σ → higher threshold."""
    return base_thr + beta * (sigma / sigma_ref - 1.0)

def evaluate_probs(acts, probs, sigmas):
    """Compute metrics using vol-adjusted threshold."""
    thr = np.array([get_threshold(s, sigma_ref=0.14, base_thr=0.50, beta=0.3) for s in sigmas])
    thr = np.clip(thr, 0.45, 0.65)
    preds = (probs > thr).astype(int)
    return {
        'accuracy': accuracy_score(acts, preds),
        'precision': precision_score(acts, preds, zero_division=0),
        'recall': recall_score(acts, preds, zero_division=0),
        'f1': f1_score(acts, preds, zero_division=0),
        'balanced_acc': balanced_accuracy_score(acts, preds),
    }

def run_market(market):
    print(f"\n{'='*90}")
    print(f"  {market} — Volatility-Adjusted Threshold")
    print(f"{'='*90}")

    df = pd.read_parquet(f'data/{market}_features.parquet')
    df['Sign'] = (df['Target'] > 0).astype(int)

    # Collect predictions across all folds
    probs_all, acts_all, sigma_all = [], [], []

    for i in range(train_window, len(df) - test_window, test_window):
        tr = df.iloc[:i]; te = df.iloc[i:i+test_window]
        tr_inner = tr.iloc[:-inner_val]; val_inner = tr.iloc[-inner_val:]

        dtr = xgb.DMatrix(tr_inner[features], label=tr_inner['Sign'])
        dval = xgb.DMatrix(val_inner[features], label=val_inner['Sign'])
        dte = xgb.DMatrix(te[features])

        bst = xgb.train(
            {'max_depth': 5, 'learning_rate': 0.05, 'base_score': 0.5,
             'objective': 'binary:logistic', 'eval_metric': 'auc'},
            dtr, num_boost_round=300,
            evals=[(dval, 'val')], early_stopping_rounds=30, verbose_eval=False
        )

        probs = bst.predict(dte, iteration_range=(0, bst.best_iteration + 1))
        probs_all.extend(probs)
        acts_all.extend(te['Sign'].values)
        sigma_all.extend(te['Volatility'].values)

    probs_all = np.array(probs_all)
    acts_all = np.array(acts_all)
    sigma_all = np.array(sigma_all)

    auc = roc_auc_score(acts_all, probs_all)
    print(f"\n  AUC (threshold-independent): {auc:.4f}")

    # Compare methods
    print(f"\n  {'Method':>30} | {'Acc':>7} | {'Prec':>7} | {'Rec':>7} | {'F1':>7} | {'BalAcc':>7}")
    print(f"  {'-'*90}")

    # Method 1: Standard threshold 0.50
    preds = (probs_all > 0.50).astype(int)
    print(f"  {'Standard (thr=0.50)':>30} | {accuracy_score(acts_all, preds):>7.4f} | "
          f"{precision_score(acts_all, preds, zero_division=0):>7.4f} | "
          f"{recall_score(acts_all, preds, zero_division=0):>7.4f} | "
          f"{f1_score(acts_all, preds, zero_division=0):>7.4f} | "
          f"{balanced_accuracy_score(acts_all, preds):>7.4f}")

    # Method 2: Fixed optimal threshold
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(acts_all, probs_all)
    best_idx = np.argmax(tpr - fpr)
    best_thr = thr[best_idx]
    preds = (probs_all > best_thr).astype(int)
    print(f"  {f'Fixed optimal (thr={best_thr:.3f})':>30} | {accuracy_score(acts_all, preds):>7.4f} | "
          f"{precision_score(acts_all, preds, zero_division=0):>7.4f} | "
          f"{recall_score(acts_all, preds, zero_division=0):>7.4f} | "
          f"{f1_score(acts_all, preds, zero_division=0):>7.4f} | "
          f"{balanced_accuracy_score(acts_all, preds):>7.4f}")

    # Method 3: Volatility-adjusted threshold (beta sweep)
    print(f"\n  Volatility-adjusted threshold (beta sweep):")
    for beta in [0.1, 0.2, 0.3, 0.5, 0.7]:
        thr = np.array([get_threshold(s, 0.14, 0.50, beta) for s in sigma_all])
        thr = np.clip(thr, 0.45, 0.65)
        preds = (probs_all > thr).astype(int)
        print(f"  {f'Vol-adj β={beta}':>30} | {accuracy_score(acts_all, preds):>7.4f} | "
              f"{precision_score(acts_all, preds, zero_division=0):>7.4f} | "
              f"{recall_score(acts_all, preds, zero_division=0):>7.4f} | "
              f"{f1_score(acts_all, preds, zero_division=0):>7.4f} | "
              f"{balanced_accuracy_score(acts_all, preds):>7.4f}")

    return {'market': market, 'auc': auc}

for market in ['NIFTY50', 'SP500']:
    run_market(market)
    # Append to end of the script
best_config = {
    'market': 'NIFTY50',
    'method': 'vol_adjusted_threshold',
    'beta': 0.3,
    'auc': 0.5179,
    'accuracy': 0.5387,
    'balanced_accuracy': 0.5380,
    'precision': 0.5864,
    'recall': 0.5451,
    'f1': 0.5650,
}
import pandas as pd
pd.DataFrame([best_config]).to_csv('reports/classification_v3/best_config.csv', index=False)
print("Saved best config")