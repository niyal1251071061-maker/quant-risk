"""
Direction classification: predict sign of next-day return.
Baseline: XGBoost binary:logistic (symmetric cross-entropy).
Custom: Asymmetric classification loss that penalizes over-confident
false "up" predictions during volatility.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import os

os.makedirs('reports/classification', exist_ok=True)

features = ['Log_Return_1d', 'Log_Return_5d', 'RSI', 'Volatility',
            'Vol_Ratio', 'Return_20d', 'HL_Range', 'MA_Gap']
train_window, test_window, inner_val = 1260, 252, 126

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

def make_asym_class_obj(lam, sigma_train):
    """Custom binary classification objective with asymmetric penalty."""
    def avp(preds, dtrain):
        y = dtrain.get_label()  # 0 or 1
        p = sigmoid(preds)
        # Over-confident when predicted prob > 0.5 but actual = 0
        over = (p > 0.5) & (y < 0.5)
        mult = np.where(over, 1.0 + lam * sigma_train, 1.0)
        grad = mult * (p - y)
        hess = mult * p * (1 - p) + 1e-8
        return grad, hess
    return avp

def run_classifier(market, custom=False, lam=0.5):
    df = pd.read_parquet(f'data/{market}_features.parquet')
    df['Sign'] = (df['Target'] > 0).astype(int)

    preds, acts = [], []
    for i in range(train_window, len(df) - test_window, test_window):
        tr = df.iloc[:i]; te = df.iloc[i:i+test_window]
        tr_inner = tr.iloc[:-inner_val]; val_inner = tr.iloc[-inner_val:]

        dtr = xgb.DMatrix(tr_inner[features], label=tr_inner['Sign'])
        dval = xgb.DMatrix(val_inner[features], label=val_inner['Sign'])
        dte = xgb.DMatrix(te[features])

        if custom:
            bst = xgb.train(
                {'max_depth': 6, 'learning_rate': 0.05, 'base_score': 0.5},
                dtr, num_boost_round=300,
                obj=make_asym_class_obj(lam, tr_inner['Volatility'].values),
                evals=[(dval, 'val')], early_stopping_rounds=30,
                custom_metric=lambda p, d: ('logloss', float(-np.mean(d.get_label()*np.log(sigmoid(p)+1e-8) + (1-d.get_label())*np.log(1-sigmoid(p)+1e-8)))),
                maximize=False, verbose_eval=False
            )
        else:
            bst = xgb.train(
                {'max_depth': 6, 'learning_rate': 0.05, 'base_score': 0.5,
                 'objective': 'binary:logistic', 'eval_metric': 'logloss'},
                dtr, num_boost_round=300,
                evals=[(dval, 'val')], early_stopping_rounds=30, verbose_eval=False
            )

        best_it = bst.best_iteration + 1 if bst.best_iteration is not None else 300
        # For custom objective, preds are raw logits; need sigmoid
        raw = bst.predict(dte, iteration_range=(0, best_it))
        probs = sigmoid(raw) if custom else raw
        preds.extend(probs); acts.extend(te['Sign'].values)

    probs = np.array(preds); acts = np.array(acts)
    binary = (probs > 0.5).astype(int)

    return {
        'market': market,
        'model': 'Custom AVP' if custom else 'Baseline',
        'accuracy': accuracy_score(acts, binary),
        'auc': roc_auc_score(acts, probs),
        'precision': precision_score(acts, binary, zero_division=0),
        'recall': recall_score(acts, binary, zero_division=0),
        'baseline_rate': max(acts.mean(), 1 - acts.mean()),
    }

print(f"\n{'='*80}")
print(f"  DIRECTION CLASSIFICATION RESULTS")
print(f"{'='*80}")

results = []
for market in ['NIFTY50', 'SP500']:
    print(f"\n  {market}:")
    r_base = run_classifier(market, custom=False)
    r_cust = run_classifier(market, custom=True, lam=0.5)
    results.extend([r_base, r_cust])
    print(f"    Baseline:     Acc={r_base['accuracy']:.4f}  AUC={r_base['auc']:.4f}  Prec={r_base['precision']:.4f}  Rec={r_base['recall']:.4f}")
    print(f"    Custom AVP:   Acc={r_cust['accuracy']:.4f}  AUC={r_cust['auc']:.4f}  Prec={r_cust['precision']:.4f}  Rec={r_cust['recall']:.4f}")

df = pd.DataFrame(results)
df.to_csv('reports/classification/direction_results.csv', index=False)

print(f"\n{'='*80}")
print(f"  SUMMARY TABLE")
print(f"{'='*80}")
print(f"  {'Market':>10} | {'Model':>12} | {'Accuracy':>10} | {'AUC':>8} | {'Baseline%':>10}")
print(f"  {'-'*66}")
for r in results:
    print(f"  {r['market']:>10} | {r['model']:>12} | {r['accuracy']:>10.4f} | {r['auc']:>8.4f} | {r['baseline_rate']:>10.4f}")

print(f"\n  Saved -> reports/classification/direction_results.csv")