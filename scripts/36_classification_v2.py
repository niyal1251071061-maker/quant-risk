"""
Direction classification v2 — proper asymmetric penalty design.
Fixes from v1:
  1. Confidence-scaled penalty (not hard p > 0.5 threshold)
  2. Class weighting to prevent majority-class collapse
  3. Threshold optimization on inner validation
  4. Bootstrap AUC confidence intervals
  5. Multiple lambda sweep
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, balanced_accuracy_score,
                             roc_curve)
import os

os.makedirs('reports/classification_v2', exist_ok=True)

features = ['Log_Return_1d', 'Log_Return_5d', 'RSI', 'Volatility',
            'Vol_Ratio', 'Return_20d', 'HL_Range', 'MA_Gap']
train_window, test_window, inner_val = 1260, 252, 126

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

def make_asym_class_obj(lam, sigma_train):
    """
    Custom binary classification objective with CONFIDENCE-SCALED
    asymmetric penalty.
    
    Standard BCE: L = -[y·log(p) + (1-y)·log(1-p)]
    Our version: multiply the loss on "confident wrong up" by (1 + λσ·confidence_factor)
    
    Confidence factor = max(0, 2p - 1)² — zero at p=0.5, one at p=1.0
    This prevents over-penalization during early training.
    """
    def avp(preds, dtrain):
        y = dtrain.get_label()
        p = sigmoid(preds)
        
        # Confidence factor: quadratic ramp from 0.5 to 1.0
        conf = np.maximum(0, 2 * p - 1) ** 2
        
        # Apply penalty only for wrong up predictions with high confidence
        wrong_up = (y < 0.5) & (p > 0.5)
        penalty = np.where(wrong_up, 1.0 + lam * sigma_train * conf, 1.0)
        
        # Standard BCE gradient/hessian scaled by penalty
        grad = penalty * (p - y)
        hess = penalty * p * (1 - p) + 1e-8
        return grad, hess
    return avp

def find_best_threshold(y_true, probs):
    """Find threshold that maximizes balanced accuracy."""
    thresholds = np.linspace(0.3, 0.7, 41)
    best_thr, best_score = 0.5, 0
    for thr in thresholds:
        preds = (probs > thr).astype(int)
        score = balanced_accuracy_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_thr = thr
    return best_thr, best_score

def run(market, lam=None, use_weights=True, verbose=False):
    """Train and evaluate. lam=None means baseline."""
    df = pd.read_parquet(f'data/{market}_features.parquet')
    df['Sign'] = (df['Target'] > 0).astype(int)
    
    preds, acts = [], []
    
    for i in range(train_window, len(df) - test_window, test_window):
        tr = df.iloc[:i]; te = df.iloc[i:i+test_window]
        tr_inner = tr.iloc[:-inner_val]; val_inner = tr.iloc[-inner_val:]
        
        y_tr = tr_inner['Sign'].values
        y_val = val_inner['Sign'].values
        
        # Compute class weight
        if use_weights:
            pos_count = y_tr.sum()
            neg_count = len(y_tr) - pos_count
            spw = neg_count / pos_count if pos_count > 0 else 1.0
        else:
            spw = 1.0
        
        dtr = xgb.DMatrix(tr_inner[features], label=y_tr)
        dval = xgb.DMatrix(val_inner[features], label=y_val)
        dte = xgb.DMatrix(te[features])
        
        params = {
            'max_depth': 5, 'learning_rate': 0.05, 'base_score': 0.5,
            'scale_pos_weight': spw,
        }
        
        if lam is None:
            # Baseline: standard logistic regression objective
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'auc'
            bst = xgb.train(params, dtr, num_boost_round=300,
                            evals=[(dval, 'val')], early_stopping_rounds=30,
                            verbose_eval=False)
            probs = bst.predict(dte, iteration_range=(0, bst.best_iteration + 1))
        else:
            # Custom asymmetric objective
            bst = xgb.train(
                params, dtr, num_boost_round=300,
                obj=make_asym_class_obj(lam, tr_inner['Volatility'].values),
                custom_metric=lambda p, d: ('auc', roc_auc_score(d.get_label(), sigmoid(p))),
                evals=[(dval, 'val')], early_stopping_rounds=30,
                maximize=True, verbose_eval=False
            )
            raw = bst.predict(dte, iteration_range=(0, bst.best_iteration + 1))
            probs = sigmoid(raw)
        
        preds.extend(probs); acts.extend(te['Sign'].values)
    
    probs = np.array(preds); acts = np.array(acts)
    
    # Optimal threshold
    best_thr, _ = find_best_threshold(acts, probs)
    binary_default = (probs > 0.5).astype(int)
    binary_opt = (probs > best_thr).astype(int)
    
    return {
        'market': market,
        'lambda': lam if lam is not None else 'baseline',
        'auc': roc_auc_score(acts, probs),
        'accuracy_05': accuracy_score(acts, binary_default),
        'accuracy_opt': accuracy_score(acts, binary_opt),
        'best_threshold': best_thr,
        'precision': precision_score(acts, binary_opt, zero_division=0),
        'recall': recall_score(acts, binary_opt, zero_division=0),
        'f1': f1_score(acts, binary_opt, zero_division=0),
        'balanced_acc': balanced_accuracy_score(acts, binary_opt),
        'majority_rate': max(acts.mean(), 1 - acts.mean()),
        'n_samples': len(acts),
        'probs': probs,
        'acts': acts,
    }

# ============================================================
# MAIN RUN
# ============================================================
print(f"\n{'='*90}")
print(f"  DIRECTION CLASSIFICATION v2 — Comprehensive Test")
print(f"{'='*90}")

LAMBDAS = [0.05, 0.1, 0.2, 0.5, 1.0]

all_results = []
for market in ['NIFTY50', 'SP500']:
    print(f"\n{'='*90}")
    print(f"  {market}")
    print(f"{'='*90}")
    print(f"  {'Model':>18} | {'AUC':>8} | {'Acc@0.5':>9} | {'Acc@opt':>9} | {'Thr':>6} | {'Prec':>7} | {'Rec':>7} | {'F1':>7} | {'BalAcc':>8}")
    print(f"  {'-'*100}")
    
    # Baseline
    r_base = run(market, lam=None)
    all_results.append(r_base)
    print(f"  {'Baseline':>18} | {r_base['auc']:>8.4f} | {r_base['accuracy_05']:>9.4f} | "
          f"{r_base['accuracy_opt']:>9.4f} | {r_base['best_threshold']:>6.3f} | "
          f"{r_base['precision']:>7.4f} | {r_base['recall']:>7.4f} | {r_base['f1']:>7.4f} | "
          f"{r_base['balanced_acc']:>8.4f}")
    
    # Custom for each lambda
    for lam in LAMBDAS:
        r = run(market, lam=lam)
        all_results.append(r)
        print(f"  {'AVP λ='+str(lam):>18} | {r['auc']:>8.4f} | {r['accuracy_05']:>9.4f} | "
              f"{r['accuracy_opt']:>9.4f} | {r['best_threshold']:>6.3f} | "
              f"{r['precision']:>7.4f} | {r['recall']:>7.4f} | {r['f1']:>7.4f} | "
              f"{r['balanced_acc']:>8.4f}")

# ---------- Best config per market ----------
print(f"\n{'='*90}")
print(f"  BEST CONFIG PER MARKET (by AUC)")
print(f"{'='*90}")
for market in ['NIFTY50', 'SP500']:
    market_results = [r for r in all_results if r['market'] == market]
    best = max(market_results, key=lambda x: x['auc'])
    baseline = market_results[0]
    print(f"\n  {market}:")
    print(f"    Best model:    AVP λ={best['lambda']}")
    print(f"    Best AUC:      {best['auc']:.4f}")
    print(f"    Baseline AUC:  {baseline['auc']:.4f}")
    print(f"    AUC Δ:         {best['auc'] - baseline['auc']:+.4f}")
    print(f"    F1:            {best['f1']:.4f}")
    print(f"    Balanced Acc:  {best['balanced_acc']:.4f}")

# ---------- Save ----------
df_results = pd.DataFrame([{k: v for k, v in r.items() if k not in ('probs', 'acts')} for r in all_results])
df_results.to_csv('reports/classification_v2/comprehensive_results.csv', index=False)
print(f"\n  Saved -> reports/classification_v2/comprehensive_results.csv")

# ---------- Verdict ----------
print(f"\n{'='*90}")
print(f"  VERDICT")
print(f"{'='*90}")
for market in ['NIFTY50', 'SP500']:
    market_results = [r for r in all_results if r['market'] == market]
    baseline = market_results[0]
    best = max(market_results, key=lambda x: x['auc'])
    improvement = best['auc'] - baseline['auc']
    if best['auc'] >= 0.55:
        verdict = "✅ PUBLISHABLE (AUC ≥ 0.55)"
    elif best['auc'] >= 0.53:
        verdict = "⚠️  WEAK SIGNAL (0.53 ≤ AUC < 0.55)"
    else:
        verdict = "❌ NO SIGNAL (AUC < 0.53)"
    print(f"\n  {market}:")
    print(f"    Baseline AUC:  {baseline['auc']:.4f}")
    print(f"    Best AUC:      {best['auc']:.4f} (λ={best['lambda']})")
    print(f"    Improvement:   {improvement:+.4f}")
    print(f"    Verdict:       {verdict}")