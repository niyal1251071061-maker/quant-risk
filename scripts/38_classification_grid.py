"""
Two-stage classification grid search.
Stage 1: Find best classifier hyperparameters (maximize AUC).
Stage 2: On best classifier, find best threshold params (maximize balanced accuracy).
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
import os, itertools, time, json

os.makedirs('reports/classification_grid', exist_ok=True)

features = ['Log_Return_1d', 'Log_Return_5d', 'RSI', 'Volatility',
            'Vol_Ratio', 'Return_20d', 'HL_Range', 'MA_Gap']
train_window, test_window, inner_val = 1260, 252, 126

# ============================================================
# STAGE 1: Classifier hyperparameter search
# ============================================================
STAGE1_GRID = {
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.03, 0.05, 0.1],
    'n_estimators': [200, 300, 500],
    'min_child_weight': [1, 5],
    'subsample': [0.8, 1.0],
}
# Total: 5 × 3 × 3 × 2 × 2 = 180 configs per market

def train_classifier(df, params, market):
    """Train and return (probs, acts, sigmas)."""
    probs_all, acts_all, sigmas_all = [], [], []
    for i in range(train_window, len(df) - test_window, test_window):
        tr = df.iloc[:i]; te = df.iloc[i:i+test_window]
        tr_inner = tr.iloc[:-inner_val]; val_inner = tr.iloc[-inner_val:]

        dtr = xgb.DMatrix(tr_inner[features], label=tr_inner['Sign'])
        dval = xgb.DMatrix(val_inner[features], label=val_inner['Sign'])
        dte = xgb.DMatrix(te[features])

        xgb_params = {
            'max_depth': params['max_depth'],
            'learning_rate': params['learning_rate'],
            'base_score': 0.5,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'min_child_weight': params['min_child_weight'],
            'subsample': params['subsample'],
        }

        bst = xgb.train(xgb_params, dtr, num_boost_round=params['n_estimators'],
                        evals=[(dval, 'val')], early_stopping_rounds=30,
                        verbose_eval=False)
        best_it = bst.best_iteration + 1 if bst.best_iteration is not None else params['n_estimators']
        probs = bst.predict(dte, iteration_range=(0, best_it))
        probs_all.extend(probs)
        acts_all.extend(te['Sign'].values)
        sigmas_all.extend(te['Volatility'].values)
    return np.array(probs_all), np.array(acts_all), np.array(sigmas_all)

print(f"{'='*90}")
print(f"  STAGE 1: CLASSIFIER HYPERPARAMETER SEARCH")
print(f"{'='*90}")

stage1_results = {}

for market in ['NIFTY50', 'SP500']:
    print(f"\n  {market}")
    print(f"  {'-'*90}")
    df = pd.read_parquet(f'data/{market}_features.parquet')
    df['Sign'] = (df['Target'] > 0).astype(int)

    configs = list(itertools.product(*STAGE1_GRID.values()))
    keys = list(STAGE1_GRID.keys())
    total = len(configs)
    print(f"  Testing {total} configs...")

    best_auc = 0
    best_config = None
    best_data = None
    results = []

    start = time.time()
    for idx, combo in enumerate(configs):
        params = dict(zip(keys, combo))
        try:
            probs, acts, sigmas = train_classifier(df, params, market)
            auc = roc_auc_score(acts, probs)
            results.append({**params, 'auc': auc})
            if auc > best_auc:
                best_auc = auc
                best_config = params
                best_data = (probs, acts, sigmas)
            if (idx + 1) % 20 == 0:
                elapsed = time.time() - start
                eta = (elapsed / (idx + 1)) * (total - idx - 1)
                print(f"    [{idx+1}/{total}] Best AUC so far: {best_auc:.4f} | ETA {eta/60:.1f}m")
        except Exception as e:
            print(f"    ERROR: {params}: {e}")
            continue

    print(f"\n  ✅ Best for {market}: AUC = {best_auc:.4f}")
    print(f"     Config: {best_config}")

    # Save stage 1 results
    pd.DataFrame(results).sort_values('auc', ascending=False).to_csv(
        f'reports/classification_grid/{market}_stage1.csv', index=False)

    # ============================================================
    # STAGE 2: Threshold optimization on best classifier
    # ============================================================
    print(f"\n  STAGE 2: Threshold optimization for {market}")

    probs, acts, sigmas = best_data
    sigma_ref = 0.14
    best_bal_acc = 0
    best_thr_config = None
    stage2_results = []

    print(f"  {'-'*90}")
    print(f"  {'beta':>6} | {'base_thr':>9} | {'Acc':>7} | {'Prec':>7} | {'Rec':>7} | {'F1':>7} | {'BalAcc':>8}")
    print(f"  {'-'*90}")

    for beta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        for base_thr in [0.45, 0.48, 0.50, 0.52, 0.55]:
            thr = np.array([base_thr + beta * (s / sigma_ref - 1.0) for s in sigmas])
            thr = np.clip(thr, 0.35, 0.70)
            preds = (probs > thr).astype(int)
            bal_acc = balanced_accuracy_score(acts, preds)
            acc = accuracy_score(acts, preds)
            prec = precision_score(acts, preds, zero_division=0)
            rec = recall_score(acts, preds, zero_division=0)
            f1 = f1_score(acts, preds, zero_division=0)

            stage2_results.append({
                'beta': beta, 'base_thr': base_thr,
                'accuracy': acc, 'precision': prec, 'recall': rec,
                'f1': f1, 'balanced_acc': bal_acc
            })

            if bal_acc > best_bal_acc:
                best_bal_acc = bal_acc
                best_thr_config = {'beta': beta, 'base_thr': base_thr}

    # Print top 10 threshold configs
    stage2_df = pd.DataFrame(stage2_results).sort_values('balanced_acc', ascending=False)
    print(f"\n  Top 10 threshold configs:")
    for _, row in stage2_df.head(10).iterrows():
        print(f"  {row['beta']:>6.1f} | {row['base_thr']:>9.2f} | {row['accuracy']:>7.4f} | "
              f"{row['precision']:>7.4f} | {row['recall']:>7.4f} | {row['f1']:>7.4f} | "
              f"{row['balanced_acc']:>8.4f}")

    print(f"\n  ✅ Best threshold for {market}: β={best_thr_config['beta']}, base={best_thr_config['base_thr']}, BalAcc={best_bal_acc:.4f}")

    stage2_df.to_csv(f'reports/classification_grid/{market}_stage2.csv', index=False)

    stage1_results[market] = {
        'best_config': best_config,
        'best_auc': best_auc,
        'best_thr_config': best_thr_config,
        'best_bal_acc': best_bal_acc,
    }

# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*90}")
print(f"  FINAL SUMMARY")
print(f"{'='*90}")

for market in ['NIFTY50', 'SP500']:
    r = stage1_results[market]
    print(f"\n  {market}:")
    print(f"    Classifier AUC:        {r['best_auc']:.4f}")
    print(f"    Best hyperparameters:  {r['best_config']}")
    print(f"    Threshold β:           {r['best_thr_config']['beta']}")
    print(f"    Threshold base:        {r['best_thr_config']['base_thr']}")
    print(f"    Balanced Accuracy:     {r['best_bal_acc']:.4f}")

with open('reports/classification_grid/final_summary.json', 'w') as f:
    json.dump(stage1_results, f, indent=2, default=str)

print(f"\n  Saved -> reports/classification_grid/")