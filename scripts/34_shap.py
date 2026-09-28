"""
SHAP explainability for AVP-Loss model.
Generates feature importance + force plots for the paper.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('reports/shap', exist_ok=True)

features = ['Log_Return_1d', 'Log_Return_5d', 'RSI', 'Volatility',
            'Vol_Ratio', 'Return_20d', 'HL_Range', 'MA_Gap']

def make_obj(lam, sigma_train):
    def avp(preds, dtrain):
        y = dtrain.get_label()
        mult = np.where(preds > y, 1.0 + lam * sigma_train, 1.0)
        return mult * (preds - y), mult
    return avp

def run_shap(market, depth, lam):
    print(f"\n{'='*72}")
    print(f"  SHAP Analysis — {market}")
    print(f"{'='*72}")

    df = pd.read_parquet(f'data/{market}_features.parquet')

    # Train on the last training window for SHAP analysis
    train_window = 1260
    tr = df.iloc[:train_window]

    dtr = xgb.DMatrix(tr[features], label=tr['Target'])
    bst = xgb.train(
        {'max_depth': depth, 'learning_rate': 0.05,
         'base_score': 0.0, 'reg_lambda': 1.0},
        dtr, num_boost_round=500,
        obj=make_obj(lam, tr['Volatility'].values),
        verbose_eval=False
    )

    print(f"  Model trained on {len(tr)} samples")

    # SHAP explainer
    explainer = shap.TreeExplainer(bst, feature_perturbation='tree_path_dependent')
    X_sample = df[features].iloc[train_window:train_window+1000]  # Use test portion
    shap_values = explainer.shap_values(X_sample)

    print(f"  SHAP values computed for {len(X_sample)} samples")

    # Summary plot (beeswarm)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False, plot_size=(10, 6))
    plt.tight_layout()
    plt.savefig(f'reports/shap/{market}_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> reports/shap/{market}_beeswarm.png")

    # Bar plot (feature importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False, plot_size=(10, 6))
    plt.tight_layout()
    plt.savefig(f'reports/shap/{market}_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> reports/shap/{market}_importance.png")

    # Print top features by mean |SHAP|
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranking = pd.DataFrame({
        'feature': features,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)

    print(f"\n  Feature importance ranking (mean |SHAP|):")
    for _, row in ranking.iterrows():
        print(f"    {row['feature']:<20} {row['mean_abs_shap']:.6f}")

    ranking.to_csv(f'reports/shap/{market}_ranking.csv', index=False)

# Run for both markets
run_shap('NIFTY50', depth=6, lam=0.5)
run_shap('SP500', depth=8, lam=0.5)

print(f"\n{'='*72}")
print(f"  SHAP analysis complete.")
print(f"  Check reports/shap/ for plots and rankings.")
print(f"{'='*72}")