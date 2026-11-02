import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

os.makedirs('reports', exist_ok=True)

features = ['Log_Return_1d', 'Log_Return_5d', 'RSI', 'Volatility',
            'Vol_Ratio', 'Return_20d', 'HL_Range', 'MA_Gap']
train_window, test_window, inner_val = 1260, 252, 126

CONFIGS = {
    'NIFTY50': {'depth': 6, 'lam': 2.0, 'lr': 0.05, 'reg': 1.0},
    'SP500': {'depth': 4, 'lam': 0.1, 'lr': 0.1, 'reg': 0.5},
}


def make_obj(lam, sigma_train):
    def avp(preds, dtrain):
        y = dtrain.get_label()
        mult = np.where(preds > y, 1.0 + lam * sigma_train, 1.0)
        return mult * (preds - y), mult
    return avp


def train_market(name, cfg):
    print(f"\n{'=' * 64}\nTraining {name} | depth={cfg['depth']}, lambda={cfg['lam']}\n{'=' * 64}")
    df = pd.read_parquet(f'data/{name}_features.parquet')

    preds_b, acts_b, dates_b = [], [], []
    for i in range(train_window, len(df), test_window):
        tr = df.iloc[:i]
        te = df.iloc[i:min(i + test_window, len(df))]
        if len(te) < 20:
            break
        dtr = xgb.DMatrix(tr[features], label=tr['Target'])
        dte = xgb.DMatrix(te[features])
        bst = xgb.train(
            {'max_depth': 4, 'learning_rate': 0.05, 'base_score': 0.0},
            dtr,
            num_boost_round=100,
        )
        preds_b.extend(bst.predict(dte))
        acts_b.extend(te['Target'].values)
        dates_b.extend(te.index)

    rb = pd.DataFrame({'Date': dates_b, 'Actual': acts_b, 'Pred': preds_b})
    rb['Strategy'] = (rb['Pred'] > 0).astype(int) * rb['Actual']
    rb.to_csv(f'reports/baseline_{name}.csv', index=False)

    preds, acts, dates = [], [], []
    for i in range(train_window, len(df), test_window):
        tr = df.iloc[:i]
        te = df.iloc[i:min(i + test_window, len(df))]
        if len(te) < 20:
            break
        tr_inner = tr.iloc[:-inner_val]
        val_inner = tr.iloc[-inner_val:]
        dtr = xgb.DMatrix(tr_inner[features], label=tr_inner['Target'])
        dval = xgb.DMatrix(val_inner[features], label=val_inner['Target'])
        dte = xgb.DMatrix(te[features])
        bst = xgb.train(
            {
                'max_depth': cfg['depth'],
                'learning_rate': cfg['lr'],
                'base_score': 0.0,
                'reg_lambda': cfg['reg'],
            },
            dtr,
            num_boost_round=500,
            obj=make_obj(cfg['lam'], tr_inner['Volatility'].values),
            evals=[(dval, 'val')],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        best_it = bst.best_iteration + 1 if bst.best_iteration is not None else 500
        preds.extend(bst.predict(dte, iteration_range=(0, best_it)))
        acts.extend(te['Target'].values)
        dates.extend(te.index)

    rc = pd.DataFrame({'Date': dates, 'Actual': acts, 'Pred': preds})
    rc['Strategy'] = (rc['Pred'] > 0).astype(int) * rc['Actual']
    rc.to_csv(f'reports/custom_{name}.csv', index=False)

    print(f'Baseline rows: {len(rb)} | Custom rows: {len(rc)}')
    print(f'Baseline RMSE: {np.sqrt(mean_squared_error(rb["Actual"], rb["Pred"])):.6f}')
    print(f'Custom RMSE:   {np.sqrt(mean_squared_error(rc["Actual"], rc["Pred"])):.6f}')


for name, cfg in CONFIGS.items():
    train_market(name, cfg)
