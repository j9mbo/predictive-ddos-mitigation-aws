# calculate_final_metrics.py (corrected version)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import os
import numpy as np

# --- SETTINGS ---
RF_THRESHOLD_FINAL = 0.63
LR_THRESHOLD = 0.23   
RPS_THRESHOLD = 500

# --- FILE PATHS ---
TRAIN_FILE = 'data/final_dataset/split_train.parquet'
VAL_FILE = 'data/final_dataset/split_val.parquet'
BATCH_A_FILE = 'batches/batch_A.parquet'
BATCH_B_FILE = 'batches/batch_B.parquet'
BATCH_C_FILE = 'batches/batch_C.parquet'
BATCH_D_FILE = 'batches/batch_D.parquet'

EXCLUDE = {"ts_min_utc", "label", "coverage_flag", "batch_start_iso", "batch_end_iso", "batch_id"}
def feat_cols(df):
    return [c for c in df.columns if c not in EXCLUDE and not c.startswith("waf_") and not c.endswith("_iso")]

def find_best_threshold(model, val_df_path):
    val_df = pd.read_parquet(val_df_path)
    X_val = val_df[feat_cols(val_df)]; y_val = val_df['label']
    probs_val = model.predict_proba(X_val)[:, 1]
    best_f1 = -1; best_threshold = 0.5
    for threshold in np.arange(0.01, 1.0, 0.01):
        preds = (probs_val >= threshold).astype(int)
        tp = ((preds == 1) & (y_val == 1)).sum(); fp = ((preds == 1) & (y_val == 0)).sum(); fn = ((preds == 0) & (y_val == 1)).sum()
        if (tp + fp == 0) or (tp + fn == 0): continue
        precision = tp / (tp + fp); recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        if f1 >= best_f1:
            best_f1 = f1; best_threshold = threshold
    return best_threshold

def train_models_and_tune(train_path, val_path):
    print(f"--- Навчання моделей на {train_path} ---")
    tr = pd.read_parquet(train_path); Xtr = tr[feat_cols(tr)]; ytr = tr["label"].astype(int)
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42, class_weight="balanced_subsample").fit(Xtr, ytr)
    lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(Xtr, ytr)
    print("Моделі навчено.")
    print(f"\n--- Пошук оптимальних порогів на {val_path} ---")
    rf_threshold = find_best_threshold(rf, val_path)
    lr_threshold = find_best_threshold(lr, val_path)
    print(f"Оптимальний поріг для RF: {rf_threshold:.4f}")
    print(f"Оптимальний поріг для LR: {lr_threshold:.4f}\n")
    return rf, lr, rf_threshold, lr_threshold

def calculate_metrics_for_scenario(df, rf_model, lr_model, rf_thr, lr_thr):
    X = df[feat_cols(df)]
    df['RF_Alarm'] = (rf_model.predict_proba(X)[:, 1] >= rf_thr).astype(int)
    df['LR_Alarm'] = (lr_model.predict_proba(X)[:, 1] >= lr_thr).astype(int)
    df['RPS_Alarm'] = (df['rps'] > RPS_THRESHOLD).astype(int)
    metrics = {}
    benign_minutes = df[df['label'] == 0]
    attack_minutes = df[df['label'] == 1]
    metrics['FP_RF'] = int(benign_minutes['RF_Alarm'].sum())
    metrics['FP_LR'] = int(benign_minutes['LR_Alarm'].sum())
    metrics['FP_RPS'] = int(benign_minutes['RPS_Alarm'].sum())
    metrics['TP_RF'] = int(attack_minutes['RF_Alarm'].sum())
    metrics['TP_LR'] = int(attack_minutes['LR_Alarm'].sum())
    metrics['TP_RPS'] = int(attack_minutes['RPS_Alarm'].sum())
    metrics['total_attack'] = len(attack_minutes)
    return metrics


def main():
    rf_model, lr_model, rf_thr, lr_thr = train_models_and_tune(TRAIN_FILE, VAL_FILE)
    
    print("--- Розрахунок метрик для кожного сценарію ---")
    metrics = {}
    for name, path in [("A", BATCH_A_FILE), ("B", BATCH_B_FILE), ("C", BATCH_C_FILE), ("D", BATCH_D_FILE)]:
        df = pd.read_parquet(path)
        metrics[name] = calculate_metrics_for_scenario(df, rf_model, lr_model, rf_thr, lr_thr)

    fp_rf = metrics['A']['FP_RF'] + metrics['C']['FP_RF']
    fp_lr = metrics['A']['FP_LR'] + metrics['C']['FP_LR']
    fp_rps = metrics['A']['FP_RPS'] + metrics['C']['FP_RPS']
    
    tp_b_rf = f"{metrics['B']['TP_RF']} / {metrics['B']['total_attack']}"
    tp_b_lr = f"{metrics['B']['TP_LR']} / {metrics['B']['total_attack']}"
    tp_b_rps = f"{metrics['B']['TP_RPS']} / {metrics['B']['total_attack']}"
    
    tp_d_rf = f"{metrics['D']['TP_RF']} / {metrics['D']['total_attack']}"
    tp_d_lr = f"{metrics['D']['TP_LR']} / {metrics['D']['total_attack']}"
    tp_d_rps = f"{metrics['D']['TP_RPS']} / {metrics['D']['total_attack']}"

    print("\n--- ДАНІ ДЛЯ ФІНАЛЬНОЇ ТАБЛИЦІ ---")
    print("="*90)
    print(f"{'Model':<25} | {'FP on Benign (A+C)':<25} | {'TP on Known Attack (B)':<25} | {'TP on Unseen Attack (D)':<25}")
    print("-"*90)
    print(f"{'RandomForest (RF)':<25} | {fp_rf:<25} | {tp_b_rf:<25} | {tp_d_rf:<25}")
    print(f"{'Logistic Regression (LR)':<25} | {fp_lr:<25} | {tp_b_lr:<25} | {tp_d_lr:<25}")
    print(f"{'RPS Threshold (500)':<25} | {fp_rps:<25} | {tp_b_rps:<25} | {tp_d_rps:<25}")
    print("="*90)

if __name__ == "__main__":
    main()