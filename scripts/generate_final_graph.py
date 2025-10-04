import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import os

# --- FILE PATHS ---
TRAIN_FILE = 'data/final_dataset/split_train.parquet'
VAL_FILE = 'data/final_dataset/split_val.parquet'
BATCH_A_FILE = 'batches/batch_A.parquet'
BATCH_B_FILE = 'batches/batch_B.parquet'
BATCH_C_FILE = 'batches/batch_C.parquet' # Legitimate surge
BATCH_D_FILE = 'batches/batch_D.parquet'   # Unseen test
RPS_THRESHOLD = 500

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
    print(f"--- Training models on {train_path} ---")
    tr = pd.read_parquet(train_path); Xtr = tr[feat_cols(tr)]; ytr = tr["label"].astype(int)
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42, class_weight="balanced_subsample").fit(Xtr, ytr)
    lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(Xtr, ytr)
    print("Model trained.")
    print(f"\n--- Searching for optimal thresholds on {val_path} ---")
    rf_threshold = find_best_threshold(rf, val_path)
    lr_threshold = find_best_threshold(lr, val_path)
    print(f"Optimal threshold for RF: {rf_threshold:.4f}")
    print(f"Optimal threshold for LR: {lr_threshold:.4f}\n")
    return rf, lr, rf_threshold, lr_threshold

def main():
    rf_model, lr_model, rf_thr, lr_thr = train_models_and_tune(TRAIN_FILE, VAL_FILE)
    
    # Process ALL scenarios, including C
    scenarios_to_plot = {
        "A": BATCH_A_FILE, 
        "B": BATCH_B_FILE, 
        "C": BATCH_C_FILE, 
        "D": BATCH_D_FILE
    }
    results = {}
    for name, path in scenarios_to_plot.items():
        df = pd.read_parquet(path)
        X = df[feat_cols(df)]
        df['RF_Alarm'] = (rf_model.predict_proba(X)[:, 1] >= rf_thr).astype(int)
        df['LR_Alarm'] = (lr_model.predict_proba(X)[:, 1] >= lr_thr).astype(int)
        df['RPS_Alarm'] = (df['rps'] > RPS_THRESHOLD).astype(int)
        results[name] = df
    
    plot_complete_graph(results)

def plot_complete_graph(results):
    df_a, df_b, df_c, df_d = results['A'], results['B'], results['C'], results['D']
    
    # Create a global timeline for ALL four scenarios
    offset = 0
    df_a['Minute_Global'] = np.arange(1, len(df_a) + 1); offset += len(df_a)
    df_b['Minute_Global'] = offset + np.arange(1, len(df_b) + 1); offset += len(df_b)
    df_c['Minute_Global'] = offset + np.arange(1, len(df_c) + 1); offset += len(df_c)
    df_d['Minute_Global'] = offset + np.arange(1, len(df_d) + 1)
    
    full_df = pd.concat([df_a, df_b, df_c, df_d], ignore_index=True)

    fig, ax = plt.subplots(1, 1, figsize=(24, 8)) # Make the plot even longer
    fig.suptitle('Full Experiment Timeline: Model Performance on Known and Unseen Data', fontsize=20, y=0.98)
    
    ax.plot(full_df['Minute_Global'], full_df['rps'], label='RPS', color='#00A8E1', linewidth=2.5, zorder=2)

    # Separators and labels for all four zones
    offset_a = len(df_a)
    offset_b = offset_a + len(df_b)
    offset_c = offset_b + len(df_c)
    ax.axvline(x=offset_a, color='gray', linestyle='--'); ax.axvline(x=offset_b, color='gray', linestyle='--'); ax.axvline(x=offset_c, color='gray', linestyle='--')
    
    ax.text(offset_a/2, 900, 'A: Benign Background', ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    ax.text(offset_a + len(df_b)/2, 900, 'B: Known Attack', ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    ax.text(offset_b + len(df_c)/2, 900, 'C: Benign Flash Crowd', ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    ax.text(offset_c + len(df_d)/2, 900, 'D: Unseen Test Data', ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # ... (the rest of the code for drawing markers remains the same) ...
    df_plot_rf = full_df[full_df['RF_Alarm'] == 1]
    df_plot_lr = full_df[full_df['LR_Alarm'] == 1]
    df_plot_rps = full_df[full_df['RPS_Alarm'] == 1]

    ax.scatter(df_plot_rps['Minute_Global'], df_plot_rps['rps'], marker='^', color='#FF00FF', s=100, label='RPS Alarm', zorder=3)
    ax.scatter(df_plot_lr['Minute_Global'], df_plot_lr['rps'], marker='X', color='#FF4500', s=120, label='LR Alarm', zorder=4, edgecolors='black')
    ax.scatter(df_plot_rf['Minute_Global'], df_plot_rf['rps'], marker='o', color='#39FF14', s=150, label='RF Alarm', zorder=5, edgecolors='black')

    ax.set_xlabel('Time (Minute of Experiment)'); ax.set_ylabel('Requests per Second (RPS)')
    ax.grid(True, linestyle='--', alpha=0.5); ax.legend(loc='upper left')
    ax.set_ylim(bottom=-50, top=1000); ax.set_xlim(left=0)

    output_filename = 'FINAL_COMPLETE_GRAPH.pdf'
    fig.savefig(output_filename, bbox_inches='tight')
    print(f"\nFinal graph: {output_filename}")
    plt.show()

if __name__ == "__main__":
    main()