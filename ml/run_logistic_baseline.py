#!/usr/bin/env python3
"""
Logistic Regression baseline on minute-level CloudFront features.

- Uses TRAIN/VAL/TEST splits with the same 5 CF features:
  ['rps', 'cf_uniq_ip', 'cnt_4xx', 'cnt_5xx', 'cf_rows']
- Standardizes features (z-score) using TRAIN stats.
- Tunes L2 strength C on VAL by maximizing F1.
- Tunes decision threshold on VAL by maximizing F1; freezes threshold (tau_logit).
- Evaluates on TEST (FP-rate, specificity) under strict FP constraint.
- Computes detection delay (minutes) on the B-core window (optional if timestamps present).

Usage example:
  python run_logistic_baseline.py \
    --train /path/split_train.parquet \
    --val   /path/split_val.parquet \
    --test  /path/split_test.parquet \
    --attack-core-start 2025-09-20T17:14:00Z \
    --attack-core-end   2025-09-20T17:22:00Z \
    --out-prefix ./lr_results

If you prefer CSV, you can pass .csv files; script will auto-detect by extension.
"""

import argparse, sys, math, json
from pathlib import Path

import numpy as np
import pandas as pd

# Try Parquet if available; fall back to CSV if extension is .csv
def load_table(path: Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    # Parquet path -> try pyarrow
    try:
        import pyarrow.parquet as pq  # requires pyarrow
        return pd.read_parquet(p)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read {p} as Parquet. "
            f"Install 'pyarrow' or provide CSV. Underlying error: {e}"
        )

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val",   required=True)
    ap.add_argument("--test",  required=True)
    ap.add_argument("--attack-core-start", type=str, default=None,
                    help="UTC ISO8601 for B-core start, e.g. 2025-09-20T17:14:00Z")
    ap.add_argument("--attack-core-end",   type=str, default=None,
                    help="UTC ISO8601 for B-core end,   e.g. 2025-09-20T17:22:00Z")
    ap.add_argument("--out-prefix", type=str, default="./lr_results")
    return ap.parse_args()

FEATURES = ["rps", "cf_uniq_ip", "cnt_4xx", "cnt_5xx", "cf_rows"]
LABEL = "label"

def ensure_columns(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Columns present: {list(df.columns)}")

def zscore_fit(train_X: pd.DataFrame):
    mu = train_X.mean(axis=0)
    sigma = train_X.std(axis=0, ddof=0).replace(0, 1.0)
    return mu, sigma

def zscore_apply(X: pd.DataFrame, mu: pd.Series, sigma: pd.Series):
    return (X - mu) / sigma

def f1_score_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0, precision, recall
    return 2*precision*recall/(precision+recall), precision, recall

def evaluate_threshold(y_true, scores, thr):
    y_pred = (scores >= thr).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    f1, prec, rec = f1_score_from_counts(tp, fp, fn)
    # FP-rate and specificity
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    spec    = tn / (fp + tn) if (fp + tn) > 0 else 1.0
    return dict(
        thr=float(thr), tp=tp, fp=fp, fn=fn, tn=tn,
        f1=f1, precision=prec, recall=rec, fp_rate=fp_rate, specificity=spec
    )

def tune_threshold_on_val(y_val, val_scores, resolution=1001):
    # Sweep thresholds across observed score range; resolution=1001 => step ~0.001
    smin, smax = float(np.min(val_scores)), float(np.max(val_scores))
    if smin == smax:
        # degenerate case -> single threshold at smin
        return dict(best_thr=smin, best=dict(f1=0.0, precision=0.0, recall=0.0))
    best = None
    best_thr = None
    for thr in np.linspace(smin, smax, num=resolution):
        m = evaluate_threshold(y_val, val_scores, thr)
        if (best is None) or (m["f1"] > best["f1"]):
            best = m
            best_thr = thr
    return dict(best_thr=best_thr, best=best)

def compute_detection_delay_minutes(val_df, scores, thr, ts_col="ts_min_utc",
                                    core_start_iso=None, core_end_iso=None):
    """
    Delay = minutes from first B-core minute to first detected positive minute within/after core start.
    Returns integer minutes or None if timestamps/labels missing.
    """
    if ts_col not in val_df.columns or core_start_iso is None or core_end_iso is None:
        return None
    # Parse timestamps
    ts = pd.to_datetime(val_df[ts_col], utc=True, errors="coerce")
    # Build predictions
    y_pred = (scores >= thr).astype(int)
    # Identify B-core window
    core_start = pd.Timestamp(core_start_iso)
    core_end   = pd.Timestamp(core_end_iso)
    # Filter to minutes at/after core_start
    mask_after = ts >= core_start
    if not mask_after.any():
        return None
    ts_after = ts[mask_after]
    pred_after = y_pred[mask_after]
    # First detection at/after core start
    if (pred_after == 1).any():
        t_first_detect = ts_after.loc[pred_after == 1].iloc[0]
        delay = (t_first_detect - core_start) / pd.Timedelta(minutes=1)
        return int(delay)
    else:
        return None

def main():
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.mkdir(parents=True, exist_ok=True)

    # Load splits
    train = load_table(Path(args.train))
    val   = load_table(Path(args.val))
    test  = load_table(Path(args.test))

    # Basic checks
    for df, name in [(train,"TRAIN"), (val,"VAL"), (test,"TEST")]:
        ensure_columns(df, FEATURES + [LABEL])

    # Prepare X/y
    X_train, y_train = train[FEATURES].copy(), train[LABEL].astype(int).values
    X_val,   y_val   = val[FEATURES].copy(),   val[LABEL].astype(int).values
    X_test,  y_test  = test[FEATURES].copy(),  test[LABEL].astype(int).values

    # Standardize (fit on TRAIN; apply to all)
    mu, sigma = zscore_fit(X_train)
    X_train_s = zscore_apply(X_train, mu, sigma)
    X_val_s   = zscore_apply(X_val,   mu, sigma)
    X_test_s  = zscore_apply(X_test,  mu, sigma)

    # Train Logistic Regression (tune C on VAL by F1)
    from sklearn.linear_model import LogisticRegression
    Cs = [0.1, 1.0, 10.0]
    best_C, best_val_f1, best_model = None, -1.0, None

    for C in Cs:
        lr = LogisticRegression(
            penalty="l2", C=C, class_weight="balanced",
            solver="liblinear", max_iter=200
        )
        lr.fit(X_train_s, y_train)
        # Get scores on VAL (use predict_proba for smooth threshold tuning)
        val_scores = lr.predict_proba(X_val_s)[:, 1]
        # Tune threshold on VAL by F1
        tuned = tune_threshold_on_val(y_val, val_scores, resolution=1001)
        if tuned["best"]["f1"] > best_val_f1:
            best_val_f1 = tuned["best"]["f1"]
            best_C = C
            best_model = lr
            best_thr = float(tuned["best_thr"])
            best_val_metrics = tuned["best"]

    # Freeze model (best_C) and threshold (best_thr); evaluate on VAL/TEST
    # Recompute scores with best_model (already fitted on TRAIN)
    val_scores = best_model.predict_proba(X_val_s)[:, 1]
    test_scores = best_model.predict_proba(X_test_s)[:, 1]

    val_eval = evaluate_threshold(y_val, val_scores, best_thr)
    test_eval = evaluate_threshold(y_test, test_scores, best_thr)

    # Detection delay on VAL B-core (if timestamps provided)
    delay_min = None
    if args.attack_core_start and args.attack_core_end:
        # Try common timestamp column names
        ts_cols = [c for c in ["ts_min_utc", "ts_utc", "timestamp", "time"] if c in val.columns]
        ts_col = ts_cols[0] if ts_cols else None
        delay_min = compute_detection_delay_minutes(
            val_df=val, scores=val_scores, thr=best_thr, ts_col=ts_col,
            core_start_iso=args.attack_core_start, core_end_iso=args.attack_core_end
        )

    # Assemble tables
    table_val = pd.DataFrame([{
        "Model": "LogisticRegression",
        "C": best_C,
        "Threshold(VAL)": round(best_thr, 6),
        "Precision(VAL)": round(val_eval["precision"], 4),
        "Recall(VAL)": round(val_eval["recall"], 4),
        "F1(VAL)": round(val_eval["f1"], 4),
        "Delay_min(B-core)": delay_min if delay_min is not None else "n/a",
    }])

    table_test = pd.DataFrame([{
        "Model": "LogisticRegression",
        "FP-rate(TEST)": round(test_eval["fp_rate"], 4),
        "Specificity(TEST)": round(test_eval["specificity"], 4),
        "TP": test_eval["tp"], "FP": test_eval["fp"],
        "FN": test_eval["fn"], "TN": test_eval["tn"],
    }])

    # Save CSVs
    table_val.to_csv(out_prefix / "../data/lr_val_metrics.csv", index=False)
    table_test.to_csv(out_prefix / "../data/lr_test_metrics.csv", index=False)

    # Print concise summary
    print("\n=== Logistic Regression (balanced, L2) ===")
    print(f"Selected C (VAL): {best_C}")
    print(f"Selected threshold τ_logit (VAL): {best_thr:.6f}")
    print("\nVAL @ τ_logit:")
    print(table_val.to_string(index=False))
    print("\nTEST @ τ_logit:")
    print(table_test.to_string(index=False))

if __name__ == "__main__":
    main()
