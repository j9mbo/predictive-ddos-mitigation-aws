#!/usr/bin/env python3
# ml/train_baseline_rf.py
# Train RF on TRAIN, choose threshold on VAL (max F1),
# report TEST metrics, and optionally also report metrics at a fixed threshold.

import argparse, os, json, tempfile, subprocess, time, joblib, math
from datetime import datetime, timezone
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, roc_auc_score,
    f1_score, precision_score, recall_score, accuracy_score
)

EXCLUDE = {"ts_min_utc","label","coverage_flag","batch_start_iso","batch_end_iso","batch_id"}

def load_parquet(path:str)->pd.DataFrame:
    if path.startswith("s3://"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet"); tmp.close()
        subprocess.run(["aws","s3","cp", path, tmp.name], check=True)
        df = pd.read_parquet(tmp.name)
        os.unlink(tmp.name)
        return df
    return pd.read_parquet(path)

def select_features(df:pd.DataFrame)->pd.DataFrame:
    cols = [c for c in df.columns
            if c not in EXCLUDE and not c.startswith("waf_") and not c.endswith("_iso")]
    return df[cols].copy(), cols

def choose_threshold_val(y_true, y_prob)->float:
    # if there are less than 2 classes in VAL — return 0.5 as a safe default
    if len(np.unique(y_true)) < 2:
        return 0.5
    p, r, thr = precision_recall_curve(y_true, y_prob)
    f1 = (2*p*r)/(p+r+1e-12)
    idx = np.nanargmax(f1[:-1]) if len(thr) > 0 else 0
    return float(thr[idx]) if len(thr) > 0 else 0.5

def eval_metrics(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    out = dict(
        threshold=float(thr),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
    )
    out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    out["roc_auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true))==2 else None
    return out

def eval_fp_specificity(y_true, y_prob, thr):
    from sklearn.metrics import confusion_matrix
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    denom = max(tn+fp, 1)
    return {
        "fp_rate": float(fp/denom),
        "specificity": float(tn/denom),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

def _parse_iso(s:str):
    return datetime.fromisoformat(s.replace("Z","+00:00")) if s else None

def save_artifacts(model, thr, valm, testm, models_prefix, tag, feat_cols):
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = f"{models_prefix.rstrip('/')}/{tag}_{ts}"
    tmpd = tempfile.mkdtemp()
    joblib.dump(dict(model=model, features=feat_cols), os.path.join(tmpd, "model.joblib"))
    for name,obj in [("threshold.json",{"threshold":thr}),
                     ("val_metrics.json",valm),
                     ("test_metrics.json",testm)]:
        with open(os.path.join(tmpd,name),"w") as f: json.dump(obj,f)
    for fname in ["model.joblib","threshold.json","val_metrics.json","test_metrics.json"]:
        subprocess.run(["aws","s3","cp", os.path.join(tmpd,fname), f"{base}/{fname}"], check=True)
    print(f"[OK] artifacts at: {base}")

def main():
    ap = argparse.ArgumentParser(description="Baseline RF (CF-only) with VAL threshold selection + fixed-τ reporting")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val",   required=True)
    ap.add_argument("--test",  required=True)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--models-prefix", default=None, help="s3://.../models (optional)")
    ap.add_argument("--tag", default="baseline_rf")
    # NEW:
    ap.add_argument("--fixed-threshold", type=float, default=None,
                    help="Additionally report metrics at this fixed τ (e.g., 0.343)")
    ap.add_argument("--attack-start-iso", type=str, default=None,
                    help="Start of attack window (ISO-UTC) to measure detection delay on VAL")
    ap.add_argument("--attack-end-iso", type=str, default=None,
                    help="End of attack window (optional)")
    args = ap.parse_args()

    tr = load_parquet(args.train); va = load_parquet(args.val); te = load_parquet(args.test)
    Xtr, feat = select_features(tr); ytr = tr["label"].astype(int).values
    Xva,_     = select_features(va); yva = va["label"].astype(int).values
    Xte,_     = select_features(te); yte = te["label"].astype(int).values

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample"
    )
    clf.fit(Xtr, ytr)

    # VAL with τ selected by max F1
    val_prob = clf.predict_proba(Xva)[:,1]
    thr = choose_threshold_val(yva, val_prob)
    val_metrics = eval_metrics(yva, val_prob, thr)
    print("[VAL]", json.dumps(val_metrics))

    # TEST with the same τ
    test_prob = clf.predict_proba(Xte)[:,1]
    test_metrics = eval_metrics(yte, test_prob, thr)
    print("[TEST]", json.dumps(test_metrics))

    # Fixed-threshold reporting (optional)
    if args.fixed_threshold is not None:
        tau = float(args.fixed_threshold)
        val_fixed = eval_metrics(yva, val_prob, tau)
        print("[VAL_fixed]", json.dumps(val_fixed))

        test_fixed_core  = eval_metrics(yte, test_prob, tau)
        test_fixed_extra = eval_fp_specificity(yte, test_prob, tau)
        test_fixed = {**test_fixed_core, **test_fixed_extra}
        print("[TEST_fixed]", json.dumps(test_fixed))

        # Optional: detection delay within attack window on VAL
        if args.attack_start_iso:
            t0 = _parse_iso(args.attack_start_iso)
            t1 = _parse_iso(args.attack_end_iso) if args.attack_end_iso else None
            delay_min = None
            if "ts_min_utc" in va.columns and t0 is not None:
                va2 = va.copy()
                va2["_ts"] = pd.to_datetime(va2["ts_min_utc"])
                yhat = (val_prob >= tau).astype(int)
                va2["_pred"] = yhat
                m = (va2["_ts"] >= t0)
                if t1 is not None:
                    m &= (va2["_ts"] <= t1)
                m &= (va2["_pred"] == 1)
                if m.any():
                    first_ts = va2.loc[m, "_ts"].min().to_pydatetime()
                    delay_min = max(0.0, (first_ts - t0).total_seconds()/60.0)
            print("[VAL_fixed_delay]", json.dumps({
                "attack_start": args.attack_start_iso,
                "attack_end": args.attack_end_iso,
                "delay_minutes": delay_min
            }))

    if args.models_prefix and args.models_prefix.startswith("s3://"):
        save_artifacts(clf, thr, val_metrics, test_metrics, args.models_prefix, args.tag, feat)

if __name__=="__main__":
    main()
