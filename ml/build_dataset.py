#!/usr/bin/env python3
# ml/build_dataset.py
# Build dataset splits from three CF-only batches: A (benign), B (attack), C (benign).
# Ensures VAL has both classes: positives from B, negatives from A.

import argparse, os, tempfile, subprocess
import pandas as pd

def _load_parquet(path: str) -> pd.DataFrame:
    if path.startswith("s3://"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet"); tmp.close()
        subprocess.run(["aws", "s3", "cp", path, tmp.name], check=True)
        df = pd.read_parquet(tmp.name); os.unlink(tmp.name); return df
    return pd.read_parquet(path)

def _save_parquet(df: pd.DataFrame, path: str):
    if path.startswith("s3://"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet"); tmp.close()
        df.to_parquet(tmp.name, index=False)
        subprocess.run(["aws", "s3", "cp", tmp.name, path], check=True); os.unlink(tmp.name)
    else:
        df.to_parquet(path, index=False)

def _attach_id(df: pd.DataFrame, bid: str) -> pd.DataFrame:
    df = df.copy(); df["batch_id"] = bid; return df

def main():
    ap = argparse.ArgumentParser(description="Build dataset with mixed-class validation.")
    ap.add_argument("--batch-a", required=True)
    ap.add_argument("--batch-b", required=True)
    ap.add_argument("--batch-c", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--val-pos-min", type=int, default=4, help="Min positives in VAL (from B)")
    ap.add_argument("--val-neg-min", type=int, default=4, help="Min negatives in VAL (from A)")
    args = ap.parse_args()

    A = _attach_id(_load_parquet(args.batch_a), "A").sort_values("ts_min_utc")
    B = _attach_id(_load_parquet(args.batch_b), "B").sort_values("ts_min_utc")
    C = _attach_id(_load_parquet(args.batch_c), "C").sort_values("ts_min_utc")

    # Basic sanity
    for df, name in [(A,"A"),(B,"B"),(C,"C")]:
        if "label" not in df.columns: raise ValueError(f"batch {name} lacks 'label'")
        if "ts_min_utc" not in df.columns: raise ValueError(f"batch {name} lacks 'ts_min_utc'")

    # Select VAL positives from B (label==1), prefer the tail (ближче до події)
    B_pos = B[B["label"]==1]
    val_pos = B_pos.tail(args.val_pos_min) if len(B_pos)>=args.val_pos_min else B_pos

    # Select VAL negatives from A (label==0), теж з хвоста для реалізму
    A_neg = A[A["label"]==0]
    val_neg = A_neg.tail(args.val_neg_min) if len(A_neg)>=args.val_neg_min else A_neg

    # Remove picked VAL rows from their sources to avoid leakage
    B_train_pool = pd.concat([B, val_pos]).drop_duplicates(keep=False) if len(val_pos)>0 else B
    A_train_pool = pd.concat([A, val_neg]).drop_duplicates(keep=False) if len(val_neg)>0 else A

    VAL = pd.concat([val_pos, val_neg], ignore_index=True).sort_values("ts_min_utc")
    TRAIN = pd.concat([A_train_pool, B_train_pool], ignore_index=True).sort_values("ts_min_utc")
    TEST = C.copy()

    ALL = pd.concat([A, B, C], ignore_index=True)

    # Report
    def stats(df):
        return {"rows": len(df), "labels": df["label"].value_counts().to_dict()}
    print(f"[ALL] {stats(ALL)}")
    print(f"[TRAIN] {stats(TRAIN)}")
    print(f"[VAL] {stats(VAL)}")
    print(f"[TEST] {stats(TEST)}")

    # Save
    prefix = args.out_prefix.rstrip("/")
    _save_parquet(ALL,   f"{prefix}/dataset_all.parquet")
    _save_parquet(TRAIN, f"{prefix}/split_train.parquet")
    _save_parquet(VAL,   f"{prefix}/split_val.parquet")
    _save_parquet(TEST,  f"{prefix}/split_test.parquet")
    print(f"[OK] Written: {prefix}/dataset_all.parquet, split_train.parquet, split_val.parquet, split_test.parquet")

if __name__ == "__main__":
    main()
