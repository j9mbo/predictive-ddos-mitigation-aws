import argparse
import pandas as pd

EXCLUDE_IF_PRESENT = ["batch_id"]  # safe to drop if suddenly present

def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def read_labels_csv(path: str) -> pd.DataFrame:
    lab = pd.read_csv(path)
    for c in ["start_utc", "end_utc"]:
        lab[c] = pd.to_datetime(lab[c], utc=True)
    return lab

def label_by_window(df: pd.DataFrame, labels_df: pd.DataFrame, start_iso: str, end_iso: str) -> pd.Series:
    s_utc = pd.to_datetime(start_iso, utc=True)
    e_utc = pd.to_datetime(end_iso, utc=True)
    # take only intervals that intersect with our window
    lab_sub = labels_df[~((labels_df["end_utc"] < s_utc) | (labels_df["start_utc"] > e_utc))].copy()
    y = pd.Series(0, index=df.index, dtype=int)
    for _, r in lab_sub.iterrows():
        if int(r.get("label", 0)) == 1:
            mask = (df["ts_min_utc"] >= r["start_utc"]) & (df["ts_min_utc"] <= r["end_utc"])
            y.loc[mask] = 1
    return y

def main():
    ap = argparse.ArgumentParser(description="CF-only join: label CF parquet by time window and write final parquet.")
    ap.add_argument("--cf-parquet", required=True, help="Path to CF aggregated parquet (per-minute).")
    ap.add_argument("--labels-csv", required=True, help="CSV with start_utc,end_utc,label.")
    ap.add_argument("--start-iso", required=True, help="Window start (ISO UTC).")
    ap.add_argument("--end-iso", required=True, help="Window end (ISO UTC).")
    ap.add_argument("--out-parquet", required=True, help="Output parquet path.")
    args = ap.parse_args()

    cf = read_parquet(args.cf_parquet).copy()
    if "ts_min_utc" not in cf.columns:
        raise ValueError("CF parquet must have 'ts_min_utc' column")
    cf["ts_min_utc"] = pd.to_datetime(cf["ts_min_utc"], utc=True)

    for c in EXCLUDE_IF_PRESENT:
        if c in cf.columns:
            cf = cf.drop(columns=[c])

    s_utc = pd.to_datetime(args.start_iso, utc=True)
    e_utc = pd.to_datetime(args.end_iso, utc=True)
    df = cf[(cf["ts_min_utc"] >= s_utc) & (cf["ts_min_utc"] <= e_utc)].copy()
    df = df.sort_values("ts_min_utc").reset_index(drop=True)

    labels = read_labels_csv(args.labels_csv)
    df["label"] = label_by_window(df, labels, args.start_iso, args.end_iso)

    df["coverage_flag"] = False
    if "cf_rows" in df.columns:
        df.loc[df["cf_rows"] > 0, "coverage_flag"] = True
    elif "rps" in df.columns:
        df.loc[df["rps"] > 0, "coverage_flag"] = True
    df["batch_start_iso"] = args.start_iso
    df["batch_end_iso"] = args.end_iso

    df.to_parquet(args.out_parquet, index=False)
    print(f"[INFO] Final rows: {len(df)}  {df['ts_min_utc'].min()} -> {df['ts_min_utc'].max()}")
    print(f"[INFO] Written final batch parquet: {args.out_parquet}")

if __name__ == "__main__":
    main()
