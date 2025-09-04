# ml/train_classifier.py
import os, io, joblib, boto3, pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

load_dotenv()
REGION = os.getenv("AWS_REGION","us-east-1")
FEATURE_BUCKET = os.getenv("S3_LOGS_BUCKET")
FEATURE_PREFIX = os.getenv("S3_FEATURE_PREFIX","feature-store/joined/")
MODEL_KEY = os.getenv("S3_MODELS_PREFIX","models/") + "classifier_v1.joblib"
LABELS_CSV = os.getenv("LABELS_CSV","data/labels/attack_intervals.csv")

s3 = boto3.client("s3", region_name=REGION)

def list_parquet(bucket, prefix):
    ks, token = [], None
    while True:
        kw = dict(Bucket=bucket, Prefix=prefix, MaxKeys=1000)
        if token: kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        for o in r.get("Contents", []):
            if o["Key"].endswith(".parquet"): ks.append(o["Key"])
        if not r.get("IsTruncated"): break
        token = r["NextContinuationToken"]
    return sorted(ks)[-10:]  # останні 10 батчів

def load_parquet_concat(bucket, keys):
    import s3fs, pyarrow.parquet as pq, pyarrow as pa
    fs = s3fs.S3FileSystem()
    tables=[]
    for k in keys:
        with fs.open(f"s3://{bucket}/{k}","rb") as f:
            tables.append(pq.read_table(f))
    return (pa.concat_tables(tables).to_pandas() 
            if tables else pd.DataFrame())

def apply_labels(df, labels_csv):
    df = df.copy()
    df["ts_min_utc"] = pd.to_datetime(df["ts_min_utc"], utc=True)
    df["label"] = 0
    lbl = pd.read_csv(labels_csv)
    for _, r in lbl.iterrows():
        s = pd.to_datetime(r["start_iso"], utc=True)
        e = pd.to_datetime(r["end_iso"], utc=True)
        df.loc[(df["ts_min_utc"]>=s) & (df["ts_min_utc"]<=e), "label"] = int(r["label"])
    return df

def main():
    keys = list_parquet(FEATURE_BUCKET, FEATURE_PREFIX)
    df = load_parquet_concat(FEATURE_BUCKET, keys)
    if df.empty:
        print("No features to train on"); return
    df = apply_labels(df, LABELS_CSV)
    feats = ["rps","cf_uniq_ip","cnt_4xx","cnt_5xx","uri_entropy","waf_uniq_ip","waf_geo_entropy"]
    X = df[feats].fillna(0)
    y = df["label"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=300))])
    pipe.fit(Xtr, ytr)
    print("Train baseline classifier OK. Test positive rate:", yte.mean())
    by = io.BytesIO(); joblib.dump(pipe, by); by.seek(0)
    s3.put_object(Bucket=FEATURE_BUCKET, Key=MODEL_KEY, Body=by.getvalue())
    print(f"Saved model → s3://{FEATURE_BUCKET}/{MODEL_KEY}")

if __name__ == "__main__":
    main()
