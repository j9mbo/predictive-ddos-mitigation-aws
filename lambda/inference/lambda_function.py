# lambda/inference/lambda_function.py
import os, io, json, boto3, joblib, pandas as pd

REGION = os.getenv("AWS_REGION","us-east-1")
FEATURE_BUCKET = os.getenv("S3_LOGS_BUCKET")
FEATURE_PREFIX = os.getenv("S3_FEATURE_PREFIX","feature-store/joined/")
MODEL_KEY = os.getenv("MODEL_KEY","models/classifier_v1.joblib")

s3 = boto3.client("s3", region_name=REGION)

def latest_parquet_keys(n=1):
    resp = s3.list_objects_v2(Bucket=FEATURE_BUCKET, Prefix=FEATURE_PREFIX, MaxKeys=1000)
    keys = [o["Key"] for o in resp.get("Contents",[]) if o["Key"].endswith(".parquet")]
    return sorted(keys)[-n:] if keys else []

def load_model():
    obj = s3.get_object(Bucket=FEATURE_BUCKET, Key=MODEL_KEY)
    by = io.BytesIO(obj["Body"].read())
    return joblib.load(by)

def load_features(keys):
    import s3fs, pyarrow.parquet as pq, pyarrow as pa
    fs = s3fs.S3FileSystem()
    tabs=[]
    for k in keys:
        with fs.open(f"s3://{FEATURE_BUCKET}/{k}","rb") as f:
            tabs.append(pq.read_table(f))
    if not tabs: return pd.DataFrame()
    return pa.concat_tables(tabs).to_pandas().sort_values("ts_min_utc")

def handler(event, context):
    model = load_model()
    keys = latest_parquet_keys(1)
    df = load_features(keys)
    if df.empty:
        return {"ok": True, "msg":"no features"}
    feats = ["rps","cf_uniq_ip","cnt_4xx","cnt_5xx","uri_entropy","waf_uniq_ip","waf_geo_entropy"]
    x = df[feats].fillna(0).tail(1)
    p = float(model.predict_proba(x)[0,1]) if hasattr(model,"predict_proba") else float(model.decision_function(x))
    decision = "NoAction" if p < 0.6 else ("Challenge" if p < 0.85 else "Mitigate")
    return {"ok": True, "p_attack": p, "decision": decision}
