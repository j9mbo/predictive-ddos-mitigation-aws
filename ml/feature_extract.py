# ml/feature_extract.py
import os, io, gzip, csv, json
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import boto3
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("AWS_REGION", "us-east-1")
CF_BUCKET = os.getenv("S3_LOGS_BUCKET")
WAF_BUCKET = os.getenv("S3_WAF_BUCKET")
FEATURE_BUCKET = CF_BUCKET
FEATURE_PREFIX = os.getenv("S3_FEATURE_PREFIX", "feature-store/joined/")
LOOKBACK_HOURS = int(os.getenv("LOOKBACK_HOURS", "6"))

s3 = boto3.client("s3", region_name=REGION)

def list_keys(bucket, prefix):
    keys, token = [], None
    while True:
        kw = dict(Bucket=bucket, Prefix=prefix, MaxKeys=1000)
        if token: kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for o in resp.get("Contents", []): keys.append(o["Key"])
        if not resp.get("IsTruncated"): break
        token = resp["NextContinuationToken"]
    return keys

def read_waf_json_gz(bucket, keys, start_min):
    rows=[]
    for k in keys:
        if not k.endswith(".gz"): continue
        obj = s3.get_object(Bucket=bucket, Key=k)
        by = io.BytesIO(obj["Body"].read())
        with gzip.GzipFile(fileobj=by) as gz:
            for line in gz:
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                ts = pd.to_datetime(j.get("timestamp"), unit="ms", utc=True)
                if ts.floor("T") < start_min: 
                    continue
                http = j.get("httpRequest", {})
                rows.append({
                    "ts_min": ts.floor("T"),
                    "client_ip": http.get("clientIp"),
                    "country": (http.get("country") or "ZZ"),
                    "uri": (http.get("uri") or "/"),
                    "action": j.get("action","ALLOW")
                })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ts_min","client_ip","country","uri","action"])

def read_cf_tsv(bucket, keys, start_min):
    cols = ["date","time","x_edge_location","sc_bytes","c_ip","cs_method","cs_host",
            "cs_uri_stem","sc_status","cs_referer","cs_user_agent","cs_uri_query",
            "cs_cookie","x_edge_result_type","x_edge_request_id","x_host_header",
            "cs_protocol","cs_bytes","time_taken","x_forwarded_for","ssl_protocol",
            "ssl_cipher","x_edge_response_result_type","cs_protocol_version"]
    recs=[]
    for k in keys:
        if not (k.endswith(".gz") or k.endswith(".log")): continue
        obj = s3.get_object(Bucket=bucket, Key=k)
        body = obj["Body"].read()
        if k.endswith(".gz"):
            body = gzip.decompress(body)
        rdr = csv.reader(io.StringIO(body.decode("utf-8", errors="ignore")), delimiter="\t")
        for row in rdr:
            if not row or row[0].startswith("#") or len(row)<len(cols): 
                continue
            d = dict(zip(cols,row))
            try:
                dt = pd.to_datetime(d["date"]+"T"+d["time"]+"Z", utc=True)
            except Exception:
                continue
            if dt.floor("T") < start_min: 
                continue
            recs.append({
                "ts_min": dt.floor("T"),
                "c_ip": d["c_ip"],
                "cs_uri_stem": d["cs_uri_stem"] or "/",
                "sc_status": int(d["sc_status"]) if d["sc_status"].isdigit() else 0
            })
    return pd.DataFrame(recs) if recs else pd.DataFrame(columns=["ts_min","c_ip","cs_uri_stem","sc_status"])

def entropy(counts: pd.Series) -> float:
    n = counts.sum()
    if n <= 0: return 0.0
    p = counts / n
    p = p[p>0]
    return float(-(p * np.log(p)).sum())

def build_features(waf_df, cf_df):
    out = pd.DataFrame()
    if not cf_df.empty:
        g = cf_df.groupby("ts_min")
        cf_feat = pd.DataFrame({
            "rps": g.size(),
            "cf_uniq_ip": g["c_ip"].nunique(),
            "cnt_4xx": g.apply(lambda x: ((x["sc_status"]>=400)&(x["sc_status"]<500)).sum()),
            "cnt_5xx": g.apply(lambda x: ((x["sc_status"]>=500)&(x["sc_status"]<600)).sum()),
            "uri_entropy": g["cs_uri_stem"].apply(lambda s: entropy(s.value_counts()))
        })
        out = cf_feat
    if not waf_df.empty:
        h = waf_df.groupby("ts_min")
        waf_feat = pd.DataFrame({
            "waf_req_total": h.size(),
            "waf_allow": h.apply(lambda x: (x["action"]=="ALLOW").sum()),
            "waf_block": h.apply(lambda x: (x["action"]=="BLOCK").sum()),
            "waf_challenge": h.apply(lambda x: (x["action"]=="CHALLENGE").sum()),
            "waf_captcha": h.apply(lambda x: (x["action"]=="CAPTCHA").sum()),
            "waf_uniq_ip": h["client_ip"].nunique(),
            "waf_geo_entropy": h["country"].apply(lambda s: entropy(s.value_counts()))
        })
        out = out.join(waf_feat, how="outer") if not out.empty else waf_feat
    if out.empty: 
        return out
    out = out.sort_index()
    out["dt"] = out.index.tz_convert("UTC").strftime("%Y-%m-%d-%H-%M")
    return out.reset_index().rename(columns={"ts_min":"ts_min_utc"}).fillna(0)

def main():
    now = datetime.now(timezone.utc)
    start = (now - timedelta(hours=LOOKBACK_HOURS)).replace(second=0, microsecond=0)
    # waf партиції: зручно почати з year=
    waf_keys = list_keys(WAF_BUCKET, f"year={start:%Y}/")
    cf_keys  = list_keys(CF_BUCKET, "cloudfront/")
    waf_df = read_waf_json_gz(WAF_BUCKET, waf_keys, start_min=start)
    cf_df  = read_cf_tsv(CF_BUCKET, cf_keys, start_min=start)
    feat = build_features(waf_df, cf_df)
    if feat.empty:
        print("No data in lookback window"); return
    import pyarrow as pa, pyarrow.parquet as pq
    table = pa.Table.from_pandas(feat)
    buf = io.BytesIO(); pq.write_table(table, buf, compression="snappy")
    key = f"{FEATURE_PREFIX}batch_{now:%Y%m%dT%H%M%SZ}.parquet"
    s3.put_object(Bucket=FEATURE_BUCKET, Key=key, Body=buf.getvalue())
    print(f"Wrote features → s3://{FEATURE_BUCKET}/{key}")

if __name__ == "__main__":
    import pandas as pd
    main()
