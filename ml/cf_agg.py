#!/usr/bin/env python3
# ml/cf_agg.py
# Part 2: CloudFront minute aggregation from a manifest of S3 keys.
# Usage:
#   python3 ml/cf_agg.py --manifest /tmp/cf_manifest_A.json \
#     --start-iso 2025-09-15T13:50:00Z --end-iso 2025-09-15T14:04:00Z \
#     --cf-bucket 913524935822-juice-ecs-logs-us-east-1 \
#     --out-parquet /tmp/cf_agg_A.parquet

import argparse, json, io, gzip, csv, sys
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter

import boto3
import pandas as pd

def parse_iso_utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)

def minute_floor_utc(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0, tzinfo=timezone.utc)

def iter_cf_records(s3, bucket: str, key: str):
    obj = s3.get_object(Bucket=bucket, Key=key)
    with gzip.GzipFile(fileobj=io.BytesIO(obj["Body"].read())) as gz:
        header = None
        for raw in gz:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if not line or line.startswith("#"):
                if line.startswith("#Fields:"):
                    header = line.split(":", 1)[1].strip().split()
                continue
            fields = line.split("\t")
            if header and len(fields) >= len(header):
                yield dict(zip(header, fields))
            else:
                # Fallback: minimal mapping by expected positions if header missing
                # date(0), time(1), c-ip(4), sc-status(8), cs-uri-stem(7)
                rec = {}
                try:
                    rec["date"] = fields[0]; rec["time"] = fields[1]
                    rec["c-ip"] = fields[4] if len(fields) > 4 else None
                    rec["sc-status"] = fields[8] if len(fields) > 8 else None
                    rec["cs-uri-stem"] = fields[7] if len(fields) > 7 else None
                except Exception:
                    continue
                yield rec

def aggregate_from_manifest(manifest_path: str, bucket: str, start_iso: str, end_iso: str):
    start_dt = parse_iso_utc(start_iso); end_dt = parse_iso_utc(end_iso)
    with open(manifest_path, "r", encoding="utf-8") as f:
        man = json.load(f)
    keys = [o["Key"] for o in man.get("objects", [])]
    s3 = boto3.client("s3")

    metrics = defaultdict(lambda: {"rps": 0, "cnt_4xx": 0, "cnt_5xx": 0, "cf_rows": 0})
    uniq_ip = defaultdict(set)
    uri_freq = defaultdict(Counter)

    for key in keys:
        for rec in iter_cf_records(s3, bucket, key):
            try:
                dt = datetime.strptime(rec["date"] + " " + rec["time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if dt < start_dt or dt > end_dt:
                continue
            m = minute_floor_utc(dt)
            metrics[m]["rps"] += 1
            metrics[m]["cf_rows"] += 1

            st = rec.get("sc-status")
            if st and st.isdigit():
                code = int(st)
                if 400 <= code < 500: metrics[m]["cnt_4xx"] += 1
                elif 500 <= code < 600: metrics[m]["cnt_5xx"] += 1

            cip = rec.get("c-ip"); 
            if cip: uniq_ip[m].add(cip)
            uri = rec.get("cs-uri-stem")
            if uri: uri_freq[m][uri] += 1

    # build full minute index
    rows = []
    cur = minute_floor_utc(start_dt)
    endf = minute_floor_utc(end_dt)
    while cur <= endf:
        m = metrics[cur]
        rows.append({
            "ts_min_utc": cur,
            "rps": m["rps"],
            "cf_uniq_ip": len(uniq_ip[cur]) if cur in uniq_ip else 0,
            "cnt_4xx": m["cnt_4xx"],
            "cnt_5xx": m["cnt_5xx"],
            "cf_rows": m["cf_rows"],
        })
        cur += timedelta(minutes=1)

    df = pd.DataFrame(rows).sort_values("ts_min_utc").reset_index(drop=True)
    return df, uri_freq

def main():
    ap = argparse.ArgumentParser(description="CloudFront minute aggregation from manifest")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cf-bucket", required=True)
    ap.add_argument("--start-iso", required=True)
    ap.add_argument("--end-iso", required=True)
    ap.add_argument("--out-parquet", default=None)
    args = ap.parse_args()

    df, uri_freq = aggregate_from_manifest(args.manifest, args.cf_bucket, args.start_iso, args.end_iso)
    print(f"[INFO] CF minutes: {len(df)} {df['ts_min_utc'].min()} -> {df['ts_min_utc'].max()}", file=sys.stderr)
    print(df.head(10).to_string(index=False))
    if args.out_parquet:
        df.to_parquet(args.out_parquet, index=False)
        print(f"[INFO] Written CF agg parquet: {args.out_parquet}", file=sys.stderr)

if __name__ == "__main__":
    main()
