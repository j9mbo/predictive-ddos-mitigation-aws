#!/usr/bin/env python3
# ml/cf_utils.py
"""
Part 1/.. : CloudFront keys collector.
Usage (example):
  python3 ml/cf_utils.py --cf-bucket 913524935822-juice-ecs-logs-us-east-1 \
     --cf-prefix cloudfront/ --start-iso 2025-09-15T13:50:00Z --end-iso 2025-09-15T14:04:00Z \
     --manifest-out /tmp/cf_manifest_A.json
"""
from datetime import datetime, timezone, timedelta
import argparse
import boto3
import json
import sys

def parse_iso_utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)

def hour_substrings(start_dt: datetime, end_dt: datetime):
    cur = start_dt.replace(minute=0, second=0, microsecond=0)
    substrs = set()
    while cur <= end_dt:
        substrs.add(cur.strftime(".%Y-%m-%d-%H."))
        cur += timedelta(hours=1)
    return substrs

def list_cf_keys_in_window(bucket: str, prefix: str, start_iso: str, end_iso: str):
    s3 = boto3.client("s3")
    start_dt = parse_iso_utc(start_iso)
    end_dt = parse_iso_utc(end_iso)
    substrs = hour_substrings(start_dt, end_dt)
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if any(sub in key for sub in substrs):
                # optional extra guard by LastModified to avoid unrelated hours
                lm = obj["LastModified"].astimezone(timezone.utc)
                if lm >= (start_dt - timedelta(hours=1)) and lm <= (end_dt + timedelta(hours=1)):
                    keys.append({"Key": key, "LastModified": lm.isoformat(), "Size": obj.get("Size", 0)})
    # sort by LastModified asc
    keys.sort(key=lambda x: x["LastModified"])
    return keys

def main():
    ap = argparse.ArgumentParser(description="Collect CloudFront S3 keys for a time window and write manifest")
    ap.add_argument("--cf-bucket", required=True)
    ap.add_argument("--cf-prefix", default="cloudfront/")
    ap.add_argument("--start-iso", required=True)
    ap.add_argument("--end-iso", required=True)
    ap.add_argument("--manifest-out", default=None, help="Optional path to write manifest JSON")
    args = ap.parse_args()

    keys = list_cf_keys_in_window(args.cf_bucket, args.cf_prefix, args.start_iso, args.end_iso)
    print(f"[INFO] Found {len(keys)} objects in window {args.start_iso} -> {args.end_iso}", file=sys.stderr)
    for k in keys:
        print(k["LastModified"], k["Key"], k["Size"])
    if args.manifest_out:
        manifest = {
            "start_iso": args.start_iso,
            "end_iso": args.end_iso,
            "cf_bucket": args.cf_bucket,
            "cf_prefix": args.cf_prefix,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "objects": keys
        }
        with open(args.manifest_out, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Manifest written to {args.manifest_out}", file=sys.stderr)

if __name__ == "__main__":
    main()
