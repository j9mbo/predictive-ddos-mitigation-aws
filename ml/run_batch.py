#!/usr/bin/env python3
# ml/run_batch.py
# Orchestrate CF-only pipeline: CF manifest -> CF aggregation -> join&label -> (optional) upload to S3.

import argparse, subprocess

def sh(cmd: str):
    print(f"[RUN] {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def iso2tag(s: str) -> str:
    # 2025-09-15T13:50:00Z -> 20250915T135000Z
    return (
        s.replace("-", "")
         .replace(":", "")
         .replace("+00:00", "Z")
         .replace("Z", "Z")
    )

def main():
    ap = argparse.ArgumentParser(description="CF-only batch runner")
    ap.add_argument("--cf-bucket", required=True, help="S3 bucket with CloudFront logs")
    ap.add_argument("--cf-prefix", default="cloudfront/", help="Prefix inside the bucket")
    ap.add_argument("--start", required=True, help="Start ISO UTC, e.g. 2025-09-15T13:50:00Z")
    ap.add_argument("--end",   required=True, help="End ISO UTC,   e.g. 2025-09-15T14:04:00Z")
    ap.add_argument("--labels", required=True, help="CSV with start_utc,end_utc,label")
    ap.add_argument("--out", required=True, help="Output parquet path (local)")
    ap.add_argument("--upload-s3", default=None, help="s3://bucket/prefix OR s3://bucket/prefix/name.parquet (optional)")
    ap.add_argument("--upload-name", default=None, help="File name to use in S3 (e.g., batch_A.parquet). Optional.")
    args = ap.parse_args()

    t_start = iso2tag(args.start)
    t_end   = iso2tag(args.end)
    cf_manifest = f"/tmp/cf_manifest_{t_start}_{t_end}.json"
    cf_agg      = f"/tmp/cf_agg_{t_start}_{t_end}.parquet"

    # 1) CF manifest
    sh(
        " ".join([
            "python3 ml/cf_utils.py",
            f"--cf-bucket {args.cf_bucket}",
            f"--cf-prefix {args.cf_prefix}",
            f"--start-iso {args.start}",
            f"--end-iso {args.end}",
            f"--manifest-out {cf_manifest}",
        ])
    )

    # 2) CF aggregation
    sh(
        " ".join([
            "python3 ml/cf_agg.py",
            f"--manifest {cf_manifest}",
            f"--cf-bucket {args.cf_bucket}",
            f"--start-iso {args.start}",
            f"--end-iso {args.end}",
            f"--out-parquet {cf_agg}",
        ])
    )

    # 3) Join & label (CF-only)
    sh(
        " ".join([
            "python3 ml/join_and_write.py",
            f"--cf-parquet {cf_agg}",
            f"--labels-csv {args.labels}",
            f"--start-iso {args.start}",
            f"--end-iso {args.end}",
            f"--out-parquet {args.out}",
        ])
    )

    # 4) Optional upload to S3
    if args.upload_s3:
        dst = args.upload_s3.rstrip("/")
        # If user passed a full S3 key (endswith .parquet), use it as-is
        if dst.startswith("s3://") and dst.endswith(".parquet"):
            s3_dst = dst
        else:
            # It's a prefix; pick name:
            if args.upload_name:
                name = args.upload_name
                if not name.endswith(".parquet"):
                    name += ".parquet"
            else:
                # default timestamp-based name
                name = f"batch_{t_start}_{t_end}.parquet"
            s3_dst = f"{dst}/{name}"

        sh(f"aws s3 cp {args.out} {s3_dst}")
        print(f"[INFO] Uploaded to {s3_dst}")

    print("[INFO] Batch completed successfully.")

if __name__ == "__main__":
    main()
