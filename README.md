# PRAGMATIC DDOS DETECTION ON AWS CLOUD WITH LIGHTWEIGHT ML

This repository accompanies a study on how training data composition and model capacity jointly affect the reliability and generalization of a pragmatic, edge-based DDoS detector on AWS.
## Study Summary

A minimal pipeline leverages CloudFront-only, one-minute aggregates (request rate, unique client IPs, and 4xx/5xx counts). Two supervised detectors—RandomForest (RF) and Logistic Regression (LR)—are trained on a comprehensive dataset spanning benign background, a known high-RPS attack, and a benign flash crowd. Evaluation is performed on a continuous A→B→C→D timeline, where D is an unseen attack.
**Key findings:**
- With representative training coverage, both detectors operate stably on benign traffic without false positives.
- Only the nonlinear RF retains sensitivity under distribution shift; the linear LR collapses on the blind scenario.
- The approach demonstrates a practical blueprint for availability-centric detection at the CDN perimeter: edge-visible signals, a validation-frozen operating point, and a lightweight nonlinear ensemble suitable for conservative WAF integration under explicit false-positive budgets.
## Pipeline Overview

- **Data sources:** CloudFront logs, one-minute aggregates.
- **Features:** Request rate (RPS), unique client IPs, 4xx/5xx error counts.
- **Models:** RandomForest and Logistic Regression.
- **Validation:** Disjoint slice used to freeze a single operating threshold ex-ante.
- **Evaluation:** Timeline includes benign, attack, and flash crowd scenarios.
## Getting Started

### Environment Setup

```sh
python3 --version
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
## Environment Configuration

Configuration values (such as AWS credentials, S3 bucket names, and model parameters) can be set in a `.env` file. See `.env.example` for a template of required variables. Copy it to `.env` and adjust values as needed:
```sh
cp .env.example .env
# Edit .env with your settings
```

## Infrastructure Provisioning (CloudFormation)

The AWS infrastructure is provisioned using a CloudFormation template located at `infra/cloudformation/cloudformationStack.yaml`.

This stack deploys:
- VPC, public subnets, and security groups
- ECS Fargate cluster and service (Juice Shop container)
- Application Load Balancer (ALB)
- CloudFront distribution with attached WAF (WebACL)
- S3 buckets for CloudFront and WAF logs
- Kinesis Firehose for WAF log delivery

**Outputs:**
- `CloudFrontDomain`: Public domain of the CloudFront distribution
- `ALBDNSName`: Public DNS of the ALB (origin)
- `LogsBucketName`: S3 bucket for CloudFront access logs
- `ECSClusterName`: ECS cluster name
- `ECSServiceName`: ECS service name

Deploy the stack in `us-east-1` (required for CloudFront WAF). Adjust parameters as needed for your experiment (see template for defaults).

## Traffic Generation with k6

Synthetic traffic for experiments can be generated using [k6](https://k6.io/) scripts in `load/k6/`. These simulate benign and attack scenarios for CloudFront endpoints.

Example: benign breeze scenario
```sh
export BASE_URL="https://your-cloudfront-domain"
k6 run load/k6/benign-breeze.js
```

Example: attack burst scenario
```sh
export BASE_URL="https://your-cloudfront-domain"
k6 run load/k6/attack-burst.js
```

You can customize parameters via environment variables (see script headers and `load/k6/howtorun`). For orchestrated multi-phase experiments, use:
```sh
./load/k6/run_sequence_v1.sh
```

### Running Batch Feature Extraction

Example for batch A:
```sh
python3 ml/run_batch.py \
   --cf-bucket "$S3_LOGS_BUCKET" \
   --start 2025-09-26T18:10:00Z --end 2025-09-26T18:50:00Z \
   --labels data/labels/attack_intervals.csv \
   --out batches/batch_A.parquet \
   --upload-s3 s3://<your-bucket>/feature-store/joined \
   --upload-name batch_A.parquet
```

### Quick Data Inspection

```sh
python3 - <<'PY'
import pandas as pd
df = pd.read_parquet("batches/batch_A.parquet")
print("rows:", len(df))
print("rps_min/max:", df["rps"].min(), df["rps"].max())
print("labels:", sorted(df["label"].unique().tolist()))
PY
```

### Model Training and Evaluation

Train Logistic Regression:
```sh
python3 ml/run_logistic_baseline.py \
   --train data/dataset_cf/split_train.parquet \
   --val   data/dataset_cf/split_val.parquet \
   --test  data/dataset_cf/split_test.parquet \
   --attack-core-start 2025-09-20T17:14:00Z \
   --attack-core-end   2025-09-20T17:22:00Z \
   --out-prefix ./lr_results
```

Train RandomForest:
```sh
python3 ml/train_baseline_rf.py \
   --train data/dataset_cf_expanded/split_train.parquet \
   --val   data/dataset_cf_expanded/split_val.parquet \
   --test  data/dataset_cf_expanded/split_test.parquet
```

### Visualization

Generate comparison figures:
```sh
python3 generate_final_graph.py
```



## Disclaimer

This repository is for academic and experimental purposes only.  
Do not use generated attack traffic against systems you do not own or operate.  
Restrict tests to controlled environments within your own AWS account.
