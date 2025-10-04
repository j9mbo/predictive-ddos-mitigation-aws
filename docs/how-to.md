python3 --version

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

% python3 ml/run_batch.py \ --start 2025-09-15T15:07:00Z \ --end 2025-09-15T15:19:00Z \ --labels data/labels/attack_intervals.csv \ --out /tmp/batch_B.parquet \ --upload-s3 s3://913524935822-juice-ecs-logs-us-east-1/feature-store/joined

# A 16:19 - 16:59
python3 ml/run_batch.py \
  --cf-bucket "$S3_LOGS_BUCKET" \
  --start 2025-09-26T18:10:00Z --end 2025-09-26T18:50:00Z \
  --labels data/labels/attack_intervals.csv \
  --out batches/batch_A.parquet \
  --upload-s3 s3://913524935822-juice-ecs-logs-us-east-1/feature-store/joined \
  --upload-name batch_A.parquet

python3 - <<'PY'
import pandas as pd
df = pd.read_parquet("batches/batch_A.parquet")
print("rows:", len(df))
print("rps_min/max:", df["rps"].min(), df["rps"].max())
tot_req = df["cf_rows"].sum() if "cf_rows" in df else df["rps"].sum()
tot_5xx = df.get("cnt_5xx", pd.Series([0]*len(df))).sum()
print("5xx_share:", round(tot_5xx / max(tot_req,1), 4))
print("labels:", sorted(df["label"].unique().tolist()))
PY

python3 - <<'PY'
import pandas as pd
df = pd.read_parquet("batches/batch_A.parquet")
print("only_zero_labels:", df["label"].eq(0).all())
PY

# B (комбо з вікном 17:14–17:34)
python3 ml/run_batch.py \
  --cf-bucket "$S3_LOGS_BUCKET" \
  --start 2025-09-20T16:59:00Z --end 2025-09-20T17:39:00Z \
  --labels data/labels/attack_intervals.csv \
  --out batches/batch_B.parquet \
  --upload-s3 s3://913524935822-juice-ecs-logs-us-east-1/feature-store/joined \
  --upload-name batch_B.parquet

python3 - <<'PY'
import pandas as pd
df = pd.read_parquet("batches/batch_B.parquet")
print("rows:", len(df))
print("rps_min/max:", df["rps"].min(), df["rps"].max())
tot_req = df["cf_rows"].sum() if "cf_rows" in df else df["rps"].sum()
tot_5xx = df.get("cnt_5xx", pd.Series([0]*len(df))).sum()
print("5xx_share:", round(tot_5xx / max(tot_req,1), 4))
print("labels:", sorted(df["label"].unique().tolist()))
PY

python3 - <<'PY'
import pandas as pd
df = pd.read_parquet("batches/batch_B.parquet")
print("only_zero_labels:", df["label"].eq(0).all())
PY

python3 - <<'PY'
import pandas as pd
df = pd.read_parquet("batches/batch_B.parquet")
pos = df[df.label==1]
neg = df[df.label==0]
print("pos_minutes:", len(pos), "neg_minutes:", len(neg))
print("pos rps min/median/max:", pos.rps.min(), pos.rps.median(), pos.rps.max())
print("pos cf_uniq_ip max:", pos.get("cf_uniq_ip").max())
print("first_pos_min:", pos.ts_min_utc.min(), "last_pos_min:", pos.ts_min_utc.max())
PY


# C
python3 ml/run_batch.py \
  --cf-bucket "$S3_LOGS_BUCKET" \
  --start 2025-09-26T12:35:00Z --end 2025-09-26T13:15:00Z \
  --labels data/labels/attack_intervals.csv \
  --out batches/batch_C.parquet \
  --upload-s3 s3://913524935822-juice-ecs-logs-us-east-1/feature-store/joined \
  --upload-name batch_C.parquet

python3 - <<'PY'
import pandas as pd
df = pd.read_parquet("batches/batch_C.parquet")
print("rows:", len(df))
print("rps_min/max:", df["rps"].min(), df["rps"].max())
tot_req = df["cf_rows"].sum() if "cf_rows" in df else df["rps"].sum()
tot_5xx = df.get("cnt_5xx", pd.Series([0]*len(df))).sum()
print("5xx_share:", round(tot_5xx / max(tot_req,1), 4))
print("labels:", sorted(df["label"].unique().tolist()))
PY

python3 - <<'PY'
import pandas as pd
df = pd.read_parquet("batches/batch_C.parquet")
print("only_zero_labels:", df["label"].eq(0).all())
PY



# D (комбо з вікном 18:34–18:54)
python3 ml/run_batch.py \
  --cf-bucket "$S3_LOGS_BUCKET" \
  --start 2025-09-28T14:15:00Z --end 2025-09-28T14:53:00Z \
  --labels data/labels/attack_intervals.csv \
  --out batches/batch_D.parquet \
  --upload-s3 s3://913524935822-juice-ecs-logs-us-east-1/feature-store/joined \
  --upload-name batch_D.parquet

python3 - <<'PY'
import pandas as pd
df = pd.read_parquet("batches/batch_D.parquet")
print("rows:", len(df))
print("rps_min/max:", df["rps"].min(), df["rps"].max())
tot_req = df["cf_rows"].sum() if "cf_rows" in df else df["rps"].sum()
tot_5xx = df.get("cnt_5xx", pd.Series([0]*len(df))).sum()
print("5xx_share:", round(tot_5xx / max(tot_req,1), 4))
print("labels:", sorted(df["label"].unique().tolist()))
PY

python3 - <<'PY'
import pandas as pd
df = pd.read_parquet("batches/batch_D.parquet")
print("only_zero_labels:", df["label"].eq(0).all())
PY



logistic regression:
python ml/run_logistic_baseline.py \
  --train data/dataset_cf/split_train.parquet \
  --val   data/dataset_cf/split_val.parquet \
  --test  data/dataset_cf/split_test.parquet \
  --attack-core-start 2025-09-20T17:14:00Z \
  --attack-core-end   2025-09-20T17:22:00Z \
  --out-prefix ./lr_results
  

python3 ml/make_fig_tables.py \
--train data/dataset_cf/split_train.parquet \
--batch-b batches/batch_B.parquet \
--batch-c batches/batch_C.parquet \
--out-dir data/fig \
--rf-thr 0.37 \
--lr-thr 0.314558

python3 ml/make_fig_tables.py \
--train data/dataset_cf/split_train.parquet \
--batch-b batches/batch_B.parquet \
--batch-c batches/batch_C.parquet \
--out-dir data/fig \
--rf-thr 0.37 \
--lr-thr 0.314558

python3 ml/make_fig_tables.py \
  --train data/dataset_cf/split_train_expanded.parquet \
  --batch-b batches/batch_B.parquet \
  --batch-c batches/batch_C.parquet \
  --out-dir data/fig \
  --rf-thr 0.37 \
  --lr-thr 0.314558


python3 ml/plots.py \
--b data/fig/table_B.csv --c data/fig/table_C.csv \
--out figure_model_comparison_v2.png \
--tau-rf 0.37 --tau-lr 0.75 --rps-thr 500 \
--attack-core-start 2025-09-20T17:14:00Z \
--attack-core-end   2025-09-20T17:22:00Z


python3 ml/build_dataset_enhanced.py \
--batch-a batches/batch_A.parquet \
--batch-b batches/batch_B.parquet \
--batch-c batches/batch_C.parquet \
--out-prefix data/dataset_cf_expanded \
--add-c-to-train 15



python3 ml/train_baseline_rf.py \
  --train data/dataset_cf_expanded/split_train.parquet \
  --val   data/dataset_cf_expanded/split_val.parquet \
  --test  data/dataset_cf_expanded/split_test.parquet