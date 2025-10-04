# Makefile — on-demand pipeline (no schedulers)
# Uses .env for AWS/S3 parameters. Comments in English per your preference.

SHELL := /bin/bash
ENV_FILE ?= .env
include $(ENV_FILE)
export $(shell sed 's/=.*//' $(ENV_FILE))

# ---- Defaults (can be overridden on command line) ----
PY ?= python
FEATURES_LOOKBACK_HOURS ?= $(LOOKBACK_HOURS)
FEATURES_START_FROM_ISO ?= $(START_FROM_ISO)
SHADOW_WINDOW_MINUTES ?= 30

.PHONY: help
help:
	@echo "Targets:"
	@echo "  make features         # Build features from S3 logs → Parquet in feature-store/joined/"
	@echo "  make train-baseline   # Train ETS baseline → s3://.../models/"
	@echo "  make shadow           # Shadow scoring → s3://.../decisions/"
	@echo "  make eval             # Evaluate metrics vs labels → prints summary"
	@echo "  make snapshot-dataset # Copy features/decisions → snapshots/<UTC>/"
	@echo "  make show-env         # Print key env variables"
	@echo "  make clean-local      # Remove __pycache__/ etc"

.PHONY: show-env
show-env:
	@echo "AWS_PROFILE=$(AWS_PROFILE)"
	@echo "AWS_REGION=$(AWS_REGION)"
	@echo "S3_LOGS_BUCKET=$(S3_LOGS_BUCKET)"
	@echo "S3_WAF_BUCKET=$(S3_WAF_BUCKET)"
	@echo "S3_FEATURE_PREFIX=$(S3_FEATURE_PREFIX)"
	@echo "S3_MODELS_PREFIX=$(S3_MODELS_PREFIX)"
	@echo "WAF_WEBACL_ARN=$(WAF_WEBACL_ARN)"
	@echo "LOOKBACK_HOURS=$(FEATURES_LOOKBACK_HOURS)"
	@echo "START_FROM_ISO=$(FEATURES_START_FROM_ISO)"
	@echo "SHADOW_WINDOW_MINUTES=$(SHADOW_WINDOW_MINUTES)"

# ---- Features (batch) ----
.PHONY: features
features:
	@echo "[features] START_FROM_ISO=$(FEATURES_START_FROM_ISO)"
	@START_FROM_ISO=$(FEATURES_START_FROM_ISO) \
	 LOOKBACK_HOURS=$(FEATURES_LOOKBACK_HOURS) \
	 $(PY) ml/feature_extract.py

# ---- Train baseline (ETS) ----
.PHONY: train-baseline
train-baseline:
	@echo "[train-baseline] Training ETS baseline"
	@$(PY) ml/train_baseline.py

# ---- Shadow scoring ----
.PHONY: shadow
shadow:
	@echo "[shadow] SHADOW_WINDOW_MINUTES=$(SHADOW_WINDOW_MINUTES)"
	@SHADOW_WINDOW_MINUTES=$(SHADOW_WINDOW_MINUTES) \
	 $(PY) ml/score_shadow.py

# ---- Quick evaluation wrapper (requires args via env) ----
# Provide EVAL_START / EVAL_END in ISO-UTC and labels path if different.
EVAL_DECISIONS_S3 ?= s3://$(S3_LOGS_BUCKET)/decisions/
EVAL_LABELS ?= data/labels/attack_intervals.csv
EVAL_START ?=
EVAL_END ?=

.PHONY: eval
eval:
	@test -n "$(EVAL_START)" || (echo "EVAL_START is required (ISO UTC)"; exit 1)
	@test -n "$(EVAL_END)"   || (echo "EVAL_END is required (ISO UTC)"; exit 1)
	@$(PY) scripts/eval_metrics.py \
		--decisions-s3 "$(EVAL_DECISIONS_S3)" \
		--labels-csv "$(EVAL_LABELS)" \
		--start "$(EVAL_START)" \
		--end   "$(EVAL_END)"

# ---- Snapshot dataset (features + decisions → snapshots/<UTC>/) ----
.PHONY: snapshot-dataset
snapshot-dataset:
	@stamp=$$(date -u +%Y%m%dT%H%M%SZ); \
	echo "[snapshot-dataset] s3://$(S3_LOGS_BUCKET)/snapshots/$$stamp/"; \
	aws s3 cp s3://$(S3_LOGS_BUCKET)/$(S3_FEATURE_PREFIX) s3://$(S3_LOGS_BUCKET)/snapshots/$$stamp/$(S3_FEATURE_PREFIX) --recursive --profile $(AWS_PROFILE) --region $(AWS_REGION); \
	aws s3 cp s3://$(S3_LOGS_BUCKET)/decisions/ s3://$(S3_LOGS_BUCKET)/snapshots/$$stamp/decisions/ --recursive --profile $(AWS_PROFILE) --region $(AWS_REGION); \
	echo "[snapshot-dataset] Done."

.PHONY: clean-local
clean-local:
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} + || true
	@find . -name ".pytest_cache" -type d -prune -exec rm -rf {} + || true
