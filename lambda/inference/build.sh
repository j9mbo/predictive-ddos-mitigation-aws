#!/usr/bin/env bash
set -euo pipefail
rm -f function.zip
pip install -r requirements.txt -t package
cp lambda_function.py package/
cd package && zip -r ../function.zip . && cd -
echo "Built function.zip"