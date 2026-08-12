#!/bin/bash

# Non-destructive smoke test runner
echo "Running smoke tests..."

python benchmark_models/train_auraseg_r18_carld_rgb.py --smoke-test
python benchmark_models/train_benchmarks_carld_rgb.py --model deeplabv3plus --smoke-test
python benchmark_models/train_benchmarks_carld_rgb.py --model segformer --smoke-test
python benchmark_models/train_fbrnet_wacv.py --dataset carl-d --output-root runs_fbrnet_carld_rgb --smoke-test

echo "Smoke tests completed."
