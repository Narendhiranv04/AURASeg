#!/bin/bash
rm -rf runs_carld_rgb/r18_mul_full_sobel_gate_seed42
rm -rf runs_carld_rgb/benchmark_deeplabv3plus_seed42
rm -rf runs_carld_rgb/benchmark_segformer_seed42
rm -rf runs_fbrnet_carld_rgb/fbrnet_carld_seed42

python benchmark_models/train_auraseg_r18_carld_rgb.py &> auraseg_carl.log
python benchmark_models/train_benchmarks_carld_rgb.py --model deeplabv3plus &> deeplabv3plus_carl.log
python benchmark_models/train_benchmarks_carld_rgb.py --model segformer &> segformer_carl.log
python benchmark_models/train_fbrnet_wacv.py --dataset carl-d --output-root runs_fbrnet_carld_rgb &> fbrnet_carl.log
