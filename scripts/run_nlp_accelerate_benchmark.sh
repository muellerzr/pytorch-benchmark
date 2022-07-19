#!/bin/bash
# Single GPU
echo "Running Single GPU scripts..."
CUDA_VISIBLE_DEVICES="0" python nlp_script_accelerate.py configs/nlp_script/single_gpu/baseline.yml
CUDA_VISIBLE_DEVICES="0" python nlp_script_accelerate.py configs/nlp_script/single_gpu/mixed_precision.yml

# Multi GPU
echo "Running Multi GPU scripts..."
CUDA_VISIBLE_DEVICES="0,1" accelerate launch nlp_script_accelerate.py configs/nlp_script/multi_gpu/baseline.yml
CUDA_VISIBLE_DEVICES="0,1" accelerate launch nlp_script_accelerate.py configs/nlp_script/multi_gpu/mixed_precision.yml
CUDA_VISIBLE_DEVICES="0,1" accelerate launch nlp_script_accelerate.py configs/nlp_script/multi_gpu/adjusted_bs.yml
CUDA_VISIBLE_DEVICES="0,1" accelerate launch nlp_script_accelerate.py configs/nlp_script/multi_gpu/adjusted_bs_mixed_precision.yml
CUDA_VISIBLE_DEVICES="0,1" accelerate launch nlp_script_accelerate.py configs/nlp_script/multi_gpu/adjusted_bs_double_lr.yml
CUDA_VISIBLE_DEVICES="0,1" accelerate launch nlp_script_accelerate.py configs/nlp_script/multi_gpu/adjusted_bs_double_lr_mixed_precision.yml