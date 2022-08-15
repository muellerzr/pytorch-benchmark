#!/bin/bash
# Single GPU
# echo "Running Single GPU scripts..."
# CUDA_VISIBLE_DEVICES="0" python nlp_script.py configs/nlp_script/single_gpu/baseline.yml
# CUDA_VISIBLE_DEVICES="0" python nlp_script.py configs/nlp_script/single_gpu/mixed_precision.yml

# # Multi GPU
# echo "Running Multi GPU scripts..."
# CUDA_VISIBLE_DEVICES="0,1" accelerate launch nlp_script.py configs/nlp_script/multi_gpu/baseline.yml
# CUDA_VISIBLE_DEVICES="0,1" accelerate launch nlp_script.py configs/nlp_script/multi_gpu/mixed_precision.yml
# CUDA_VISIBLE_DEVICES="0,1" accelerate launch nlp_script.py configs/nlp_script/multi_gpu/adjusted_bs.yml
# CUDA_VISIBLE_DEVICES="0,1" accelerate launch nlp_script.py configs/nlp_script/multi_gpu/adjusted_bs_mixed_precision.yml
# CUDA_VISIBLE_DEVICES="0,1" accelerate launch nlp_script.py configs/nlp_script/multi_gpu/adjusted_bs_double_lr.yml
# CUDA_VISIBLE_DEVICES="0,1" accelerate launch nlp_script.py configs/nlp_script/multi_gpu/adjusted_bs_double_lr_mixed_precision.yml

# TPU
echo "Runing TPU scripts..."
for config in configs/baseline_nlp/*.yml; do
    rm -rf bert_base_cased_tpu_acccelerate_experiments/{*,.*}
    aim init
    accelerate launch xla_nlp_script.py "$config"
done