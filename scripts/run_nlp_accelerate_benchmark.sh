#!/bin/bash

# TPU
echo "Runing TPU scripts..."
accelerate launch xla_nlp_script_accelerate.py configs/baseline_nlp/8lr.yml
accelerate launch xla_nlp_script_accelerate.py configs/baseline_nlp/baseline.yml
