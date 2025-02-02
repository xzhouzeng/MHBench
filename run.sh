#!/bin/bash
cd baselines

# conda activate videollm
python  ../main.py \
    --model_name VideoChat2-Mistral \
    --eval_task classification \
    --output_dir ../output \
    # --use_mcd

