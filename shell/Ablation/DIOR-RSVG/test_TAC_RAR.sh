#!/bin/sh
set -euo pipefail

# Activate your conda environment.
source /opt/conda/bin/activate PC2F

# Repository root and working directory.
ROOT="/root/Documents/Code/PC2F"
cd "${ROOT}"

# GPU selection for testing.
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

python3 -c "import torch_patch; exec(open('inference_rsvg.py').read())" \
    --dataset_file rsvg --num_queries 10 \
    --with_box_refine \
    --binary \
    --freeze_text_encoder \
    --resume /root/Documents/Model/Ablation/DIOR-RSVG/TAC_RAR/checkpoint.pth \
    --backbone resnet50 \
    --rsvg_path /root/Documents/Dataset/DIOR-RSVG \
    --use_conditional_norm \
    --cn_insertion_mode after_input_proj \
    --cn_use_residual \
    --use_dynamic_loss_weights \
    --activation_type glu \
    --cn_beta_activation tanh \
    --cn_gamma_activation relu \
    --cn_f_cond_refine \
    --use_solution_f3 \
    --solution_f3_max_fg_tokens 500 \
    --solution_f3_use_lang_guidance \
    --solution_f3_use_fg_guidance \
    --use_adaptive_cn_f3_fusion \