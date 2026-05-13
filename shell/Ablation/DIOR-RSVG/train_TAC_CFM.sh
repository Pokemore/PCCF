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


python main.py \
    --dataset_file rsvg \
    --backbone resnet50 \
    --batch_size 1 \
    --epochs 20 \
    --lr_drop 40 \
    --num_queries 10 \
    --num_frames 1 \
    --output_dir /root/Documents/Model/Ablation/DIOR-RSVG/TAC_CFM \
    --binary \
    --with_box_refine \
    --tokenizer_path /root/Documents/PreTrained/RoBERTa \
    --text_encoder_path /root/Documents/PreTrained/RoBERTa \
    --rsvg_path /root/Documents/Dataset/DIOR-RSVG \
    --stage1_epochs 20 \
    --stage2_epochs 50 \
    --stage3_epochs 70 \
    --use_conditional_norm \
    --cn_insertion_mode after_input_proj \
    --cn_use_residual \
    --use_dynamic_loss_weights \
    --activation_type glu \
    --cn_beta_activation tanh \
    --cn_gamma_activation relu \
    --cn_f_cond_refine

python main.py \
    --dataset_file rsvg \
    --backbone resnet50 \
    --batch_size 1 \
    --epochs 50 \
    --lr_drop 40 \
    --num_queries 10 \
    --num_frames 1 \
    --output_dir /root/Documents/Model/Ablation/DIOR-RSVG/TAC_CFM \
    --binary \
    --with_box_refine \
    --tokenizer_path /root/Documents/PreTrained/RoBERTa \
    --text_encoder_path /root/Documents/PreTrained/RoBERTa \
    --rsvg_path /root/Documents/Dataset/DIOR-RSVG \
    --resume /root/Documents/Model/Ablation/DIOR-RSVG/TAC_CFM/checkpoint.pth \
    --stage1_epochs 20 \
    --stage2_epochs 50 \
    --stage3_epochs 70 \
    --use_conditional_norm \
    --cn_insertion_mode after_input_proj \
    --cn_use_residual \
    --use_dynamic_loss_weights \
    --activation_type glu \
    --cn_beta_activation tanh \
    --cn_gamma_activation relu \
    --cn_f_cond_refine \
    --use_dsats \
    --fts_loss_coef 0.5 \

python main.py \
    --dataset_file rsvg \
    --backbone resnet50 \
    --batch_size 1 \
    --epochs 70 \
    --lr_drop 60 \
    --num_queries 10 \
    --num_frames 1 \
    --output_dir /root/Documents/Model/Ablation/DIOR-RSVG/TAC_CFM \
    --binary \
    --with_box_refine \
    --tokenizer_path /root/Documents/PreTrained/RoBERTa \
    --text_encoder_path /root/Documents/PreTrained/RoBERTa \
    --rsvg_path /root/Documents/Dataset/DIOR-RSVG \
    --resume /root/Documents/Model/Ablation/DIOR-RSVG/TAC_CFM/checkpoint.pth \
    --lr 0.00005 \
    --stage1_epochs 20 \
    --stage2_epochs 50 \
    --stage3_epochs 70 \
    --use_conditional_norm \
    --cn_insertion_mode after_input_proj \
    --cn_use_residual \
    --use_dynamic_loss_weights \
    --activation_type glu \
    --cn_beta_activation tanh \
    --cn_gamma_activation relu \
    --cn_f_cond_refine \
    --use_dsats \
    --fts_loss_coef 0.5 \