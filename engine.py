"""
Train and eval functions used in main.py
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import math
from models import postprocessors
import os
import sys
from typing import Iterable

import torch
import torch.distributed as dist

import util.misc as utils


def adjust_loss_weights(epoch, stage1_epochs=20, stage2_epochs=50, stage3_epochs=70):
    
    if epoch < stage1_epochs:
       
        progress = epoch / stage1_epochs
        otsm_weight = 0.15 + 0.10 * progress
        fts_weight = 0.0  
    elif epoch < stage2_epochs:
       
        progress = (epoch - stage1_epochs) / (stage2_epochs - stage1_epochs)
        fts_weight = 0.3 + 0.4 * progress
        otsm_weight = 0.2 - 0.05 * progress
    else:
       
        progress = (epoch - stage2_epochs) / (stage3_epochs - stage2_epochs)
        fts_weight = 0.8 + 0.4 * progress
        otsm_weight = 0.15 - 0.05 * progress
    
    return fts_weight, otsm_weight


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    args=None):
    model.train()
    criterion.train()
    
    
    if args is not None and getattr(args, 'use_dynamic_loss_weights', False):
        stage1_epochs = getattr(args, 'stage1_epochs', 20)
        stage2_epochs = getattr(args, 'stage2_epochs', 50)
        stage3_epochs = getattr(args, 'stage3_epochs', 70)
        
        fts_weight, otsm_weight = adjust_loss_weights(
            epoch, stage1_epochs, stage2_epochs, stage3_epochs
        )
        
       
        if 'loss_foreground_token_selector' in criterion.weight_dict:
            criterion.weight_dict['loss_foreground_token_selector'] = fts_weight
        if 'loss_match' in criterion.weight_dict:
            criterion.weight_dict['loss_match'] = otsm_weight
        
       
        if utils.is_main_process():
            print(f"Epoch {epoch}: FTS weight = {fts_weight:.4f}, OTSM weight = {otsm_weight:.4f}")
    
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        targets = utils.targets_to(targets, device)

        outputs = model(samples, captions, targets) 
        loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

       
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

   
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}







