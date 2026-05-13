import argparse
import datetime
import json
import random
import time
from pathlib import Path
from collections import namedtuple
from functools import partial

import os
import numpy as np
import torch


import torch_patch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
import datasets.samplers as samplers
from datasets.coco_eval import CocoEvaluator
from datasets import build_dataset, get_coco_api_from_dataset
from engine import train_one_epoch
from models import build_model
from models.postprocessors import build_postprocessors
from tools.load_pretrained_weights import pre_trained_model_to_finetune
import opts
# os.environ["CUDA_VISIBLE_DEVICES"] = '7'
# os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
# os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'
#
def main(args):
    # set environ
    os.environ["MDETR_CPU_REDUCE"] = "1"

    args.masks = False
    assert args.dataset_file in ["rsvg", "rsvg_mm", "refcoco", "refcoco+", "refcocog", "all"]

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # lr_backbone_names = ["backbone.0", "text_encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    # for n, p in model_without_ddp.named_parameters():
    #    print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_text_encoder_names)
                 and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_text_encoder_names) and p.requires_grad],
            "lr": args.lr_text_encoder,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)

    # build train  dataset
    if args.dataset_file != "all":
        dataset_train = build_dataset(args.dataset_file, image_set='train', args=args)
        print('trainset:', len(dataset_train))
    else:
        dataset_names = ["refcoco", "refcoco+", "refcocog"]
        dataset_train = torch.utils.data.ConcatDataset(
            [build_dataset(name, image_set="train", args=args) for name in dataset_names]
        )

    # dataset_train = build_dataset(args.dataset_file, image_set='train', args=args)
    print('trainset:', len(dataset_train))


    print("\nTrain dataset sample number: ", len(dataset_train))
    print("\n")

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)

    if args.pretrained_weights != None:
        checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
        checkpoint_dict = pre_trained_model_to_finetune(checkpoint, args)
        model_without_ddp.load_state_dict(checkpoint_dict, strict=False)
        print("============================================>")


    if args.dataset_file != "all":
        dataset_names = [args.dataset_file]
    else:
        dataset_names = ["refcoco", "refcoco+", "refcocog"]


    # build evaluator list for dataset_val
    def build_evaluator_list(base_ds, dataset_name):
        """Helper function to build the list of evaluators for a given dataset"""
        evaluator_list = []
        iou_types = ["bbox"]
        if args.masks:
            iou_types.append("segm")

        evaluator_list.append(CocoEvaluator(base_ds, tuple(iou_types), useCats=False))
        # TODO: currently ont support RefExpEvaluator (memory error)
        return evaluator_list



    output_dir = Path(args.output_dir)
    if args.resume:
        print("Resume from {}".format(args.resume))
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        
        
        
        model_state_dict = model_without_ddp.state_dict()
        checkpoint_state_dict = checkpoint['model']
        
        
        model_keys = set(model_state_dict.keys())
        checkpoint_keys = set(checkpoint_state_dict.keys())
        
        
        missing_keys = list(model_keys - checkpoint_keys)
        # unexpected_keys: 
        unexpected_keys = list(checkpoint_keys - model_keys)
        # matched_keys: 
        matched_keys = list(model_keys & checkpoint_keys)
        
        
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        
        
        filtered_checkpoint = {k: v for k, v in checkpoint_state_dict.items() if k in matched_keys}
        model_without_ddp.load_state_dict(filtered_checkpoint, strict=False)
        
        
        print("=" * 80)
        print("Checkpoint Loading Summary:")
        print("=" * 80)
        print(f"✓ Loaded {len(matched_keys)} parameters from checkpoint")
        
        if len(missing_keys) > 0:
            print(f"\n⚠ Missing Keys ({len(missing_keys)} parameters):")
            print("  These parameters exist in current model but not in checkpoint.")
            print("  They will be initialized with default values (random initialization).")
            
            missing_by_module = {}
            for key in missing_keys:
                module = key.split('.')[0] if '.' in key else key
                if module not in missing_by_module:
                    missing_by_module[module] = []
                missing_by_module[module].append(key)
            
            for module, keys in missing_by_module.items():
                print(f"  - {module}: {len(keys)} parameters")
                
                for key in keys[:5]:
                    print(f"    • {key}")
                if len(keys) > 5:
                    print(f"    ... and {len(keys) - 5} more")
        
        if len(unexpected_keys) > 0:
            print(f"\n⊘ Unexpected Keys ({len(unexpected_keys)} parameters):")
            print("  These parameters exist in checkpoint but not in current model.")
            print("  They will be ignored (kept in checkpoint but not loaded).")
            
            unexpected_by_module = {}
            for key in unexpected_keys:
                module = key.split('.')[0] if '.' in key else key
                if module not in unexpected_by_module:
                    unexpected_by_module[module] = []
                unexpected_by_module[module].append(key)
            
            for module, keys in unexpected_by_module.items():
                print(f"  - {module}: {len(keys)} parameters")
                
                for key in keys[:5]:
                    print(f"    • {key}")
                if len(keys) > 5:
                    print(f"    ... and {len(keys) - 5} more")
        
        print("=" * 80)
        # ================================================================
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            
            
            print("\n" + "=" * 80)
            print("Optimizer Loading:")
            print("=" * 80)
            
            try:
                checkpoint_optimizer = checkpoint['optimizer']
                
                
                if len(checkpoint_optimizer['param_groups']) == len(optimizer.param_groups):
                    
                    can_load = True
                    param_mismatch_info = []
                    
                    for idx, (ckpt_pg, curr_pg) in enumerate(zip(checkpoint_optimizer['param_groups'], optimizer.param_groups)):
                        ckpt_param_count = len(ckpt_pg['params'])
                        curr_param_count = len(curr_pg['params'])
                        if ckpt_param_count != curr_param_count:
                            can_load = False
                            param_mismatch_info.append(f"  Group {idx}: checkpoint has {ckpt_param_count} params, current has {curr_param_count} params")
                    
                    if can_load:
                        
                        optimizer.load_state_dict(checkpoint_optimizer)
                        for pg, pg_old in zip(optimizer.param_groups, p_groups):
                            pg['lr'] = pg_old['lr']
                            pg['initial_lr'] = pg_old['initial_lr']
                        print("✓ Loaded optimizer state from checkpoint (all parameters matched)")
                        
                        try:
                            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                            print("✓ Loaded lr_scheduler state from checkpoint")
                        except Exception as e:
                            print(f"⚠ Warning: Failed to load lr_scheduler state: {e}")
                            print("  Continuing with fresh lr_scheduler state.")
                    else:
                        
                        print("⚠ Optimizer parameter groups size mismatch detected:")
                        for info in param_mismatch_info:
                            print(info)
                        print("\n  This is expected when model structure changes (e.g., adding new modules).")
                        print("  New parameters will use fresh optimizer state (default initialization).")
                        print("  Existing parameters' optimizer state cannot be loaded due to structure mismatch.")
                        print("  Continuing training with fresh optimizer state for all parameters.")
                else:
                    
                    print(f"⚠ Optimizer parameter groups count mismatch:")
                    print(f"  Checkpoint has {len(checkpoint_optimizer['param_groups'])} groups")
                    print(f"  Current model has {len(optimizer.param_groups)} groups")
                    print("\n  This is expected when model structure changes (e.g., adding new modules).")
                    print("  Continuing training with fresh optimizer state.")
                    
            except (ValueError, KeyError) as e:
                print(f"⚠ Warning: Failed to load optimizer state: {e}")
                print("  This is expected when model structure changes (e.g., adding new modules).")
                print("  Continuing training with fresh optimizer state.")
            
            print("=" * 80)
            
            
            
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            
            
            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch'] + 1
                try:
                    
                    lr_scheduler.step(lr_scheduler.last_epoch)
                except AttributeError:
                    
                    for _ in range(args.start_epoch):
                        lr_scheduler.step()
                    print(f"Starting from epoch {args.start_epoch} with fresh optimizer and lr_scheduler state.")


    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, args=args)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every epochs
            # if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 1 == 0:
            if (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ReferFormer pretrain training and evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

