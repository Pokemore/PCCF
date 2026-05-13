import torch_patch  # 导入PyTorch兼容性补丁
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import util.misc as utils
from util.misc import AverageMeter
from models import build_model
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
import warnings
import shutil
from collections import defaultdict
import os
from PIL import Image, ImageDraw, ImageFont
from datasets import build_dataset, get_coco_api_from_dataset
import opts
from torch.utils.data import DataLoader

from tools.colormap import colormap


def setup_chinese_font():
    

    chinese_fonts = [
        'SimHei',   
        'Microsoft YaHei',  
        'WenQuanYi Micro Hei',  
        'Noto Sans CJK SC',  
        'Source Han Sans CN',  
        'Arial Unicode MS',  
        'STHeiti',  
        'STSong',  
    ]
    
    
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            matplotlib.rcParams['font.sans-serif'] = [font_name] + matplotlib.rcParams['font.sans-serif']
            matplotlib.rcParams['axes.unicode_minus'] = False
            return True
    
    
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False
    return False


_has_chinese_font = setup_chinese_font()
if not _has_chinese_font:
    
    warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing.*')

# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()

Visualize_bbox = False #False #True
save_visualize_path_prefix = "test_output"
version = "test"



def main(args):
    args.masks = False
    # args.batch_size == 1
    print("Inference only supports for batch size = 1")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.visualize:
        if not os.path.exists(save_visualize_path_prefix):
            os.makedirs(save_visualize_path_prefix)

    test_dataset = build_dataset(args.dataset_file, image_set='test', args=args)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             pin_memory=True, drop_last=True, num_workers=4)

    # model
    model, criterion, _ = build_model(args)
    device = args.device
    model.to(device)

    # model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
    else:
        raise ValueError('Please specify the checkpoint for inference.')

    # start inference
    evaluate(test_loader, model, args)

def evaluate(test_loader, model, args):
    batch_time = AverageMeter()
    acc5 = AverageMeter()
    acc6 = AverageMeter()
    acc7 = AverageMeter()
    acc8 = AverageMeter()
    acc9 = AverageMeter()
    meanIoU = AverageMeter()
    inter_area = AverageMeter()
    union_area = AverageMeter()

    device = args.device
    model.eval()
    end = time.time()

    img_list = []
    count=0
    
    
    stats_data_by_score = defaultdict(lambda: {'samples': 0, 'final_ious': [], 'potential_ious': []})
    
    stats_data_by_iou = defaultdict(lambda: {'final_count': 0, 'potential_count': 0})
    
   
    if getattr(args, 'visualize_all_boxes', False):
       
        user_dir = getattr(args, 'visualize_all_boxes_dir', None)
        
       
        if user_dir and user_dir != 'visualize_all_boxes':
            
            if os.path.isabs(user_dir):
                args.visualize_all_boxes_dir = user_dir
            else:
                
                if hasattr(args, 'resume') and args.resume:
                    checkpoint_dir = os.path.dirname(os.path.abspath(args.resume))
                    if checkpoint_dir:
                        args.visualize_all_boxes_dir = os.path.join(checkpoint_dir, user_dir)
                    else:
                        args.visualize_all_boxes_dir = user_dir
                else:
                    args.visualize_all_boxes_dir = user_dir
        else:
            
            if hasattr(args, 'resume') and args.resume:
                checkpoint_dir = os.path.dirname(os.path.abspath(args.resume))
                if checkpoint_dir:
                    args.visualize_all_boxes_dir = os.path.join(checkpoint_dir, 'visualize_all_boxes')
                else:
                    args.visualize_all_boxes_dir = 'visualize_all_boxes'
            else:
                args.visualize_all_boxes_dir = 'visualize_all_boxes'
    
    for batch_idx, (img, targets, dw, dh, img_path, ratio) in enumerate(test_loader):
        h_resize, w_resize = img.shape[ -2:]
        img = img.to(device)
        captions = targets["caption"]
        size = torch.as_tensor([int(h_resize), int(w_resize)]).to(device)
        target = {"size": size}

        with torch.no_grad():
            outputs = model(img, captions, [target])

        
        pred_logits = outputs["pred_logits"][0]
        pred_bbox = outputs["pred_boxes"][0]
        pred_score = pred_logits.sigmoid()  # [t, q, k]
        pred_score = pred_score.squeeze(0)# [q, k]
        
        max_score, _ = pred_score.max(-1)  # [q,]
        _, max_ind = max_score.max(-1)  # [1,] # which query
        final_score = max_score[max_ind].item()
        pred_bbox_selected = pred_bbox[0, max_ind]  # [xc,yc, w_b, h_b]

        
        all_pred_boxes_resized = []
        for q_idx in range(pred_bbox.shape[1]):
            box_cxcywh = pred_bbox[0, q_idx].detach()
            box_xyxy = rescale_bboxes(box_cxcywh, (w_resize, h_resize)).numpy()
            
            box_xyxy[0], box_xyxy[2] = (box_xyxy[0] - dw) / ratio, (box_xyxy[2] - dw) / ratio
            box_xyxy[1], box_xyxy[3] = (box_xyxy[1] - dh) / ratio, (box_xyxy[3] - dh) / ratio
            all_pred_boxes_resized.append(box_xyxy)
        all_pred_boxes_resized = np.array(all_pred_boxes_resized)  # [num_queries, 4]

        
        pred_bbox = rescale_bboxes(pred_bbox_selected.detach(), (w_resize, h_resize)).numpy()
        target_bbox = rescale_bboxes(targets["boxes"].squeeze(), (w_resize, h_resize)).numpy()

       
        pred_bbox[0], pred_bbox[2] = (pred_bbox[0] - dw) / ratio, (pred_bbox[2] - dw) / ratio
        pred_bbox[1], pred_bbox[3] = (pred_bbox[1] - dh) / ratio, (pred_bbox[3] - dh) / ratio
        target_bbox[0], target_bbox[2] = (target_bbox[0] - dw) / ratio, (target_bbox[2] - dw) / ratio
        target_bbox[1], target_bbox[3] = (target_bbox[1] - dh) / ratio, (target_bbox[3] - dh) / ratio
        
       
        all_ious = []
        for q_idx in range(len(all_pred_boxes_resized)):
            iou, _, _ = bbox_iou(all_pred_boxes_resized[q_idx], target_bbox)
            all_ious.append(iou.item())
        all_ious = np.array(all_ious)
        
      
        final_iou = all_ious[max_ind.item()] if len(all_ious) > 0 else 0.0
        potential_iou = np.max(all_ious) if len(all_ious) > 0 else 0.0
        
       
        if final_score < 0.9:
            score_interval = 0.1
            score_lower = int(final_score / score_interval) * score_interval
            score_upper = score_lower + score_interval
        else:
            score_interval = 0.2
            score_lower = 0.9
            score_upper = 1.0
        
        interval_key = f"{score_lower:.1f}-{score_upper:.1f}"
        stats_data_by_score[interval_key]['samples'] += 1
        stats_data_by_score[interval_key]['final_ious'].append(final_iou)
        stats_data_by_score[interval_key]['potential_ious'].append(potential_iou)
        
        
        iou_interval_size = 0.05  # 5%
        
        final_iou_lower = int(final_iou / iou_interval_size) * iou_interval_size
        final_iou_upper = final_iou_lower + iou_interval_size
        if final_iou >= 1.0:
            final_iou_lower = 0.95
            final_iou_upper = 1.0
        final_iou_key = f"{final_iou_lower:.2f}-{final_iou_upper:.2f}"
        stats_data_by_iou[final_iou_key]['final_count'] += 1
        
        
        potential_iou_lower = int(potential_iou / iou_interval_size) * iou_interval_size
        potential_iou_upper = potential_iou_lower + iou_interval_size
        if potential_iou >= 1.0:
            potential_iou_lower = 0.95
            potential_iou_upper = 1.0
        potential_iou_key = f"{potential_iou_lower:.2f}-{potential_iou_upper:.2f}"
        stats_data_by_iou[potential_iou_key]['potential_count'] += 1
        
       
        if getattr(args, 'visualize_all_boxes', False):
            
            all_scores = max_score.cpu().numpy()  
            
           
            save_dir = getattr(args, 'visualize_all_boxes_dir', 'visualize_all_boxes')
            
            
            if getattr(args, 'visualize_all_boxes_by_score', False):
                score_interval = getattr(args, 'visualize_all_boxes_score_interval', 0.1)
               
                if final_score >= 1.0:
                    score_lower = 1.0 - score_interval
                    score_upper = 1.0
                else:
                    score_lower = int(final_score / score_interval) * score_interval
                    score_upper = score_lower + score_interval
                
                score_subdir = f"score_{score_lower:.1f}-{score_upper:.1f}"
                save_dir = os.path.join(save_dir, score_subdir)
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            
            img_name = img_path[0].split('/')[-1]
            img_base_name = os.path.splitext(img_name)[0]
            img_ext = os.path.splitext(img_name)[1]

            save_filename = f"{img_base_name}_score{final_score:.4f}{img_ext}"
            save_path = os.path.join(save_dir, save_filename)
            
            
            visualize_all_boxes(
                img_path=img_path[0],
                all_pred_boxes=all_pred_boxes_resized,
                all_scores=all_scores,
                all_ious=all_ious,
                target_bbox=target_bbox,
                final_score=final_score,
                final_box_idx=max_ind.item(),
                save_path=save_path,
                captions=captions
            )
            
           
            if potential_iou > final_iou:
                
                base_save_dir = getattr(args, 'visualize_all_boxes_dir', 'visualize_all_boxes')
                optimizable_dir = os.path.join(base_save_dir, 'optimizable_samples')
                if not os.path.exists(optimizable_dir):
                    os.makedirs(optimizable_dir)
                
                
                iou_diff = potential_iou - final_iou
                optimizable_filename = f"{img_base_name}_final{final_iou:.3f}_potential{potential_iou:.3f}_diff{iou_diff:.3f}{img_ext}"
                optimizable_path = os.path.join(optimizable_dir, optimizable_filename)
                
                
                shutil.copy2(save_path, optimizable_path)

        if Visualize_bbox:
                source_img = Image.open(img_path[0]).convert('RGB')  # PIL image

                draw = ImageDraw.Draw(source_img)
                draw_boxes = pred_bbox.tolist()

                # draw boxes
                xmin, ymin, xmax, ymax = draw_boxes[0:4]

                # draw_boxes_gt = target_bbox.tolist()
                # xmin_gt, ymin_gt, xmax_gt, ymax_gt = draw_boxes_gt[0:4]

                draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=tuple(color_list[9]), width=2)
                # draw.rectangle(((xmin_gt, ymin_gt), (xmax_gt, ymax_gt)), outline=tuple(color_list[9]), width=2)
                # fontStyle = ImageFont.truetype("SimHei.ttf", 30)
                # draw.text((20, 20), captions[0], (200, 0, 0), font=fontStyle)
                # save
                save_visualize_path_dir = os.path.join(save_visualize_path_prefix, version)
                if not os.path.exists(save_visualize_path_dir):
                    os.makedirs(save_visualize_path_dir)
                img_name = img_path[0].split('/')[-1]
                if img_name not in img_list:
                    img_list.append(img_name)
                else:
                    count += 1
                    img_name = str(count) + '_' + img_name
                save_visualize_path = os.path.join(save_visualize_path_dir, img_name)
                source_img.save(save_visualize_path)

        # box iou
        iou, interArea, unionArea = bbox_iou(pred_bbox, target_bbox)
        cumInterArea = np.sum(np.array(interArea.data.numpy()))
        cumUnionArea = np.sum(np.array(unionArea.data.numpy()))
        # accuracy
        accu5 = np.sum(np.array((iou.data.numpy() > 0.5), dtype=float)) / 1
        accu6 = np.sum(np.array((iou.data.numpy() > 0.6), dtype=float)) / 1
        accu7 = np.sum(np.array((iou.data.numpy() > 0.7), dtype=float)) / 1
        accu8 = np.sum(np.array((iou.data.numpy() > 0.8), dtype=float)) / 1
        accu9 = np.sum(np.array((iou.data.numpy() > 0.9), dtype=float)) / 1

        # metrics  7
        meanIoU.update(torch.mean(iou).item(), img.size(0))
        inter_area.update(cumInterArea)
        union_area.update(cumUnionArea)


        acc5.update(accu5, img.size(0))
        acc6.update(accu6, img.size(0))
        acc7.update(accu7, img.size(0))
        acc8.update(accu8, img.size(0))
        acc9.update(accu9, img.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 50 == 0:
            print_str = '[{0}/{1}]\t' \
                        'Time {batch_time.avg:.3f}\t' \
                        'acc@0.5: {acc5.avg:.4f}\t' \
                        'acc@0.6: {acc6.avg:.4f}\t' \
                        'acc@0.7: {acc7.avg:.4f}\t' \
                        'acc@0.8: {acc8.avg:.4f}\t' \
                        'acc@0.9: {acc9.avg:.4f}\t' \
                        'meanIoU: {meanIoU.avg:.4f}\t' \
                        'cumuIoU: {cumuIoU:.4f}\t' \
                .format( \
                batch_idx, len(test_loader), batch_time=batch_time, \
                acc5=acc5, acc6=acc6, acc7=acc7, acc8=acc8, acc9=acc9, \
                meanIoU=meanIoU, cumuIoU=inter_area.sum / union_area.sum)
            print(print_str)
            # logging.info(print_str)
    final_str = 'acc@0.5: {acc5.avg:.4f}\t' 'acc@0.6: {acc6.avg:.4f}\t' 'acc@0.7: {acc7.avg:.4f}\t' \
                'acc@0.8: {acc8.avg:.4f}\t' 'acc@0.9: {acc9.avg:.4f}\t' \
                'meanIoU: {meanIoU.avg:.4f}\t' 'cumuIoU: {cumuIoU:.4f}\t' \
        .format(acc5=acc5, acc6=acc6, acc7=acc7, acc8=acc8, acc9=acc9, \
                meanIoU=meanIoU, cumuIoU=inter_area.sum / union_area.sum)
    print(final_str)
    print(version)
    
   
    if getattr(args, 'visualize_all_boxes', False) and len(stats_data_by_score) > 0:
       
        save_dir = getattr(args, 'visualize_all_boxes_dir', 'visualize_all_boxes')
        plot_statistics_by_score(stats_data_by_score, save_dir)
        plot_statistics_by_iou(stats_data_by_iou, save_dir)




def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = torch.tensor(box1[0]), torch.tensor(box1[1]), torch.tensor(box1[2]), torch.tensor(box1[3])
    b2_x1, b2_y1, b2_x2, b2_y2 = torch.tensor(box2[0]), torch.tensor(box2[1]), torch.tensor(box2[2]), torch.tensor(box2[3])

    # get the coordinates of the intersection rectangle

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area

    return (inter_area + 1e-6) / (union_area + 1e-6), inter_area, union_area

# visuaize functions
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(0)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=0)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# Visualization functions
def draw_reference_points(draw, reference_points, img_size, color):
    W, H = img_size
    for i, ref_point in enumerate(reference_points):
        init_x, init_y = ref_point
        x, y = W * init_x, H * init_y
        cur_color = color
        draw.line((x - 10, y, x + 10, y), tuple(cur_color), width=4)
        draw.line((x, y - 10, x, y + 10), tuple(cur_color), width=4)


def draw_sample_points(draw, sample_points, img_size, color_list):
    alpha = 255
    for i, samples in enumerate(sample_points):
        for sample in samples:
            x, y = sample
            cur_color = color_list[i % len(color_list)][::-1]
            cur_color += [alpha]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2),
                         fill=tuple(cur_color), outline=tuple(cur_color), width=1)


def vis_add_mask(img, mask, color):
    origin_img = np.asarray(img.convert('RGB')).copy()
    color = np.array(color)

    mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8')  # np
    mask = mask > 0.5

    origin_img[mask] = origin_img[mask] * 0.5 + color * 0.5
    origin_img = Image.fromarray(origin_img)
    return origin_img


def visualize_all_boxes(img_path, all_pred_boxes, all_scores, all_ious, target_bbox, 
                         final_score, final_box_idx, save_path, captions=None):
  
   
    PRED_BOX_COLOR = (0, 100, 255)  
    
    FINAL_BOX_COLOR = (255, 0, 0)  

    GT_BOX_COLOR = (0, 255, 0)  
    
   
    source_img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(source_img)
    
    
    if target_bbox is not None:
        xmin_gt, ymin_gt, xmax_gt, ymax_gt = target_bbox[0:4]
        draw.rectangle(((xmin_gt, ymin_gt), (xmax_gt, ymax_gt)), 
                      outline=GT_BOX_COLOR, width=4)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except:
            font = ImageFont.load_default()
        draw.text((xmin_gt, ymin_gt - 25), "GT", fill=GT_BOX_COLOR, font=font)
    
    
    num_boxes = len(all_pred_boxes)
    
    for box_idx in range(num_boxes):
        box = all_pred_boxes[box_idx]
        score = all_scores[box_idx]
        iou = all_ious[box_idx] if all_ious is not None else 0.0
        
        xmin, ymin, xmax, ymax = box[0:4]
        
        
        is_final_box = (box_idx == final_box_idx)
        
        
        if is_final_box:
            box_color = FINAL_BOX_COLOR
            line_width = 4  
            label_prefix = "★ FINAL / 最终"
        else:
            box_color = PRED_BOX_COLOR
            line_width = 2  
        
        
        draw.rectangle(((xmin, ymin), (xmax, ymax)), 
                      outline=box_color, width=line_width)
        
        
        if is_final_box:
            label_text = f"{label_prefix}\nQ{box_idx}: S={score:.3f}, IoU={iou:.3f}"
        else:
            label_text = f"Q{box_idx}: S={score:.3f}, IoU={iou:.3f}"
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        
        try:
            
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
           
            text_width, text_height = draw.textsize(label_text, font=font)
        
       
        text_y = ymin - text_height - 2 if ymin > text_height + 2 else ymax + 2
        draw.rectangle([(xmin, text_y - text_height - 2), (xmin + text_width + 4, text_y + 2)], 
                      fill=(255, 255, 255))
        
        
        draw.text((xmin + 2, text_y - text_height), label_text, 
                 fill=tuple(box_color), font=font)
    
    
    if captions:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        caption_text = captions[0] if isinstance(captions, list) else str(captions)
        try:
            
            bbox = draw.textbbox((0, 0), caption_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
           
            text_width, text_height = draw.textsize(caption_text, font=font)
        
        draw.rectangle([(10, 10), (10 + text_width + 10, 10 + text_height + 10)], 
                      fill=(0, 0, 0))
        draw.text((15, 15), caption_text, fill=(255, 255, 255), font=font)
    
    
    source_img.save(save_path)


def plot_statistics_by_score(stats_data, save_dir):
   
    
    figure_dir = os.path.join(save_dir, 'figure')
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    
    
    intervals = sorted(stats_data.keys(), key=lambda x: float(x.split('-')[0]))
    
    
    sample_counts = []
    avg_final_ious = []
    avg_potential_ious = []
    interval_labels = []
    
    for interval in intervals:
        data = stats_data[interval]
        sample_counts.append(data['samples'])
        if len(data['final_ious']) > 0:
            avg_final_ious.append(np.mean(data['final_ious']))
        else:
            avg_final_ious.append(0.0)
        if len(data['potential_ious']) > 0:
            avg_potential_ious.append(np.mean(data['potential_ious']))
        else:
            avg_potential_ious.append(0.0)
        interval_labels.append(interval)
    
    
    all_final_ious = []
    all_potential_ious = []
    for data in stats_data.values():
        all_final_ious.extend(data['final_ious'])
        all_potential_ious.extend(data['potential_ious'])
    
    overall_avg_final_iou = np.mean(all_final_ious) if len(all_final_ious) > 0 else 0.0
    overall_avg_potential_iou = np.mean(all_potential_ious) if len(all_potential_ious) > 0 else 0.0
    
    
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    x_pos = np.arange(len(intervals))
    bars1 = ax1.bar(x_pos, sample_counts, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
    if _has_chinese_font:
        ax1.set_xlabel('Score Interval ', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Sample Count ', fontsize=12, fontweight='bold')
        ax1.set_title('Sample Count by Score Interval ', fontsize=14, fontweight='bold')
    else:
        ax1.set_xlabel('Score Interval', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
        ax1.set_title('Sample Count by Score Interval', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(interval_labels, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    
    for i, (bar, count) in enumerate(zip(bars1, sample_counts)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_path1 = os.path.join(figure_dir, 'statistics_by_score_interval_samples.png')
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    print(f"Sample count plot saved to : {save_path1}")
    plt.close()
    
    
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))
    x_pos2 = np.arange(len(intervals))
    width = 0.35
    
    if _has_chinese_font:
        bars2 = ax2.bar(x_pos2 - width/2, avg_final_ious, width, 
                        label='Output IoU ', color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars3 = ax2.bar(x_pos2 + width/2, avg_potential_ious, width,
                        label='Potential IoU ', color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        
        ax2.axhline(y=overall_avg_final_iou, color='#FF6B6B', linestyle='--', linewidth=2, 
                    label=f'Avg Output IoU : {overall_avg_final_iou:.3f}')
        ax2.axhline(y=overall_avg_potential_iou, color='#4ECDC4', linestyle='--', linewidth=2,
                    label=f'Avg Potential IoU : {overall_avg_potential_iou:.3f}')
        
        ax2.set_xlabel('Score Interval ', fontsize=12, fontweight='bold')
        ax2.set_ylabel('IoU', fontsize=12, fontweight='bold')
        ax2.set_title('IoU Comparison by Score Interval ', fontsize=14, fontweight='bold')
    else:
        bars2 = ax2.bar(x_pos2 - width/2, avg_final_ious, width, 
                        label='Output IoU', color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars3 = ax2.bar(x_pos2 + width/2, avg_potential_ious, width,
                        label='Potential IoU', color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        
        ax2.axhline(y=overall_avg_final_iou, color='#FF6B6B', linestyle='--', linewidth=2, 
                    label=f'Avg Output IoU: {overall_avg_final_iou:.3f}')
        ax2.axhline(y=overall_avg_potential_iou, color='#4ECDC4', linestyle='--', linewidth=2,
                    label=f'Avg Potential IoU: {overall_avg_potential_iou:.3f}')
        
        ax2.set_xlabel('Score Interval', fontsize=12, fontweight='bold')
        ax2.set_ylabel('IoU', fontsize=12, fontweight='bold')
        ax2.set_title('IoU Comparison by Score Interval', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(interval_labels, rotation=45, ha='right')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 1.0])
    
    
    for bars in [bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:  
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    save_path2 = os.path.join(figure_dir, 'statistics_by_score_interval_iou.png')
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"IoU comparison plot saved to : {save_path2}")
    plt.close()
    
    
    txt_path = os.path.join(figure_dir, 'statistics_by_score_data.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Statistics by Score Interval\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Overall Average Output IoU : {overall_avg_final_iou:.4f}\n")
        f.write(f"Overall Average Potential IoU : {overall_avg_potential_iou:.4f}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        for interval in intervals:
            data = stats_data[interval]
            f.write(f"Interval : {interval}\n")
            f.write(f"  Sample Count : {data['samples']}\n")
            if len(data['final_ious']) > 0:
                f.write(f"  Average Output IoU : {np.mean(data['final_ious']):.4f}\n")
            if len(data['potential_ious']) > 0:
                f.write(f"  Average Potential IoU : {np.mean(data['potential_ious']):.4f}\n")
            f.write("\n")
    
    print(f"Statistics data saved to : {txt_path}")


def plot_statistics_by_iou(stats_data, save_dir):
  
    
    figure_dir = os.path.join(save_dir, 'figure')
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    
    
    intervals = sorted(stats_data.keys(), key=lambda x: float(x.split('-')[0]))
    
    
    final_counts = []
    potential_counts = []
    interval_labels = []
    
    for interval in intervals:
        data = stats_data[interval]
        final_counts.append(data['final_count'])
        potential_counts.append(data['potential_count'])
        interval_labels.append(interval)
    
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    x_pos = np.arange(len(intervals))
    width = 0.35
    
    if _has_chinese_font:
        bars1 = ax.bar(x_pos - width/2, final_counts, width, 
                       label='Output IoU ', color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x_pos + width/2, potential_counts, width,
                       label='Potential IoU ', color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('IoU Interval ', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sample Count ', fontsize=12, fontweight='bold')
        ax.set_title('Sample Count by IoU Interval (5% per interval) ', 
                     fontsize=14, fontweight='bold')
    else:
        bars1 = ax.bar(x_pos - width/2, final_counts, width, 
                       label='Output IoU', color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x_pos + width/2, potential_counts, width,
                       label='Potential IoU', color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('IoU Interval', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
        ax.set_title('Sample Count by IoU Interval (5% per interval)', fontsize=14, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(interval_labels, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='best', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    
    save_path = os.path.join(figure_dir, 'statistics_by_iou_interval.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"IoU interval statistics plot saved to : {save_path}")
    
    
    txt_path = os.path.join(figure_dir, 'statistics_by_iou_data.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Statistics by IoU Interval (5% per interval) \n")
        f.write("=" * 80 + "\n\n")
        total_final = sum(final_counts)
        total_potential = sum(potential_counts)
        f.write(f"Total Samples (Output IoU) / : {total_final}\n")
        f.write(f"Total Samples (Potential IoU) / : {total_potential}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        for interval in intervals:
            data = stats_data[interval]
            f.write(f"IoU Interval : {interval}\n")
            f.write(f"  Output IoU Count : {data['final_count']}\n")
            f.write(f"  Potential IoU Count : {data['potential_count']}\n")
            f.write("\n")
    
    print(f"IoU statistics data saved to : {txt_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Refer_RSVG inference script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    main(args)
