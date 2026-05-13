import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .segmentation import (dice_loss, sigmoid_focal_loss)

from einops import rearrange

class SetCriterion(nn.Module):
    """ This class computes the loss for ReferFormer.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, focal_alpha=0.25, args=None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.focal_alpha = focal_alpha
        self.mask_out_stride = 4
        
       
        self.use_otsm = getattr(args, 'use_otsm', False) if args is not None else False
        self.otsm_loss_coef = getattr(args, 'otsm_loss_coef', 0.1) if args is not None else 0.1
       
        
       
        self.use_dsats = getattr(args, 'use_dsats', False) if args is not None else False
        self.fts_loss_coef = getattr(args, 'fts_loss_coef', 1.0) if args is not None else 1.0
       
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] 
        _, nf, nq = src_logits.shape[:3]
        src_logits = rearrange(src_logits, 'b t q k -> b (t q) k')

        # judge the valid frames
        valid_indices = []
        valids = [target['valid'] for target in targets]
        for valid, (indice_i, indice_j) in zip(valids, indices): 
            valid_ind = valid.nonzero().flatten() 
            valid_i = valid_ind * nq + indice_i
            valid_j = valid_ind + indice_j * nf
            valid_indices.append((valid_i, valid_j))

        idx = self._get_src_permutation_idx(valid_indices) # NOTE: use valid indices
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, valid_indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device) 
        if self.num_classes == 1: # binary referred
            target_classes[idx] = 0
        else:
            target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            pass
        return losses


    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes']  
        bs, nf, nq = src_boxes.shape[:3]
        src_boxes = src_boxes.transpose(1, 2)  

        idx = self._get_src_permutation_idx(indices)
        src_boxes = src_boxes[idx]  
        src_boxes = src_boxes.flatten(0, 1)  # [b*t, 4]

        target_boxes = torch.cat([t['boxes'] for t in targets], dim=0)  # [b*t, 4]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses


    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        # tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"] 
        src_masks = src_masks.transpose(1, 2) 

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets], 
                                                              size_divisibility=32, split=False).decompose()
        target_masks = target_masks.to(src_masks) 

        # downsample ground truth masks with ratio mask_out_stride
        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        
        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride] 
        assert target_masks.size(2) * self.mask_out_stride == im_h
        assert target_masks.size(3) * self.mask_out_stride == im_w

        src_masks = src_masks[src_idx] 
        # upsample predictions to the target size
        # src_masks = interpolate(src_masks, size=target_masks.shape[-2:], mode="bilinear", align_corners=False) 
        src_masks = src_masks.flatten(1) # [b, thw]

        target_masks = target_masks.flatten(1) # [b, thw]

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
       
        if self.use_otsm:
            loss_map['match'] = self.loss_match
       
        if self.use_dsats:
            loss_map['foreground_token_selector'] = self.loss_foreground_token_selector
      
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
       
        indices = self.matcher(outputs_without_aux, targets)

        
        target_valid = torch.stack([t["valid"] for t in targets], dim=0).reshape(-1) # [B, T] -> [B*T]
        num_boxes = target_valid.sum().item() 
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

       
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        
        if self.use_otsm and 'otsm_match_scores' in outputs:
            match_losses = self.loss_match(outputs, targets, indices, num_boxes)
            losses.update(match_losses)
       
        
      
        if self.use_dsats and 'foreground_token_selector' in self.losses:
            fts_losses = self.loss_foreground_token_selector(outputs, targets, indices, num_boxes)
            losses.update(fts_losses)
        

        return losses

    def loss_match(self, outputs, targets, indices, num_boxes):
       
        if 'otsm_match_scores' not in outputs:
            return {'loss_match': torch.tensor(0.0, device=next(iter(outputs.values())).device)}
        
        match_scores_dict = outputs['otsm_match_scores']
        pos_scores = match_scores_dict['pos_scores']  # [B]
        neg_scores = match_scores_dict.get('neg_scores', None)  # [B*N_neg] 或 None
        num_neg_per_sample = match_scores_dict.get('num_neg_per_sample', 0)
        
        B = pos_scores.shape[0]
        
        
        pos_labels = torch.ones(B, device=pos_scores.device)
        
       
        if neg_scores is not None and neg_scores.numel() > 0:
           
            neg_labels = torch.zeros(neg_scores.shape[0], device=neg_scores.device)
            
           
            all_scores = torch.cat([pos_scores, neg_scores], dim=0)  # [B + B*N_neg]
            all_labels = torch.cat([pos_labels, neg_labels], dim=0)  # [B + B*N_neg]
        else:
           
            all_scores = pos_scores
            all_labels = pos_labels
        
        
        loss_match = F.binary_cross_entropy(all_scores, all_labels, reduction='mean')
        
        return {'loss_match': loss_match}

    def loss_foreground_token_selector(self, outputs, targets, indices, num_boxes):
        
        if not self.use_dsats:
            return {}
        
       
        if 'foreground_scores' not in outputs:
            return {}
        
        foreground_scores_list = outputs['foreground_scores']  # List of [B*T, H*W]
        
        if foreground_scores_list is None or len(foreground_scores_list) == 0:
            
            device = next(iter(outputs.values())).device if outputs else torch.device('cuda')
            return {'loss_foreground_token_selector': torch.tensor(0.0, device=device, requires_grad=False)}
        
        losses = {}
        total_fts_loss = 0.0
        num_valid_levels = 0
        
        
        has_labels = any('foreground_labels' in target for target in targets)
        
        for lvl, fg_scores in enumerate(foreground_scores_list):
            # fg_scores: [B*T, H*W]
            BT, HW = fg_scores.shape
            B = len(targets)
            T = BT // B if B > 0 else 1
            
           
            label_tensors = []
            valid_samples = 0
            
            for b, target in enumerate(targets):
                if 'foreground_labels' not in target:
                  
                    label_tensors.append(torch.zeros(T, HW, device=fg_scores.device, dtype=fg_scores.dtype))
                    continue
                
                foreground_labels_list = target['foreground_labels']
                if foreground_labels_list is None or lvl >= len(foreground_labels_list) or foreground_labels_list[lvl] is None:
                    label_tensors.append(torch.zeros(T, HW, device=fg_scores.device, dtype=fg_scores.dtype))
                    continue
                
                sample_labels = foreground_labels_list[lvl] 
                
               
                if sample_labels.dim() == 1:
                   
                    if sample_labels.shape[0] == HW:
                        sample_labels = sample_labels.unsqueeze(0).expand(T, -1)
                    else:
                       
                        sample_labels = torch.zeros(T, HW, device=fg_scores.device, dtype=fg_scores.dtype)
                elif sample_labels.dim() == 2:
                  
                    if sample_labels.shape[1] != HW:
                        
                        sample_labels = torch.zeros(T, HW, device=fg_scores.device, dtype=fg_scores.dtype)
                    elif sample_labels.shape[0] == 1:
                        
                        sample_labels = sample_labels.expand(T, -1)
                    elif sample_labels.shape[0] != T:
                        
                        if sample_labels.shape[0] > T:
                            sample_labels = sample_labels[:T]
                        else:
                           
                            padding = torch.zeros(T - sample_labels.shape[0], HW, 
                                                 device=fg_scores.device, dtype=fg_scores.dtype)
                            sample_labels = torch.cat([sample_labels, padding], dim=0)
                else:
                   
                    sample_labels = torch.zeros(T, HW, device=fg_scores.device, dtype=fg_scores.dtype)
                
                label_tensors.append(sample_labels)
                valid_samples += 1
            
           
            
           
            fg_labels = torch.cat(label_tensors, dim=0)  # [B*T, H*W]
            
           
            if fg_labels.shape != fg_scores.shape:
               
                if fg_labels.numel() == fg_scores.numel():
                    fg_labels = fg_labels.view(fg_scores.shape)
                else:
                   
                    continue
            
          
            probs = fg_scores.flatten()  # [B*T*H*W]
            labels = fg_labels.flatten()  # [B*T*H*W]
            
           
            alpha = self.focal_alpha
            gamma = 2.0
            
           
            p_t = probs * labels + (1 - probs) * (1 - labels)  
            alpha_t = alpha * labels + (1 - alpha) * (1 - labels)  
            focal_weight = alpha_t * (1 - p_t) ** gamma
            
            
            bce = F.binary_cross_entropy(probs, labels, reduction='none')
            
           
            focal_loss = focal_weight * bce
            
           
            level_loss = focal_loss.mean()
            total_fts_loss += level_loss
            num_valid_levels += 1
        
        
        if num_valid_levels > 0:
            avg_fts_loss = total_fts_loss / num_valid_levels
            losses['loss_foreground_token_selector'] = avg_fts_loss
        else:
            
            losses['loss_foreground_token_selector'] = torch.tensor(0.0, device=foreground_scores_list[0].device, requires_grad=False)
        
        return losses


