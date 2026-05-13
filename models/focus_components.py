

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from einops import rearrange


class ForegroundTokenSelector(nn.Module):
    
    def __init__(self, d_model: int = 256, num_feature_levels: int = 4, activation_type: str = 'relu'):
       
        super().__init__()
        self.d_model = d_model
        self.num_feature_levels = num_feature_levels
        
       
        from models.activation_utils import get_activation, GLUMLP
        
      
        if activation_type.lower() == 'glu':
            self.foreground_predictor = GLUMLP(d_model, d_model // 2, 1, num_layers=2)
        else:
            activation = get_activation(activation_type)
            self.foreground_predictor = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                activation,
                nn.Linear(d_model // 2, 1)
            )
        
        
        self.modulation_coeffs = nn.Parameter(torch.ones(num_feature_levels - 1))
        
    def forward(self, 
                srcs: List[torch.Tensor], 
                spatial_shapes: List[Tuple[int, int]],
                targets: Optional[List[dict]] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        
        foreground_scores = []
        foreground_labels = []
        
       
        high_level_scores = None
        prev_spatial_shape = None
        
        for lvl in range(self.num_feature_levels - 1, -1, -1):
            src = srcs[lvl]  # [B*T, C, H, W]
            B, C, H, W = src.shape
            current_spatial_shape = (H, W)
            
           
            src_flat = src.flatten(2).transpose(1, 2)  # [B*T, H*W, C]
            
           
            scores = self.foreground_predictor(src_flat)  # [B*T, H*W, 1]
            scores = scores.squeeze(-1)  # [B*T, H*W]
            
            
            if high_level_scores is not None and lvl < self.num_feature_levels - 1 and prev_spatial_shape is not None:
               
                H_prev, W_prev = prev_spatial_shape
             
                if high_level_scores.shape[1] == H_prev * W_prev:
                    high_scores_2d = high_level_scores.view(B, H_prev, W_prev)
                    high_scores_2d = F.interpolate(
                        high_scores_2d.unsqueeze(1), 
                        size=(H, W), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(1)
                    high_scores_flat = high_scores_2d.flatten(1)  # [B*T, H*W]
                    
                    
                    coeff_idx = (self.num_feature_levels - 2) - lvl
                    if 0 <= coeff_idx < len(self.modulation_coeffs):
                        modulation_coeff = self.modulation_coeffs[coeff_idx]
                        scores = scores + modulation_coeff * high_scores_flat
            
            scores = torch.sigmoid(scores)  # [B*T, H*W]
            foreground_scores.append(scores)
            high_level_scores = scores
            prev_spatial_shape = current_spatial_shape
            
            
            if targets is not None and self.training:
                labels = self._generate_foreground_labels(
                    scores, spatial_shapes[lvl], targets, lvl
                )
                foreground_labels.append(labels)
            else:
                foreground_labels.append(None)
        
        
        foreground_scores = foreground_scores[::-1]
        foreground_labels = foreground_labels[::-1]
        
        return foreground_scores, foreground_labels
    
    def _generate_foreground_labels(self, 
                                    scores: torch.Tensor,
                                    spatial_shape: Tuple[int, int],
                                    targets: List[dict],
                                    lvl: int) -> torch.Tensor:
       
        BT, HW = scores.shape
        H, W = spatial_shape
        labels = torch.zeros(BT, HW, dtype=torch.float32, device=scores.device)
        
       
        scale_factor = 2 ** (self.num_feature_levels - 1 - lvl)
        
        
        num_targets = len(targets)
        if num_targets > 0:
            T = BT // num_targets if num_targets > 0 else 1
            
            for b, target in enumerate(targets):
                if 'boxes' not in target:
                    continue
                boxes = target['boxes']  
                
                
                if boxes.dim() == 1:
                    
                    boxes = boxes.unsqueeze(0)  # [1, 4]
                
                if boxes.dim() != 2 or boxes.shape[1] != 4:
                    continue
                
                
                if boxes.shape[0] == T:
                    boxes_per_timestep = [boxes[t] for t in range(T)]
                else:
                   
                    boxes_per_timestep = [boxes] * T
                
               
                for t in range(T):
                    batch_idx = b * T + t
                    if batch_idx >= BT:
                        break
                    
                   
                    current_boxes = boxes_per_timestep[t]
                    if current_boxes.dim() == 1:
                        current_boxes = current_boxes.unsqueeze(0)
                  
                    for box_idx in range(current_boxes.shape[0]):
                        box = current_boxes[box_idx]  # [4]
                        cx, cy, bw, bh = box[0].item(), box[1].item(), box[2].item(), box[3].item()
                        
                      
                        cx_f = cx * W
                        cy_f = cy * H
                        bw_f = bw * W
                        bh_f = bh * H
                        
                       
                        x_min = max(0, int((cx_f - bw_f / 2) * 0.5))
                        x_max = min(W, int((cx_f + bw_f / 2) * 1.5))
                        y_min = max(0, int((cy_f - bh_f / 2) * 0.5))
                        y_max = min(H, int((cy_f + bh_f / 2) * 1.5))
                        
                       
                        for y in range(y_min, y_max):
                            for x in range(x_min, x_max):
                                idx = y * W + x
                                if idx < HW:
                                    labels[batch_idx, idx] = 1.0
        
        return labels


class MultiCategoryScorePredictor(nn.Module):
   
    def __init__(self, d_model: int = 256, num_classes: int = 91, activation_type: str = 'relu'):
        
        super().__init__()
        self.num_classes = num_classes
        
        
        from models.activation_utils import get_activation, GLUMLP
        
        
        if activation_type.lower() == 'glu':
           
            self.category_predictor = nn.Sequential(
                GLUMLP(d_model, d_model, d_model, num_layers=2),
                nn.Dropout(0.1),
                nn.Linear(d_model, num_classes)
            )
        else:
            activation = get_activation(activation_type)
            self.category_predictor = nn.Sequential(
                nn.Linear(d_model, d_model),
                activation,
                nn.Dropout(0.1),
                nn.Linear(d_model, num_classes)
            )
    
    def forward(self, src: torch.Tensor) -> torch.Tensor:
       
        category_scores = self.category_predictor(src)  # [B*T, H*W, num_classes]
        category_probs = torch.softmax(category_scores, dim=-1)  # [B*T, H*W, num_classes]
        return category_probs


class DynamicSemanticAwareTokenSelector(nn.Module):
   
    def __init__(self, 
                 d_model: int = 256, 
                 num_feature_levels: int = 4,
                 num_classes: int = 91,
                 activation_type: str = 'relu'):
       
        super().__init__()
        self.d_model = d_model
        self.num_feature_levels = num_feature_levels
        
        
        self.fts = ForegroundTokenSelector(d_model, num_feature_levels, activation_type=activation_type)
        
        
        self.category_predictor = MultiCategoryScorePredictor(d_model, num_classes, activation_type=activation_type)
        
        
        self.lang_guided_attn = nn.MultiheadAttention(
            d_model, num_heads=8, dropout=0.1, batch_first=False
        )
        
       
        from models.activation_utils import get_activation, GLUMLP
        
        
        if activation_type.lower() == 'glu':
            self.semantic_fusion = nn.Sequential(
                GLUMLP(d_model, d_model, d_model, num_layers=2),
                nn.Linear(d_model, 1)
            )
        else:
            activation = get_activation(activation_type)
            self.semantic_fusion = nn.Sequential(
                nn.Linear(d_model, d_model),
                activation,
                nn.Linear(d_model, 1)
            )
        
    def forward(self,
                srcs: List[torch.Tensor],
                text_features: torch.Tensor,
                spatial_shapes: List[Tuple[int, int]],
                text_masks: Optional[torch.Tensor] = None,
                targets: Optional[List[dict]] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        
        foreground_scores, foreground_labels = self.fts(srcs, spatial_shapes, targets)
        
        
        selected_tokens = []
        selected_indices = []
        dynamic_scores = []
        
        for lvl, (src, fg_scores) in enumerate(zip(srcs, foreground_scores)):
            B, C, H, W = src.shape
            src_flat = src.flatten(2).transpose(1, 2)  # [B*T, H*W, C]
            src_flat = src_flat.transpose(0, 1)  # [H*W, B*T, C] for attention
            
           
            HW, BT, C = src_flat.shape
            L, B, _ = text_features.shape
            T = BT // B
            
           
            src_flat_reshaped = src_flat.view(HW, B, T, C).mean(dim=2)  # [H*W, B, C]
            
            lang_relevance, _ = self.lang_guided_attn(
                query=text_features,  # [L, B, C]
                key=src_flat_reshaped,  # [H*W, B, C]
                value=src_flat_reshaped,  # [H*W, B, C]
                key_padding_mask=None
            )  # [L, B, C]
            
        
            lang_relevance = lang_relevance.mean(dim=0, keepdim=True)  # [1, B, C]
            lang_relevance = lang_relevance.expand(HW, -1, -1)  # [H*W, B, C]
            lang_relevance = lang_relevance.unsqueeze(2).expand(-1, -1, T, -1)  # [H*W, B, T, C]
            lang_relevance = lang_relevance.reshape(HW, BT, C)  # [H*W, B*T, C]
            
           
            semantic_scores = self.semantic_fusion(
                (src_flat + lang_relevance).transpose(0, 1)
            )  # [B*T, H*W, 1]
            semantic_scores = semantic_scores.squeeze(-1)  # [B*T, H*W]
            semantic_scores = torch.sigmoid(semantic_scores)
            
           
            dynamic_score = fg_scores * semantic_scores  # [B*T, H*W]
            dynamic_scores.append(dynamic_score)
            
            
            category_probs = self.category_predictor(src_flat.transpose(0, 1))  # [B*T, H*W, num_classes]
            max_category_prob = category_probs.max(dim=-1)[0]  # [B*T, H*W]
            
            
            fine_grained_score = dynamic_score * max_category_prob  # [B*T, H*W]
            
            
            retention_ratio = 0.5 + 0.3 * (lvl / (self.num_feature_levels - 1))  
            k = int(H * W * retention_ratio)
            k = max(1, min(k, H * W))
            
            _, top_indices = fine_grained_score.topk(k, dim=1)  # [B*T, k]
            
           
            BT = fine_grained_score.shape[0]
            batch_indices = torch.arange(BT, device=src.device).unsqueeze(1).expand(-1, k)
            selected_token = src_flat.transpose(0, 1)[batch_indices, top_indices]  # [B*T, k, C]
            selected_tokens.append(selected_token)
            selected_indices.append(top_indices)
        
        return selected_tokens, foreground_scores, foreground_labels, selected_indices


class SemanticEnhancedDualAttentionEncoderLayer(nn.Module):
   
    def __init__(self,
                 d_model: int = 256,
                 d_ffn: int = 1024,
                 dropout: float = 0.1,
                 activation: str = "relu",
                 n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        
       
        self.fine_grained_self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=False
        )
        
        
        self.lang_guided_cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=False
        )
        
       
        self.foreground_global_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=False
        )
        
       
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
       
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.dropout = nn.Dropout(dropout)
        self.norm4 = nn.LayerNorm(d_model)
        
    def forward(self,
                fine_grained_tokens: torch.Tensor,
                foreground_tokens: torch.Tensor,
                all_tokens: torch.Tensor,
                lang_features: torch.Tensor,
                fine_grained_pos: Optional[torch.Tensor] = None,
                foreground_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
      
        if fine_grained_pos is not None:
            q_fg = k_fg = fine_grained_tokens + fine_grained_pos
        else:
            q_fg = k_fg = fine_grained_tokens
        
        enhanced_fg = self.fine_grained_self_attn(
            q_fg, k_fg, fine_grained_tokens
        )[0]  # [N_fg, B, C]
        enhanced_fg = self.norm1(enhanced_fg + fine_grained_tokens)
        
        
        lang_enhanced_fg = self.lang_guided_cross_attn(
            query=lang_features,  # [1, B, C]
            key=enhanced_fg,  # [N_fg, B, C]
            value=enhanced_fg  # [N_fg, B, C]
        )[0]  # [1, B, C]
        
        
        lang_enhanced_fg_expanded = lang_enhanced_fg.expand(enhanced_fg.shape[0], -1, -1)
        enhanced_fg = enhanced_fg + lang_enhanced_fg_expanded
        enhanced_fg = self.norm2(enhanced_fg)
        
        
        if foreground_pos is not None:
            q_f = foreground_tokens + foreground_pos
        else:
            q_f = foreground_tokens
        
        enhanced_foreground = self.foreground_global_attn(
            query=q_f,  # [N_f, B, C]
            key=torch.cat([enhanced_fg, all_tokens], dim=0),  # [N_fg + N_all, B, C]
            value=torch.cat([enhanced_fg, all_tokens], dim=0)  # [N_fg + N_all, B, C]
        )[0]  # [N_f, B, C]
        enhanced_foreground = self.norm3(enhanced_foreground + foreground_tokens)
        
       
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(enhanced_foreground))))
        enhanced_foreground = self.norm4(enhanced_foreground + self.dropout(ffn_out))
        
        return enhanced_foreground

