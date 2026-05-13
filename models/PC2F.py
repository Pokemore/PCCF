import torch
import torch.nn.functional as F
from torch import nn

import os
import math
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       nested_tensor_from_videos_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .position_encoding import PositionEmbeddingSine1D
from .backbone import build_backbone
from .deformable_transformer import build_deforamble_transformer
from .segmentation import VisionLanguageFusionModule
from .matcher import build_matcher
from .criterion import SetCriterion
from .postprocessors import build_postprocessors
from .focus_components import DynamicSemanticAwareTokenSelector, SemanticEnhancedDualAttentionEncoderLayer

from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizerFast

import copy
from einops import rearrange, repeat


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


os.environ["TOKENIZERS_PARALLELISM"] = "false"  


class PC2F(nn.Module):

    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 num_frames, aux_loss=False, with_box_refine=False, two_stage=False,
                 freeze_text_encoder=False, args=None):

        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides[-3:])
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[-3:][_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):  # downsample 2x
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-3:][0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.num_frames = num_frames
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        assert two_stage == False, "args.two_stage must be false!"

        # initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

       
        self.tokenizer = RobertaTokenizerFast.from_pretrained('/root/Documents/PreTrained/RoBERTa')
        
        # self.text_encoder = RobertaModel.from_pretrained('/root/Documents/PreTrained/RoBERTa', use_safetensors=False)
        self.text_encoder = RobertaModel.from_pretrained('/root/Documents/PreTrained/RoBERTa')
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

       
        self.resizer = FeatureResizer(
            input_feat_size=768,
            output_feat_size=hidden_dim,
            dropout=0.1,
        )

        self.fusion_module = VisionLanguageFusionModule(d_model=hidden_dim, nhead=8)
        self.fusion_module_text = VisionLanguageFusionModule(d_model=hidden_dim, nhead=8)

        self.text_pos = PositionEmbeddingSine1D(hidden_dim, normalize=True)
        self.poolout_module = RobertaPoolout(d_model=hidden_dim)
        
        
        use_cn = getattr(args, 'use_cn', False) if args is not None else False
        if use_cn:
            self.use_conditional_norm = True
            self.cn_insertion_mode = getattr(args, 'cn_insertion_mode', 'after_input_proj') if args is not None else 'after_input_proj'
            self.cn_use_residual = getattr(args, 'cn_use_residual', True) if args is not None else True
            _cn_gamma = 'relu'
            _cn_beta = 'tanh'
            _cn_f_cond_refine = True
        else:
            self.use_conditional_norm = getattr(args, 'use_conditional_norm', False) if args is not None else False
            self.cn_insertion_mode = getattr(args, 'cn_insertion_mode', 'after_input_proj') if args is not None else 'after_input_proj'
            self.cn_use_residual = getattr(args, 'cn_use_residual', False) if args is not None else False
            _cn_gamma = getattr(args, 'cn_gamma_activation', 'none') if args is not None else 'none'
            _cn_beta = getattr(args, 'cn_beta_activation', 'none') if args is not None else 'none'
            _cn_f_cond_refine = getattr(args, 'cn_f_cond_refine', False) if args is not None else False
        
        if self.use_conditional_norm:
            activation_type = getattr(args, 'activation_type', 'relu') if args is not None else 'relu'
            if getattr(args, 'use_adaptive_cn_f3_fusion', False) and args is not None:
                activation_type = 'glu'
            cn_gamma_activation = _cn_gamma
            cn_beta_activation = _cn_beta
            cn_f_cond_refine = _cn_f_cond_refine
            cn_f_cond_refine_dim = getattr(args, 'cn_f_cond_refine_dim', None) if args is not None else None
            
           
            self.text_condition_generator = TextConditionGenerator(
                hidden_dim,
                activation_type=activation_type,
                use_f_cond_refine=cn_f_cond_refine,
                f_cond_refine_dim=cn_f_cond_refine_dim
            )
            
            if self.cn_insertion_mode == 'after_input_proj':
              
                num_cn_layers = num_feature_levels
                self.conditional_norm_layers = nn.ModuleList([
                    ConditionalNormLayer(
                        visual_feat_dim=hidden_dim, 
                        f_cond_dim=hidden_dim,
                        use_residual=self.cn_use_residual,
                        activation_type=activation_type,
                        gamma_activation=cn_gamma_activation,
                        beta_activation=cn_beta_activation
                    ) 
                    for _ in range(num_cn_layers)
                ])
            elif self.cn_insertion_mode == 'in_backbone':
                
                activation_type = getattr(args, 'activation_type', 'relu') if args is not None else 'relu'
                if getattr(args, 'use_adaptive_cn_f3_fusion', False) and args is not None:
                    activation_type = 'glu'
               
                self.conditional_norm_layers_backbone = nn.ModuleList([
                    ConditionalNormLayer(
                        visual_feat_dim=backbone.num_channels[-3:][i],
                        f_cond_dim=hidden_dim,  
                        use_residual=self.cn_use_residual,
                        activation_type=activation_type,
                        gamma_activation=cn_gamma_activation,
                        beta_activation=cn_beta_activation
                    ) 
                    for i in range(num_backbone_stages)
                ])
            else:
                raise ValueError(f"Unsupported cn_insertion_mode: {self.cn_insertion_mode}")
        
        self.use_adaptive_cn_f3_fusion = getattr(args, 'use_adaptive_cn_f3_fusion', False) if args is not None else False
        
        if self.use_adaptive_cn_f3_fusion:
            
            use_solution_f3 = getattr(args, 'use_solution_f3', False) if args is not None else False
            if not (self.use_conditional_norm and use_solution_f3):
                raise ValueError("use_adaptive_cn_f3_fusion requires both use_conditional_norm and use_solution_f3 to be enabled")
            
           
            self.adaptive_cn_f3_fusion = AdaptiveCNF3Fusion(
                hidden_dim=hidden_dim,
                use_residual=self.cn_use_residual
            )
       
        
        self.use_otsm = getattr(args, 'use_otsm', False) if args is not None else False
        self.otsm_neg_sample_method = getattr(args, 'otsm_neg_sample_method', 'iou_distribution') if args is not None else 'iou_distribution'
        self.num_neg_samples = getattr(args, 'num_neg_samples', 1) if args is not None else 1
        
        if self.use_otsm:
            
            self.matching_head = MatchingHead(hidden_dim, nhead=8, dropout=0.1)
            
            
            self.roi_feature_extractor = RoIFeatureExtractor(hidden_dim, output_size=7)
        
        self.use_dsats = getattr(args, 'use_dsats', False) if args is not None else False
       
        self.use_solution_f3 = getattr(args, 'use_solution_f3', False) if args else False
        
        if self.use_dsats:
            
            activation_type = getattr(args, 'activation_type', 'relu') if args is not None else 'relu'
            if getattr(args, 'use_adaptive_cn_f3_fusion', False) and args is not None:
                activation_type = 'glu'
            self.dsats = DynamicSemanticAwareTokenSelector(
                d_model=hidden_dim,
                num_feature_levels=num_feature_levels,
                num_classes=num_classes,
                activation_type=activation_type
            )
            
           
            if self.use_solution_f3:
                if not self.use_dsats:
                    raise ValueError("--use_solution_f3 requires --use_dsats to be enabled")
        
        self.use_sedae = getattr(args, 'use_sedae', False) if args is not None else False
        if self.use_sedae:
            
            pass
        

    def forward(self, samples: NestedTensor, captions, targets):

       
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list(samples)

       
        features, pos = self.backbone(samples)

        b = len(captions)
        t = pos[0].shape[0] // b

        if 'valid_indices' in targets[0]:
            valid_indices = torch.tensor([i * t + target['valid_indices'] for i, target in enumerate(targets)]).to(
                pos[0].device)
            for feature in features:
                feature.tensors = feature.tensors.index_select(0, valid_indices)
                feature.mask = feature.mask.index_select(0, valid_indices)
            for i, p in enumerate(pos):
                pos[i] = p.index_select(0, valid_indices)
            samples.mask = samples.mask.index_select(0, valid_indices)
            
            t = 1

        text_features = self.forward_text(captions, device=pos[0].device)

        
        if self.use_conditional_norm:
            
            text_word_features_for_cond, _ = text_features.decompose()
            text_word_features_for_cond = text_word_features_for_cond.permute(1, 0, 2)  # [length, batch_size, hidden_dim]
            f_cond = self.text_condition_generator(text_word_features_for_cond)  # [batch_size, hidden_dim]
        else:
            f_cond = None
        
        srcs = []
        masks = []
        poses = []

        text_pos = self.text_pos(text_features).permute(2, 0, 1)  # [length, batch_size, c]
        text_word_features, text_word_masks = text_features.decompose()

        text_word_features = text_word_features.permute(1, 0, 2)  # [length, batch_size, c]
        text_word_initial_features = text_word_features

        
        for l, (feat, pos_l) in enumerate(zip(features[-3:], pos[-3:])):
            src, mask = feat.decompose()
            
            
            if self.use_conditional_norm and self.cn_insertion_mode == 'in_backbone':
               
                src = self.conditional_norm_layers_backbone[l](src, f_cond)
           
            
            src_proj_l = self.input_proj[l](src)
            n, c, h, w = src_proj_l.shape

           
            if self.use_conditional_norm and self.cn_insertion_mode == 'after_input_proj':
               
                src_proj_l = self.conditional_norm_layers[l](src_proj_l, f_cond)
           

            
            src_proj_l = rearrange(src_proj_l, '(b t) c h w -> (t h w) b c', b=b, t=t)
            mask = rearrange(mask, '(b t) h w -> b (t h w)', b=b, t=t)
            pos_l = rearrange(pos_l, '(b t) c h w -> (t h w) b c', b=b, t=t)
            text_word_features = self.fusion_module_text(tgt=text_word_features,
                                                         memory=src_proj_l,
                                                         memory_key_padding_mask=mask,
                                                         pos=pos_l,
                                                         query_pos=None)

            src_proj_l = self.fusion_module(tgt=src_proj_l,
                                            memory=text_word_initial_features,
                                            memory_key_padding_mask=text_word_masks,
                                            pos=text_pos,
                                            query_pos=None)
            src_proj_l = rearrange(src_proj_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
            mask = rearrange(mask, 'b (t h w) -> (b t) h w', t=t, h=h, w=w)
            pos_l = rearrange(pos_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)

            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos_l)
            assert mask is not None

        if self.num_feature_levels > (len(features) - 1):
            _len_srcs = len(features) - 1  
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                
                
                if self.use_conditional_norm and self.cn_insertion_mode == 'after_input_proj':
                    src = self.conditional_norm_layers[l](src, f_cond)
               
                
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                n, c, h, w = src.shape

               
                src = rearrange(src, '(b t) c h w -> (t h w) b c', b=b, t=t)
                mask = rearrange(mask, '(b t) h w -> b (t h w)', b=b, t=t)
                pos_l = rearrange(pos_l, '(b t) c h w -> (t h w) b c', b=b, t=t)

                text_word_features = self.fusion_module_text(tgt=text_word_features,
                                                             memory=src,
                                                             memory_key_padding_mask=mask,
                                                             pos=pos_l,
                                                             query_pos=None)
                src = self.fusion_module(tgt=src,
                                         memory=text_word_initial_features,
                                         memory_key_padding_mask=text_word_masks,
                                         pos=text_pos,
                                         query_pos=None
                                         )
                src = rearrange(src, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
                mask = rearrange(mask, 'b (t h w) -> (b t) h w', t=t, h=h, w=w)
                pos_l = rearrange(pos_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)

                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)

        
        foreground_labels_list = None
        foreground_scores_list = None
        selected_tokens = None
        foreground_scores = None
        selected_indices = None
        if self.use_dsats and hasattr(self, 'dsats'):
           
            spatial_shapes = []
            for src in srcs:
                _, _, h, w = src.shape
                spatial_shapes.append((h, w))
            
           
            text_word_features_for_dsats = text_word_initial_features  # [L, B, C]
            
           
            selected_tokens, foreground_scores, foreground_labels_list, selected_indices = self.dsats(
                srcs=srcs,
                text_features=text_word_features_for_dsats,
                spatial_shapes=spatial_shapes,
                text_masks=text_word_masks,
                targets=targets if self.training else None
            )
            
            
            foreground_scores_list = foreground_scores
            
           
            if foreground_labels_list is not None:
                
                B = len(targets)
                if B > 0 and foreground_labels_list[0] is not None:
                    BT = foreground_labels_list[0].shape[0]
                    T = BT // B if B > 0 else 1
                    
                    for b, target in enumerate(targets):
                       
                        sample_labels_list = []
                        for fg_labels in foreground_labels_list:
                            if fg_labels is not None:
                                
                                batch_start = b * T
                                batch_end = (b + 1) * T
                                sample_labels = fg_labels[batch_start:batch_end]  # [T, H*W]
                                sample_labels_list.append(sample_labels)
                            else:
                                sample_labels_list.append(None)
                        target['foreground_labels'] = sample_labels_list
        

        text_word_features = rearrange(text_word_features, 'l b c -> b l c')
        text_sentence_features = self.poolout_module(text_word_features)

        
        query_embeds = self.query_embed.weight  # [num_queries, c]
        text_embed = repeat(text_sentence_features, 'b c -> b t q c', t=t, q=self.num_queries)
        
       
        solution_f3_kwargs = {}
        if self.use_solution_f3 and self.use_dsats and hasattr(self, 'dsats'):
           
            solution_f3_fine_grained_mask = None
            solution_f3_foreground_mask = None
            solution_f3_lang_features = None
            
            if selected_tokens is not None and foreground_scores is not None:
               
                BT = srcs[0].shape[0]
                fine_grained_mask_list = []
                foreground_mask_list = []
                
                for lvl, (src, fg_scores) in enumerate(zip(srcs, foreground_scores)):
                    B, C, H, W = src.shape
                    HW = H * W
                    
                   
                    fg_mask = torch.zeros(BT, HW, dtype=torch.bool, device=src.device)
                    if selected_indices is not None and lvl < len(selected_indices) and selected_indices[lvl] is not None:
                        top_indices = selected_indices[lvl]  # [B*T, k]
                        
                        if top_indices.shape[0] == BT and top_indices.numel() > 0:
                            
                            top_indices = top_indices.clamp(min=0, max=HW - 1)
                            batch_indices = torch.arange(BT, device=src.device).unsqueeze(1).expand(-1, top_indices.shape[1])
                            fg_mask[batch_indices, top_indices] = True
                    fine_grained_mask_list.append(fg_mask)
                    
                   
                    foreground_threshold = 0.5  
                    fg_mask = fg_scores > foreground_threshold
                    foreground_mask_list.append(fg_mask)
                
                
                solution_f3_fine_grained_mask = torch.cat(fine_grained_mask_list, dim=1)  # [B*T, Σ(H*W)]
                solution_f3_foreground_mask = torch.cat(foreground_mask_list, dim=1)  # [B*T, Σ(H*W)]
            
            
            if text_sentence_features is not None:
               
                solution_f3_lang_features = text_sentence_features.unsqueeze(0)  # [1, B, C]
            
            solution_f3_kwargs = {
                'solution_f3_lang_features': solution_f3_lang_features,
                'solution_f3_fine_grained_mask': solution_f3_fine_grained_mask,
                'solution_f3_foreground_mask': solution_f3_foreground_mask
            }
        
        
        hs, memory_features, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, inter_samples = \
            self.transformer(srcs, text_embed, masks, poses, query_embeds, **solution_f3_kwargs)

        
        if self.use_adaptive_cn_f3_fusion and f_cond is not None:
           
            memory_features = self.adaptive_cn_f3_fusion(memory_features, f_cond)
      

        out = {}
       
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()  # cxcywh, range in [0,1]
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
       
        l, bt, q, k = outputs_class.shape
        
        if bt % t == 0:
            b_actual = bt // t
            t_actual = t
        else:
            
            b_actual = bt
            t_actual = 1
        
        outputs_class = rearrange(outputs_class, 'l (b t) q k -> l b t q k', b=b_actual, t=t_actual)
        outputs_coord = rearrange(outputs_coord, 'l (b t) q n -> l b t q n', b=b_actual, t=t_actual)
        out['pred_logits'] = outputs_class[-1]  
        out['pred_boxes'] = outputs_coord[-1]  

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

       
        if self.use_dsats and foreground_scores_list is not None:
            out['foreground_scores'] = foreground_scores_list
        

      
        if self.use_otsm:
            
            pos_boxes = []
            image_sizes = []
            for target in targets:
                if 'boxes' in target:
                   
                    boxes = target['boxes']
                    if boxes.dim() == 2 and boxes.shape[0] > 1:
                        
                        pos_box = boxes[0].cpu().numpy().tolist()  
                    else:
                        if boxes.dim() == 2:
                            pos_box = boxes[0].cpu().numpy().tolist()
                        else:
                            pos_box = boxes.cpu().numpy().tolist()
                    pos_boxes.append(pos_box)
                    
                   
                    if 'size' in target:
                        size = target['size']
                        if isinstance(size, torch.Tensor):
                            H, W = size.cpu().numpy().tolist()
                        else:
                            H, W = size
                    else:
                       
                        H, W = samples.tensors.shape[-2:]
                    image_sizes.append((H, W))
            
            if len(pos_boxes) > 0:
                
                neg_boxes_list = []
                for pos_box, image_size in zip(pos_boxes, image_sizes):
                    neg_boxes = generate_negative_samples(
                        pos_box, image_size,
                        method=self.otsm_neg_sample_method,
                        num_neg=self.num_neg_samples,
                        shift_range=0.1,
                        scale_range=0.1,
                        target_iou=0.35,
                        iou_tolerance=0.05,
                        size_variance=0.2
                    )
                    neg_boxes_list.append(neg_boxes)
                
              
                pos_features_list = []
                for i, (pos_box, image_size) in enumerate(zip(pos_boxes, image_sizes)):
                    pos_feat = self.roi_feature_extractor(
                        srcs, [pos_box], image_size
                    )  # [1, C]
                    pos_features_list.append(pos_feat)
                pos_features = torch.cat(pos_features_list, dim=0)  # [B, C]
                
               
                neg_features_list = []
                for i, (neg_boxes, image_size) in enumerate(zip(neg_boxes_list, image_sizes)):
                    for neg_box in neg_boxes:
                        neg_feat = self.roi_feature_extractor(
                            srcs, [neg_box], image_size
                        )  # [1, C]
                        neg_features_list.append(neg_feat)
                neg_features = torch.cat(neg_features_list, dim=0) if len(neg_features_list) > 0 else None  # [B*N_neg, C]
                
               
                pos_match_scores = self.matching_head(pos_features, text_sentence_features)  # [B]
                if neg_features is not None:
                    neg_match_scores = self.matching_head(neg_features, text_sentence_features)  # [B*N_neg]
                else:
                    neg_match_scores = None
                
                
                out['otsm_match_scores'] = {
                    'pos_scores': pos_match_scores,  # [B]
                    'neg_scores': neg_match_scores,  # [B*N_neg] 或 None
                    'num_neg_per_sample': self.num_neg_samples
                }
        # ================================================================

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        
        return [{"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def forward_text(self, captions, device):
        if isinstance(captions[0], str):
            tokenized = self.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt").to(device)
            encoded_text = self.text_encoder(**tokenized)
            text_attention_mask = tokenized.attention_mask.ne(1).bool()

            text_features = encoded_text.last_hidden_state
            text_features = self.resizer(text_features)
            text_masks = text_attention_mask
            text_features = NestedTensor(text_features, text_masks)  # NestedTensor
        else:
            raise ValueError("Please mask sure the caption is a list of string")
        return text_features


class MLP(nn.Module):
    

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class RobertaPoolout(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output



def compute_iou(box1, box2):
    """计算两个边界框的 IoU"""
    import numpy as np
    
    
    if len(box1) == 4:
        if box1[2] <= box1[0]:  
            cx1, cy1, w1, h1 = box1
            x1_1 = cx1 - w1 / 2
            y1_1 = cy1 - h1 / 2
            x2_1 = cx1 + w1 / 2
            y2_1 = cy1 + h1 / 2
        else:  
            x1_1, y1_1, x2_1, y2_1 = box1
    else:
        raise ValueError(f"Unsupported box format: {len(box1)} values")
    
    if len(box2) == 4:
        if box2[2] <= box2[0]: 
            cx2, cy2, w2, h2 = box2
            x1_2 = cx2 - w2 / 2
            y1_2 = cy2 - h2 / 2
            x2_2 = cx2 + w2 / 2
            y2_2 = cy2 + h2 / 2
        else:  
            x1_2, y1_2, x2_2, y2_2 = box2
    else:
        raise ValueError(f"Unsupported box format: {len(box2)} values")
    
   
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    return iou


def apply_affine_transform_to_box(box, shift_range=0.1, scale_range=0.1, image_size=(640, 640)):
    
    import random
    import numpy as np
    
    
    if len(box) == 4:
        if box[2] > box[0]:  
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
        else: 
            cx, cy, w, h = box
           
            H, W = image_size
            cx = cx * W
            cy = cy * H
            w = w * W
            h = h * H
    else:
        raise ValueError(f"Unsupported box format: {len(box)} values")
    
    H, W = image_size
    
   
    shift_x = w * shift_range * (2 * random.random() - 1)
    shift_y = h * shift_range * (2 * random.random() - 1)
    cx_new = cx + shift_x
    cy_new = cy + shift_y
    
    
    scale = 1 + scale_range * (2 * random.random() - 1)
    w_new = w * scale
    h_new = h * scale
    
    
    aspect_ratio_change = 1 + 0.1 * (2 * random.random() - 1)
    w_new = w_new * aspect_ratio_change
    h_new = h_new / aspect_ratio_change
    
   
    cx_new = cx_new / W
    cy_new = cy_new / H
    w_new = w_new / W
    h_new = h_new / H
    
    
    cx_new = max(0, min(cx_new, 1))
    cy_new = max(0, min(cy_new, 1))
    w_new = max(0.01, min(w_new, 1))
    h_new = max(0.01, min(h_new, 1))
    
    return [cx_new, cy_new, w_new, h_new]


def generate_negative_box_by_iou_distribution(pos_box, image_size, target_iou_range=(0.2, 0.5),
                                               size_variance=0.2, max_attempts=100):
   
    import random
    import numpy as np
    
    
    if len(pos_box) == 4:
        if pos_box[2] > pos_box[0]:  
            x1_pos, y1_pos, x2_pos, y2_pos = pos_box
            cx_pos = (x1_pos + x2_pos) / 2
            cy_pos = (y1_pos + y2_pos) / 2
            w_pos = x2_pos - x1_pos
            h_pos = y2_pos - y1_pos
        else:  
            cx_pos, cy_pos, w_pos, h_pos = pos_box
    else:
        raise ValueError(f"Unsupported box format: {len(pos_box)} values")
    
    min_iou, max_iou = target_iou_range
    
    for attempt in range(max_attempts):
        
        std_x = max(abs(w_pos) * 0.3, 0.01)
        std_y = max(abs(h_pos) * 0.3, 0.01)
        offset_x = np.random.normal(0, std_x)
        offset_y = np.random.normal(0, std_y)
        cx_neg = cx_pos + offset_x
        cy_neg = cy_pos + offset_y
        
        
        w_scale = np.random.lognormal(0, size_variance)
        h_scale = np.random.lognormal(0, size_variance)
        w_neg = w_pos * w_scale
        h_neg = h_pos * h_scale
        
        
        w_neg = np.clip(w_neg, w_pos * (1 - size_variance), w_pos * (1 + size_variance))
        h_neg = np.clip(h_neg, h_pos * (1 - size_variance), h_pos * (1 + size_variance))
        
        
        cx_neg = max(0, min(cx_neg, 1))
        cy_neg = max(0, min(cy_neg, 1))
        w_neg = max(0.01, min(w_neg, 1))
        h_neg = max(0.01, min(h_neg, 1))
        
        neg_box = [cx_neg, cy_neg, w_neg, h_neg]
        
        
        iou = compute_iou(pos_box, neg_box)
        
       
        if min_iou <= iou <= max_iou:
            return neg_box
    
    
    return apply_affine_transform_to_box(pos_box, shift_range=0.1, scale_range=0.1, image_size=image_size)


def generate_negative_box_by_iterative_refinement(pos_box, image_size, target_iou=0.35,
                                                   iou_tolerance=0.05, max_iterations=20):
   
    import numpy as np
    
    
    if len(pos_box) == 4:
        if pos_box[2] > pos_box[0]:  # [x1, y1, x2, y2]
            x1_pos, y1_pos, x2_pos, y2_pos = pos_box
            cx_pos = (x1_pos + x2_pos) / 2
            cy_pos = (y1_pos + y2_pos) / 2
            w_pos = x2_pos - x1_pos
            h_pos = y2_pos - y1_pos
        else:  
            cx_pos, cy_pos, w_pos, h_pos = pos_box
    else:
        raise ValueError(f"Unsupported box format: {len(pos_box)} values")
    
   
    std_x = max(abs(w_pos) * 0.2, 0.01)
    std_y = max(abs(h_pos) * 0.2, 0.01)
    cx_neg = cx_pos + np.random.normal(0, std_x)
    cy_neg = cy_pos + np.random.normal(0, std_y)
    w_neg = w_pos * np.random.uniform(0.8, 1.2)
    h_neg = h_pos * np.random.uniform(0.8, 1.2)
    
    
    for iteration in range(max_iterations):
        
        cx_neg = max(0, min(cx_neg, 1))
        cy_neg = max(0, min(cy_neg, 1))
        w_neg = max(0.01, min(w_neg, 1))
        h_neg = max(0.01, min(h_neg, 1))
        
        neg_box = [cx_neg, cy_neg, w_neg, h_neg]
        current_iou = compute_iou(pos_box, neg_box)
        
        
        if abs(current_iou - target_iou) <= iou_tolerance:
            return neg_box
        
        
        iou_error = current_iou - target_iou
        
        
        if iou_error > 0:
            direction = np.array([cx_neg - cx_pos, cy_neg - cy_pos])
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                cx_neg += direction[0] * w_pos * 0.05
                cy_neg += direction[1] * h_pos * 0.05
            w_neg *= 0.98
            h_neg *= 0.98
        else:
            
            direction = np.array([cx_neg - cx_pos, cy_neg - cy_pos])
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                cx_neg -= direction[0] * w_pos * 0.05
                cy_neg -= direction[1] * h_pos * 0.05
            w_neg *= 1.02
            h_neg *= 1.02
        
        
        cx_neg = np.clip(cx_neg, w_neg/2, 1 - w_neg/2)
        cy_neg = np.clip(cy_neg, h_neg/2, 1 - h_neg/2)
        w_neg = np.clip(w_neg, w_pos * 0.5, w_pos * 1.5)
        h_neg = np.clip(h_neg, h_pos * 0.5, h_pos * 1.5)
    
   
    cx_neg = max(0, min(cx_neg, 1))
    cy_neg = max(0, min(cy_neg, 1))
    w_neg = max(0.01, min(w_neg, 1))
    h_neg = max(0.01, min(h_neg, 1))
    
    return [cx_neg, cy_neg, w_neg, h_neg]


def generate_negative_box_guaranteed(pos_box, image_size, target_iou=0.35, iou_tolerance=0.05,
                                     size_variance=0.2, max_attempts=50):
   
    
    min_iou = target_iou - iou_tolerance
    max_iou = target_iou + iou_tolerance
    neg_box = generate_negative_box_by_iou_distribution(
        pos_box, image_size, target_iou_range=(min_iou, max_iou),
        size_variance=size_variance, max_attempts=max_attempts
    )
    
   
    actual_iou = compute_iou(pos_box, neg_box)
    if min_iou <= actual_iou <= max_iou:
        return neg_box
    
    
    return generate_negative_box_by_iterative_refinement(
        pos_box, image_size, target_iou, iou_tolerance
    )


def generate_negative_samples(pos_box, image_size, method='iou_distribution', num_neg=1, **kwargs):
   
    neg_boxes = []
    
    if method == 'random_perturb':
        shift_range = kwargs.get('shift_range', 0.1)
        scale_range = kwargs.get('scale_range', 0.1)
        
        for _ in range(num_neg):
            neg_box = apply_affine_transform_to_box(
                pos_box, shift_range=shift_range, scale_range=scale_range, image_size=image_size
            )
            neg_boxes.append(neg_box)
    
    elif method == 'iou_distribution':
        
        if num_neg == 1:
            target_ious = [0.35]
        elif num_neg == 2:
            target_ious = [0.3, 0.4]
        elif num_neg == 3:
            target_ious = [0.25, 0.35, 0.45]
        else:
            
            target_ious = [0.25 + 0.2 * i / (num_neg - 1) for i in range(num_neg)]
        
        iou_tolerance = kwargs.get('iou_tolerance', 0.05)
        size_variance = kwargs.get('size_variance', 0.2)
        
        for target_iou in target_ious:
            neg_box = generate_negative_box_guaranteed(
                pos_box, image_size, target_iou=target_iou,
                iou_tolerance=iou_tolerance, size_variance=size_variance
            )
            neg_boxes.append(neg_box)
    
    else:
        raise ValueError(f"Unsupported negative sample generation method: {method}")
    
    return neg_boxes
# ============================================


class ConditionalNormLayer(nn.Module):
   
    def __init__(self, visual_feat_dim, f_cond_dim=None, eps=1e-5, use_residual=False,
                 activation_type='relu', gamma_activation='none', beta_activation='none'):
       
        super().__init__()
        self.visual_feat_dim = visual_feat_dim
        self.f_cond_dim = f_cond_dim if f_cond_dim is not None else visual_feat_dim
        self.eps = eps
        self.use_residual = use_residual
        self.activation_type = activation_type
        self.gamma_activation = gamma_activation
        self.beta_activation = beta_activation
        
        
        if self.f_cond_dim != self.visual_feat_dim:
            self.f_cond_proj = nn.Linear(self.f_cond_dim, self.visual_feat_dim)
        else:
            self.f_cond_proj = None
        
        
        from models.activation_utils import get_activation, build_mlp_with_activation
        
        
        if activation_type.lower() == 'glu':
            from models.activation_utils import GLUMLP
            self.gamma_mlp = GLUMLP(self.visual_feat_dim, self.visual_feat_dim, self.visual_feat_dim, num_layers=2)
            self.beta_mlp = GLUMLP(self.visual_feat_dim, self.visual_feat_dim, self.visual_feat_dim, num_layers=2)
        else:
            activation = get_activation(activation_type)
            self.gamma_mlp = nn.Sequential(
                nn.Linear(self.visual_feat_dim, self.visual_feat_dim),
                activation,
                nn.Linear(self.visual_feat_dim, self.visual_feat_dim)
            )
            self.beta_mlp = nn.Sequential(
                nn.Linear(self.visual_feat_dim, self.visual_feat_dim),
                activation,
                nn.Linear(self.visual_feat_dim, self.visual_feat_dim)
            )
        
        
        if gamma_activation.lower() == 'relu':
            self.gamma_output_activation = nn.ReLU()
        elif gamma_activation.lower() == 'softplus':
            self.gamma_output_activation = nn.Softplus()
        else:
            self.gamma_output_activation = None
        
        if beta_activation.lower() == 'tanh':
            self.beta_output_activation = nn.Tanh()
        else:
            self.beta_output_activation = None
        
        
        if activation_type.lower() != 'glu':
            nn.init.constant_(self.gamma_mlp[-1].weight, 0)
            nn.init.constant_(self.gamma_mlp[-1].bias, 1)
            nn.init.constant_(self.beta_mlp[-1].weight, 0)
            nn.init.constant_(self.beta_mlp[-1].bias, 0)
        else:
            
            nn.init.constant_(self.gamma_mlp.layers[-1].weight, 0)
            nn.init.constant_(self.gamma_mlp.layers[-1].bias, 1)
            nn.init.constant_(self.beta_mlp.layers[-1].weight, 0)
            nn.init.constant_(self.beta_mlp.layers[-1].bias, 0)
        
        
        if self.use_residual:
            
            self.cn_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, visual_feat, f_cond):
       
        
        if visual_feat.dim() == 4:  # [B*T, C, H, W]
            B_T, C_vis, H, W = visual_feat.shape
        elif visual_feat.dim() == 2:  # [B*T, C]
            B_T, C_vis = visual_feat.shape
        else:
            raise ValueError(f"Unsupported visual_feat dimension: {visual_feat.dim()}")
        
        B = f_cond.shape[0]
        T = B_T // B
        
       
        if self.f_cond_proj is not None:
            f_cond = self.f_cond_proj(f_cond)  # [B, C_cond] -> [B, C_vis]
        
        
        f_cond_expanded = f_cond.unsqueeze(1).expand(B, T, C_vis).reshape(B_T, C_vis)
        
       
        if visual_feat.dim() == 4:
            
            mean = visual_feat.mean(dim=[2, 3], keepdim=True)  # [B*T, C, 1, 1]
            std = visual_feat.std(dim=[2, 3], keepdim=True) + self.eps  # [B*T, C, 1, 1]
        else:
            
            mean = visual_feat.mean(dim=0, keepdim=True)  # [1, C]
            std = visual_feat.std(dim=0, keepdim=True) + self.eps  # [1, C]
        
       
        gamma = self.gamma_mlp(f_cond_expanded)  # [B*T, C_vis]
        beta = self.beta_mlp(f_cond_expanded)  # [B*T, C_vis]
        
        
        if self.gamma_output_activation is not None:
            gamma = self.gamma_output_activation(gamma)
        if self.beta_output_activation is not None:
            beta = self.beta_output_activation(beta)
        
       
        if visual_feat.dim() == 4:
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B*T, C, 1, 1]
            beta = beta.unsqueeze(-1).unsqueeze(-1)  # [B*T, C, 1, 1]
        
        
        normalized = (visual_feat - mean) / std
        modulated_feat = gamma * normalized + beta
        
        
        if self.use_residual:
            
            weight = torch.sigmoid(self.cn_weight)
            
            output = (1.0 - weight) * visual_feat + weight * modulated_feat
            return output
        else:
            return modulated_feat


class AdaptiveCNF3Fusion(nn.Module):
    
    def __init__(self, hidden_dim, use_residual=False):
        
        super().__init__()
        
        self.cn_weight = nn.Parameter(torch.tensor(0.5))
        self.f3_weight = nn.Parameter(torch.tensor(0.5))
        
        
        self.cn_layer = ConditionalNormLayer(
            visual_feat_dim=hidden_dim,
            f_cond_dim=hidden_dim,
            use_residual=use_residual
        )
    
    def forward(self, f3_features, f_cond):
        
        
        cn_w = torch.sigmoid(self.cn_weight)
        f3_w = torch.sigmoid(self.f3_weight)
        
        total_w = cn_w + f3_w + 1e-8
        cn_w = cn_w / total_w
        f3_w = f3_w / total_w
        
        
        if isinstance(f3_features, list):
            fused_features = []
            for feat in f3_features:
                
                B_T, C, H, W = feat.shape
                B = f_cond.shape[0]
                T = B_T // B
                
                
                cn_feat = self.cn_layer(feat, f_cond)  # [B*T, C, H, W]
                
               
                fused_feat = cn_w * cn_feat + f3_w * feat
                fused_features.append(fused_feat)
            return fused_features
        else:
            
            B_T, N, C = f3_features.shape
            B = f_cond.shape[0]
            T = B_T // B
            
            
            feat_reshaped = f3_features.permute(0, 2, 1)  # [B*T, C, N]
            
            
            cn_feat_reshaped = self.cn_layer(feat_reshaped, f_cond)  # [B*T, C, N]
            
            
            cn_feat = cn_feat_reshaped.permute(0, 2, 1)  # [B*T, N, C]
            
            
            fused_feat = cn_w * cn_feat + f3_w * f3_features
            return fused_feat


class MatchingHead(nn.Module):
    
    def __init__(self, hidden_dim, nhead=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, visual_feat, text_feat):
       
        if visual_feat.shape[0] != text_feat.shape[0]:
            
            B_vis, C = visual_feat.shape
            B_text = text_feat.shape[0]
            N_per_sample = B_vis // B_text
            
            
            text_feat_expanded = text_feat.unsqueeze(1).expand(B_text, N_per_sample, C)
            text_feat_expanded = text_feat_expanded.reshape(B_vis, C)
        else:
            text_feat_expanded = text_feat
        
        
        visual_feat_norm = self.norm(visual_feat)
        text_feat_norm = self.norm(text_feat_expanded)
        combined = torch.cat([visual_feat_norm, text_feat_norm], dim=1)  
        
       
        match_scores = self.score_head(combined).squeeze(-1)  
        
        return match_scores


class RoIFeatureExtractor(nn.Module):
    
    def __init__(self, hidden_dim, output_size=7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, multi_scale_features, boxes, image_size):
        
        
        H, W = image_size
        device = multi_scale_features[0].device
        
       
        roi_features_list = []
        for feat in multi_scale_features:
           
            B_T, C, H_feat, W_feat = feat.shape
            
            
            roi_feat_list = []
            for box in boxes:
                if len(box) == 4:
                    if box[2] > box[0]:  
                        x1, y1, x2, y2 = box
                    else:  
                        cx, cy, w, h = box
                        x1 = cx - w/2
                        y1 = cy - h/2
                        x2 = cx + w/2
                        y2 = cy + h/2
                else:
                    raise ValueError(f"Unsupported box format: {len(box)} values")
                
                
                x1_feat = int(x1 * W_feat)
                y1_feat = int(y1 * H_feat)
                x2_feat = int(x2 * W_feat)
                y2_feat = int(y2 * H_feat)
                
                
                x1_feat = max(0, min(x1_feat, W_feat - 1))
                y1_feat = max(0, min(y1_feat, H_feat - 1))
                x2_feat = max(x1_feat + 1, min(x2_feat, W_feat))
                y2_feat = max(y1_feat + 1, min(y2_feat, H_feat))
                
               
                roi_patch = feat[0:1, :, y1_feat:y2_feat, x1_feat:x2_feat]  # [1, C, h, w]
                
                roi_patch = F.adaptive_avg_pool2d(roi_patch, (self.output_size, self.output_size))  # [1, C, output_size, output_size]
                
               
                roi_feat = roi_patch.mean(dim=[2, 3]).squeeze(0)  # [C]
                roi_feat_list.append(roi_feat)
            
            roi_feat_batch = torch.stack(roi_feat_list, dim=0)  # [N, C]
            roi_features_list.append(roi_feat_batch)
        
      
        roi_features_concat = torch.cat(roi_features_list, dim=1)  # [N, C*num_scales]
        roi_features = self.fusion(roi_features_concat)  # [N, C]
        
        return roi_features


class TextConditionGenerator(nn.Module):
 
    def __init__(self, hidden_dim, activation_type='relu', use_f_cond_refine=False, f_cond_refine_dim=None):
        
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_f_cond_refine = use_f_cond_refine
        
        
        self.alpha_sen = nn.Parameter(torch.tensor(0.5))
        self.alpha_mean = nn.Parameter(torch.tensor(0.5))
        
       
        from models.activation_utils import get_activation, GLUMLP
        
        if activation_type.lower() == 'glu':
            self.phi = GLUMLP(hidden_dim, hidden_dim, hidden_dim, num_layers=2)
        else:
            activation = get_activation(activation_type)
            self.phi = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, hidden_dim)
            )
        

        if use_f_cond_refine:
            refine_dim = f_cond_refine_dim if f_cond_refine_dim is not None else hidden_dim
            if activation_type.lower() == 'glu':
                self.f_cond_refiner = GLUMLP(hidden_dim, refine_dim, hidden_dim, num_layers=2)
            else:
                activation = get_activation(activation_type)
                self.f_cond_refiner = nn.Sequential(
                    nn.Linear(hidden_dim, refine_dim),
                    nn.LayerNorm(refine_dim),
                    activation,
                    nn.Dropout(0.1),
                    nn.Linear(refine_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim)
                )
        else:
            self.f_cond_refiner = None
    
    def forward(self, text_word_features):
       
        f_sen = text_word_features[0]  # [batch_size, hidden_dim]
        
       
        f_mean = text_word_features.mean(dim=0)  # [batch_size, hidden_dim]
        
       
        f_mean_transformed = self.phi(f_mean)  # [batch_size, hidden_dim]
        
        
        f_cond = self.alpha_sen * f_sen + self.alpha_mean * f_mean_transformed
        
        
        if self.use_f_cond_refine and self.f_cond_refiner is not None:
            f_cond = self.f_cond_refiner(f_cond)
        
        return f_cond


def build(args):
    if args.binary:
        num_classes = 1
    else:
        if args.dataset_file == 'ytvos':
            num_classes = 65
        elif args.dataset_file == 'davis':
            num_classes = 78
        elif args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb':
            num_classes = 1
        else:
            num_classes = 91 
    device = torch.device(args.device)

   
    if 'video_swin' in args.backbone:
        from .video_swin_transformer import build_video_swin_backbone
        backbone = build_video_swin_backbone(args)
    elif 'swin' in args.backbone:
        from .swin_transformer import build_swin_backbone
        backbone = build_swin_backbone(args)
    else:
        backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)

    model = PC2F(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        num_frames=args.num_frames,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        freeze_text_encoder=args.freeze_text_encoder,
        args=args
    )
    matcher = build_matcher(args)
    weight_dict = {}
    weight_dict['loss_ce'] = args.cls_loss_coef
    weight_dict['loss_bbox'] = args.bbox_loss_coef
    weight_dict['loss_giou'] = args.giou_loss_coef
   
    if getattr(args, 'use_otsm', False):
        weight_dict['loss_match'] = getattr(args, 'otsm_loss_coef', 0.1)
   
    if getattr(args, 'use_dsats', False):
        weight_dict['loss_foreground_token_selector'] = getattr(args, 'fts_loss_coef', 1.0)
   
    if args.masks:  # always true
        weight_dict['loss_mask'] = args.mask_loss_coef
        weight_dict['loss_dice'] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']
    if args.masks:
        losses += ['masks']
   
    if getattr(args, 'use_dsats', False):
        losses += ['foreground_token_selector']
   
    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        focal_alpha=args.focal_alpha,
        args=args)
    criterion.to(device)

    
    postprocessors = build_postprocessors(args, args.dataset_file)
    return model, criterion, postprocessors