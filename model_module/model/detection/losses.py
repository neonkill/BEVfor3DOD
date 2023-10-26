# -----------------------------------------------------------------------
# Copyright (C) 2020 Brady Zhou
# Released under the MIT license
# https://github.com/bradyz/cross_view_transformers/blob/master/LICENSE
# -----------------------------------------------------------------------


import torch
import logging


import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss
from ..detection.core.bbox.util import normalize_bbox
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models import build_loss
from mmdet3d.core.bbox.coders import build_bbox_coder
from omegaconf import OmegaConf, DictConfig, SCMode
from mmdet3d.core.bbox import get_box_type
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes

logger = logging.getLogger(__name__)

class bipartiteMatchingLoss(torch.nn.Module):
    def __init__(self, loss_cls, loss_bbox, loss_iou, assigner, sampler, bbox_coder):
        super().__init__()

        loss_cls = OmegaConf.to_container(loss_cls, resolve=True)
        loss_bbox = OmegaConf.to_container(loss_bbox, resolve=True)
        loss_iou = OmegaConf.to_container(loss_iou, resolve=True)
        assigner = OmegaConf.to_container(assigner, resolve=True)
        sampler = OmegaConf.to_container(sampler, resolve=True)
        bbox_coder = OmegaConf.to_container(bbox_coder, resolve=True)

        self.assigner = build_assigner(assigner)
        self.sampler = build_sampler(sampler, context=self)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.box_type_3d, self.box_mode_3d = get_box_type('LiDAR')

        self.num_classes = 10
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = False
        self.cls_out_channels = self.num_classes
        self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.code_size = 10
        self.pc_range = [-50.0,-50.0,-5.0, 50.0, 50.0, 3.0]
        self.code_weights = torch.nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):

        num_bboxes = bbox_pred.size(0)
        
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        
        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
    

        gt_labels = gt_labels.to(torch.int64)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        code_size =  9
        bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        bbox_targets_device = bbox_targets.device 
        # print(gt_bboxes.size(), bbox_pred.size())
        # DETR
        if sampling_result.pos_gt_bboxes.shape[-1]==4:
            sampling_result.pos_gt_bboxes = torch.zeros((0,9),dtype=torch.float32).to(bbox_targets_device)
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, 
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]
        
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, 
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list


    def forward(self, pred, batch):
        all_cls_scores = pred['det_pred']['all_cls_preds']
        all_bbox_preds = pred['det_pred']['all_bbox_preds']

        num_dec_layers = len(all_cls_scores)

        device=batch['gt_boxes'][0].device

        gt_bboxes_list = batch['gt_boxes']
        gt_labels_list = batch['gt_labels']


        gt_bboxes_list = [LiDARInstance3DBoxes(
            gt_box,
            box_dim=gt_box.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d) if not gt_box.shape[0] == 0 else gt_box for gt_box in gt_bboxes_list]

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) if not isinstance(gt_bboxes,torch.Tensor) else gt_bboxes  for gt_bboxes in gt_bboxes_list]

        gt_bboxes_ignore = None

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, 
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        return_loss = loss_dict['loss_cls'] + loss_dict['loss_bbox']
        
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            # loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            # loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            return_loss += loss_cls_i + loss_bbox_i
        
        return return_loss

