# -----------------------------------------------------------------------
# Copyright (C) 2020 Brady Zhou
# Released under the MIT license
# https://github.com/bradyz/cross_view_transformers/blob/master/LICENSE
# -----------------------------------------------------------------------


import torch
import numpy as np

from torchmetrics import Metric
from typing import List, Optional
import torch.nn.functional as F

class_mapping = {0: 'road_iou',
                1: 'lane_iou',
                2: 'vehicle_iou'}

class BaseIoUMetric(Metric):
    """
    Computes intersection over union at given thresholds
    """
    def __init__(self, class_name=None, thresholds=[0.5]):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)

        thresholds = torch.FloatTensor(thresholds)

        self.cls = class_name
        self.add_state('thresholds', default=thresholds, dist_reduce_fx='mean')
        self.add_state('tp', default=torch.zeros_like(thresholds), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.zeros_like(thresholds), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.zeros_like(thresholds), dist_reduce_fx='sum')

    def update(self, pred, label):
        pred = pred.detach().sigmoid().reshape(-1)
        label = label.detach().bool().reshape(-1)

        pred = pred[:, None] >= self.thresholds[None]
        label = label[:, None]

        self.tp += (pred & label).sum(0)
        self.fp += (pred & ~label).sum(0)
        self.fn += (~pred & label).sum(0)

    def compute(self):
        ious = self.tp / (self.tp + self.fp + self.fn + 1e-7)
        return [(self.cls, ious.item())] # {f'{self.cls}': ious.item()}
        # return [(t.item(), i.item()) for t, i in zip(thresholds, ious)]

class IoUMetric(BaseIoUMetric):
    def __init__(self, label_indices: List[List[int]], min_visibility: Optional[int] = None):
        """
        label_indices:
            transforms labels (c, h, w) to (len(labels), h, w)
            see config/experiment/* for examples

        min_visibility:
            passing "None" will ignore the visibility mask
            otherwise uses visibility values to ignore certain labels
            visibility mask is in order of "increasingly visible" {1, 2, 3, 4, 255 (default)}
            see https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md#visibility
        """
        super().__init__()

        self.label_indices = label_indices
        self.min_visibility = min_visibility

    def update(self, pred, batch):
        if isinstance(pred, dict):
            pred = pred['bev']                                                              # b c h w

        label = batch['bev']                                                                # b n h w
        label = [label[:, idx].max(1, keepdim=True).values for idx in self.label_indices]
        label = torch.cat(label, 1)                                                         # b c h w

        if self.min_visibility is not None:
            mask = batch['visibility'] >= self.min_visibility
            mask = mask[:, None].expand_as(pred)                                            # b c h w

            pred = pred[mask]                                                               # m
            label = label[mask]                                                             # m

        return super().update(pred, label)


class SegMetrics(Metric):
    """
    Computes Seg metrics
    """
    def __init__(self, num_cls):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.K = num_cls
        self.cnt = 0
        self.add_state('inter', default=torch.zeros(num_cls), dist_reduce_fx='sum')
        self.add_state('union', default=torch.zeros(num_cls), dist_reduce_fx='sum')
        self.add_state('target', default=torch.zeros(num_cls), dist_reduce_fx='sum')

        
    def update(self, output, batch, ingore_idx=255):      
        b, n, h, w = batch['seg_gt'].shape
        target = batch['seg_gt'].view(b*n,h,w)
        output = output['seg_logits']

        if output.shape != target.shape:
            output = F.interpolate(output, size=target.shape[1:],
                            mode='bilinear', align_corners=True)

        output = output.max(1)[1]

        assert output.shape == target.shape, f'{output.shape} does not match {target.shape}'
        
        output = output.view(-1)
        target = target.view(-1)

        output[target==ingore_idx] = ingore_idx

        intersection = output[output == target]

        inter_area = torch.histc(intersection, bins=self.K, min=0, max=self.K-1)
        output_area = torch.histc(output, bins=self.K, min=0, max=self.K-1)
        target_area = torch.histc(target, bins=self.K, min=0, max=self.K-1)
        union_area = output_area + target_area - inter_area

        self.inter += inter_area
        self.union += union_area
        self.target += target_area

        self.cnt = self.cnt + 1
        

    def compute(self):
        inter, union, target = self.inter.cpu().numpy(), self.union.cpu().numpy(), self.target.cpu().numpy()
        iou_cls = inter / (union + 1e-10)
        acc_cls = inter / (target + 1e-10)

        mIoU = np.mean(iou_cls)
        mAcc = np.mean(acc_cls)
        allAcc = sum(inter) / (sum(target) + 1e-10)

        return {'mIoU': mIoU, 'mAcc': mAcc, 'allAcc': allAcc}

class DepthMetrics(Metric):
    """
    Computes Depth metrics
    """
    def __init__(self):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.add_state("abs_rel", default = torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("sq_rel", default = torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("rmse", default = torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("rmse_log", default = torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("a1", default = torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("a2", default = torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("a3", default = torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("silog", default = torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default = torch.tensor(0.), dist_reduce_fx="sum")

        
    def update(self, pred, batch, kind ='bin'):
        # pred['depth'] = [BxN, C, H, W]
        # batch['depth'] = [B, 6, 256, 704]
        depth_gt = batch['depths']
        b, n, h, w = depth_gt.shape
        
        if kind == 'bin':
            pred_depth = pred['depth_bin'].detach()
            pred_depth = F.interpolate(pred_depth, (h, w), mode='bilinear')
            pred_depth = torch.argmax(pred_depth, dim=1, keepdim=True) * 0.5 + 2 #112 bin -> value
            mask = torch.logical_and(depth_gt > 0.0, depth_gt < 59.0)  #[B, 6, 256, 704]


        if kind == 'disp':
            pred_depth = pred['depth'].detach()
            pred_depth = torch.clamp(pred_depth, min=2.0, max=58.0)
            mask = depth_gt > 0.0
            
        pred_depth = pred_depth.reshape(b, n, h, w) #[B, 6, 256, 704]

        for i in range(pred_depth.shape[0]): #for batch
            empty_count = 0
            for n in range(pred_depth.shape[1]): #for camera
                gt = depth_gt[i][n][mask[i][n]]
                if len(gt) == 0:
                    empty_count += 1
                    continue
                pred = pred_depth[i][n][mask[i][n]]
                
                if kind == 'disp':
                    pred *= torch.median(gt) / torch.median(pred)
                    pred = torch.clamp(pred, min=1e-3, max=80)
                
                abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, silog = self.compute_depth_errors(pred, gt)
                
                self.abs_rel += abs_rel.float()
                self.sq_rel += sq_rel.float()
                self.rmse += rmse.float()
                self.rmse_log += rmse_log.float()
                self.a1 += a1.float()                
                self.a2 += a2.float()
                self.a3 += a3.float()
                self.silog += silog.float()
        
            self.total = self.total + pred_depth.shape[1] - empty_count
        
    
    def compute_depth_errors(self, pred, gt):
        """Computation of error metrics between predicted and ground truth depths
        """
        thresh = torch.max((gt / pred), (pred / gt))
        a1 = (thresh < 1.25     ).float().mean()
        a2 = (thresh < 1.25 ** 2).float().mean()
        a3 = (thresh < 1.25 ** 3).float().mean()

        rmse = (gt - pred) ** 2
        rmse = torch.sqrt(rmse.mean())

        rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
        rmse_log = torch.sqrt(rmse_log.mean())

        abs_rel = torch.mean(torch.abs(gt - pred) / gt)

        sq_rel = torch.mean((gt - pred) ** 2 / gt)

        # * square invariant error metric
        log_diff = torch.log(gt) - torch.log(pred)
        silog = torch.sqrt(torch.mean(log_diff**2)-torch.mean(log_diff)**2)*100

        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, silog
        
    
    def compute(self):
        return {'abs_rel' : self.abs_rel.float() / self.total,
                'a1' : self.a1.float() / self.total,
                'a2' : self.a2.float() / self.total,
                'a3' : self.a3.float() / self.total,}