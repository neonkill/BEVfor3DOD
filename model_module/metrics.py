# -----------------------------------------------------------------------
# Copyright (C) 2020 Brady Zhou
# Released under the MIT license
# https://github.com/bradyz/cross_view_transformers/blob/master/LICENSE
# -----------------------------------------------------------------------


import torch

from torchmetrics import Metric
from typing import List, Optional
import torch.nn.functional as F

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
            pred_depth = pred.detach()
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