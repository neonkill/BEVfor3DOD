# -----------------------------------------------------------------------
# Copyright (C) 2020 Brady Zhou
# Released under the MIT license
# https://github.com/bradyz/cross_view_transformers/blob/master/LICENSE
# -----------------------------------------------------------------------


import torch
import logging

import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss
from einops import rearrange


logger = logging.getLogger(__name__)

class DepthBCELoss(torch.nn.Module):
    def __init__(self, downsample_factor=8, variance_focus = 0.85):
        super(DepthBCELoss, self).__init__()
        self.variance_focus = variance_focus
        self.downsample_factor = downsample_factor
        self.dbound = [2.0, 58.0, 0.5] #2 ~ 58 m , 0.5 bin
        self.depth_channels = int((self.dbound[1] - self.dbound[0]) / self.dbound[2])
    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        # if self.downsample_factor == 4:
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        # elif self.downsample_factor == 16:
        #     gt_depths = gt_depths.view(
        #         B * N,
        #         round(H / self.downsample_factor),
        #         self.downsample_factor,
        #         round(W / self.downsample_factor),
        #         self.downsample_factor,
        #         1,
        #     )

        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous() #[B*N, H/d, W/d, 1, d, d]
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                    W // self.downsample_factor)

        gt_depths = (gt_depths -
                        (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                                num_classes=self.depth_channels + 1).view(
                                    -1, self.depth_channels + 1)[:, 1:]

        return gt_depths.float()

    def forward(self, pred, batch):
        label = batch['depths']
        pred = pred['depth_bin']
        depth_labels = self.get_downsampled_gt_depth(label)
        depth_preds = pred.permute(0, 2, 3, 1).contiguous().view(-1, self.depth_channels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        
        depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return depth_loss

class SegmentationBCELoss(torch.nn.Module):
    """
    Explain about loss function.
    Arguments:
        ignore_label: Integer, label to ignore.
        weight: Tensor, a manual rescaling weight given to each class.
    """
    def __init__(self, ignore_label=5, weight=None):
        super().__init__()
        self.ignore_label = ignore_label
        # self.weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 
        #                                 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 
        #                                 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        # self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight, ignore_index=ignore_label)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        # self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)
        self.num_classes = 19
        
    def forward(self, pred, batch):
        logits = pred['seg_logits']
        # print("seg gt shape : ", batch['seg_gt'].shape)
        labels = rearrange(batch['seg_gt'], 'b n ... -> (b n) ...').long()
        
        shape = labels.shape
        
        one_hot = torch.zeros((shape[0], self.num_classes) + shape[1:]).cuda()
        
        one_hot_labels = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + 1e-6
        ph, pw = logits.size(2), logits.size(3)
        h, w = labels.size(1), labels.size(2)
        
        if ph != h or pw != w:
            logits = F.interpolate(input=logits, size=(
                h, w), mode='bilinear', align_corners=True)

        pixel_losses = self.criterion(logits, one_hot_labels)
        return pixel_losses

 
class SegmentationBCELossWithSqueeze(torch.nn.Module):
    """
    BCE Segmentation Loss for Squeezed class (19 -> 2, road, vehicle)
    Arguments:
        ignore_label: Integer, label to ignore.
        weight: Tensor, a manual rescaling weight given to each class.
    """
    def __init__(self, ignore_label=5, weight=None):
        super().__init__()
        self.ignore_label = ignore_label
        # self.weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 
        #                                 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 
        #                                 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        # self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight, ignore_index=ignore_label)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        # self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)
        self.num_classes = 2
        
    def forward(self, pred, batch):
        logits = pred['seg_logits']
        # print("seg gt shape : ", batch['seg_gt'].shape)
        labels = rearrange(batch['seg_gt'], 'b n ... -> (b n) ...').long()
        
        shape = labels.shape
        
        one_hot = torch.zeros((shape[0], self.ignore_label + 1) + shape[1:]).cuda()
        
        one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + 1e-6
        one_hot_labels = torch.split(one_hot, [self.num_classes, self.ignore_label+1-self.num_classes], dim=1)[0]
        ph, pw = logits.size(2), logits.size(3)
        h, w = labels.size(1), labels.size(2)
        
        if ph != h or pw != w:
            logits = F.interpolate(input=logits, size=(
                h, w), mode='bilinear', align_corners=True)

        pixel_losses = self.criterion(logits, one_hot_labels)
        return pixel_losses
    
class SegmentationLoss(torch.nn.Module):
    """
    Explain about loss function.
    Arguments:
        ignore_label: Integer, label to ignore.
        weight: Tensor, a manual rescaling weight given to each class.
    """
    def __init__(self, ignore_label=255, weight=None):
        super().__init__()
        self.ignore_label = ignore_label
        self.weight = torch.FloatTensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 
                                        1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 
                                        0.2500])
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight, ignore_index=ignore_label)
        # self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)
        # self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)
        self.loss_weights = [1.0]

    def forward(self, pred, batch):
        logits = pred['seg_logits']
        # print("seg gt shape : ", batch['seg_gt'].shape)
        labels = rearrange(batch['seg_gt'], 'b n ... -> (b n) ...').long()
        ph, pw = logits.size(2), logits.size(3)
        h, w = labels.size(1), labels.size(2)
        if ph != h or pw != w:
            logits = F.interpolate(input=logits, size=(
                h, w), mode='bilinear', align_corners=True)

        pixel_losses = self.criterion(logits, labels)
        return pixel_losses

class SigmoidFocalLoss(torch.nn.Module):
    def __init__(
        self,
        alpha=-1.0,
        gamma=2.0,
        reduction='mean'
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, label):
        # print('after')
        # print(pred.shape, label.shape)
        return sigmoid_focal_loss(pred, label, self.alpha, self.gamma, self.reduction)


class BinarySegmentationLoss(SigmoidFocalLoss):
    def __init__(
        self,
        cls,
        label_indices=None,
        min_visibility=None,
        alpha=-1.0,
        gamma=2.0
    ):
        super().__init__(alpha=alpha, gamma=gamma, reduction='none')

        self.cls = cls
        self.label_indices = label_indices
        self.min_visibility = min_visibility

    def forward(self, pred, batch):
        if isinstance(pred, dict):
            pred = pred[self.cls]

        label = batch['bev']

        if self.label_indices is not None:
            label = [label[:, idx].max(1, keepdim=True).values for idx in self.label_indices]
            label = torch.cat(label, 1)
        
        # print('before')
        # print(pred.shape, label.shape)
        loss = super().forward(pred.contiguous(), label.contiguous())

        if self.min_visibility is not None:
            mask = batch['visibility'] >= self.min_visibility
            loss = loss[mask[:, None]]

        return loss.mean()

class BinarySegmentationLoss_1ch(SigmoidFocalLoss):
    def __init__(
        self,
        label_indices=None,
        min_visibility=None,
        alpha=-1.0,
        gamma=2.0
    ):
        super().__init__(alpha=alpha, gamma=gamma, reduction='none')

        self.label_indices = label_indices
        self.min_visibility = min_visibility

    def forward(self, pred, batch):
        if isinstance(pred, dict):
            pred = pred['bev']
        label = batch['bev']

        if self.label_indices is not None:
            label = [label[:, idx].max(1, keepdim=True).values for idx in self.label_indices]
            label = torch.cat(label, 1)

        loss = super().forward(pred.contiguous(), label.contiguous())

        if self.min_visibility is not None:
            mask = batch['visibility'] >= self.min_visibility
            loss = loss[mask[:, None]]

        return loss.mean()


class CenterLoss(SigmoidFocalLoss):
    def __init__(
        self,
        min_visibility=None,
        alpha=-1.0,
        gamma=2.0
    ):
        super().__init__(alpha=alpha, gamma=gamma, reduction='none')

        self.min_visibility = min_visibility

    def forward(self, pred, batch):
        pred = pred['center'].contiguous()
        label = batch['center'].contiguous()
        loss = super().forward(pred, label)

        if self.min_visibility is not None:
            mask = batch['visibility'] >= self.min_visibility
            loss = loss[mask[:, None]]

        return loss.mean()


class SigmoidCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, label):

        if isinstance(pred, dict):
            pred = pred['bev'][:,self.class_idx]        # b 1 h w

        label = batch['bev']

        if self.label_indices is not None:
            label = [label[:, idx].max(1, keepdim=True).values for idx in self.label_indices]
            label = torch.cat(label, 1)     # b 1 h w


        num_pos = (label == 1).float().sum(dim=1).clamp(min=1.0)
        num_neg = (label == 0).float().sum(dim=1)
        pos_weight = (num_neg / num_pos).unsqueeze(1)

        weight_loss = label * pos_weight + (1 - label)
        loss = F.binary_cross_entropy_with_logits(pred, label, 
                                                    reduction="mean", 
                                                    weight=weight_loss)
        
        if self.min_visibility is not None:
            mask = batch['visibility'] >= self.min_visibility
            loss = loss[mask[:, None]]

        return loss.mean()


class MultipleLoss(torch.nn.ModuleDict):
    """
    losses = MultipleLoss({'bce': torch.nn.BCEWithLogitsLoss(), 'bce_weight': 1.0})
    loss, unweighted_outputs = losses(pred, label)
    """
    def __init__(self, modules_or_weights):
        modules = dict()
        weights = dict()

        # Parse only the weights
        for key, v in modules_or_weights.items():
            if isinstance(v, float):
                weights[key.replace('_weight', '')] = v

        # Parse the loss functions
        for key, v in modules_or_weights.items():
            if not isinstance(v, float):
                modules[key] = v

                # Assign weight to 1.0 if not explicitly set.
                if key not in weights:
                    logger.warn(f'Weight for {key} was not specified.')
                    weights[key] = 1.0

        assert modules.keys() == weights.keys()

        super().__init__(modules)

        self._weights = weights

    def forward(self, pred, batch):
        outputs = {k: v(pred, batch) for k, v in self.items()}
        total = sum(self._weights[k] * o for k, o in outputs.items())

        return total, outputs

