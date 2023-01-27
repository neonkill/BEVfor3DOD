# Copyright (c) Megvii Inc. All rights reserved.
import os
from functools import partial

import mmcv
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from pytorch_lightning.core import LightningModule
from torch.cuda.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import MultiStepLR



from model_module.model.detection.base_bev_depth import BaseBEVDepth
from model_module.torch_dist import all_gather_object, get_rank, synchronize
from model_module.det_evaluators import DetNuscEvaluator
from data_module.nusc_det_dataset import NuscDetDataset, collate_fn



# H = 900
# W = 1600
# final_dim = (256, 704)


# backbone_conf = {
#     'x_bound': [-51.2, 51.2, 0.8],
#     'y_bound': [-51.2, 51.2, 0.8],
#     'z_bound': [-5, 3, 8],
#     'd_bound': [2.0, 58.0, 0.5],
#     'final_dim':
#     final_dim,
#     'output_channels':
#     80,
#     'downsample_factor':
#     16,
#     'img_backbone_conf':
#     dict(
#         type='ResNet',
#         depth=50,
#         frozen_stages=0,
#         out_indices=[0, 1, 2, 3],
#         norm_eval=False,
#         init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
#     ),
#     'img_neck_conf':
#     dict(
#         type='SECONDFPN',
#         in_channels=[256, 512, 1024, 2048],
#         upsample_strides=[0.25, 0.5, 1, 2],
#         out_channels=[128, 128, 128, 128],
#     ),
#     'depth_net_conf':
#     dict(in_channels=512, mid_channels=512)
# }


# bev_backbone = dict(
#     type='ResNet',
#     in_channels=80,
#     depth=18,
#     num_stages=3,
#     strides=(1, 2, 2),
#     dilations=(1, 1, 1),
#     out_indices=[0, 1, 2],
#     norm_eval=False,
#     base_channels=160,
# )

# bev_neck = dict(type='SECONDFPN',
#                 in_channels=[80, 160, 320, 640],
#                 upsample_strides=[1, 2, 4, 8],
#                 out_channels=[64, 64, 64, 64])

CLASSES = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]

# TASKS = [
#     dict(num_class=1, class_names=['car']),
#     dict(num_class=2, class_names=['truck', 'construction_vehicle']),
#     dict(num_class=2, class_names=['bus', 'trailer']),
#     dict(num_class=1, class_names=['barrier']),
#     dict(num_class=2, class_names=['motorcycle', 'bicycle']),
#     dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
# ]

# common_heads = dict(reg=(2, 2),
#                     height=(1, 2),
#                     dim=(3, 2),
#                     rot=(2, 2),
#                     vel=(2, 2))

# bbox_coder = dict(
#     type='CenterPointBBoxCoder',
#     post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
#     max_num=500,
#     score_threshold=0.1,
#     out_size_factor=4,
#     voxel_size=[0.2, 0.2, 8],
#     pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
#     code_size=9,
# )

# train_cfg = dict(
#     point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
#     grid_size=[512, 512, 1],
#     voxel_size=[0.2, 0.2, 8],
#     out_size_factor=4,
#     dense_reg=1,
#     gaussian_overlap=0.1,
#     max_objs=500,
#     min_radius=2,
#     code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
# )

# test_cfg = dict(
#     post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
#     max_per_img=500,
#     max_pool_nms=False,
#     min_radius=[4, 12, 10, 1, 0.85, 0.175],
#     score_threshold=0.1,
#     out_size_factor=4,
#     voxel_size=[0.2, 0.2, 8],
#     nms_type='circle',
#     pre_max_size=1000,
#     post_max_size=83,
#     nms_thr=0.2,
# )

# head_conf = {
#     'bev_backbone_conf': bev_backbone,
#     'bev_neck_conf': bev_neck,
#     'tasks': TASKS,
#     'common_heads': common_heads,
#     'bbox_coder': bbox_coder,
#     'train_cfg': train_cfg,
#     'test_cfg': test_cfg,
#     'in_channels': 256,  # Equal to bev_neck output_channels.
#     'loss_cls': dict(type='GaussianFocalLoss', reduction='mean'),
#     'loss_bbox': dict(type='L1Loss', reduction='mean', loss_weight=0.25),
#     'gaussian_overlap': 0.1,
#     'min_radius': 2,
# }




class BEVDepthLightningModel(LightningModule):
    MODEL_NAMES = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith('__')
                         and callable(models.__dict__[name]))

    def __init__(self,
                #  gpus: int = 1,
                #  eval_interval=1,
                #  batch_size_per_device=4,
                 class_names=CLASSES,
                #  backbone_conf=backbone_conf,
                #  head_conf=head_conf,
                 default_root_dir='./outputs/',
                 cfg):
        super().__init__()
        self.save_hyperparameters()
        # self.gpus = 4 #! 
        # self.eval_interval = eval_interval #! 
        # self.batch_size_per_device = batch_size_per_device #!

        self.basic_lr_per_img = 2e-4 / 64 #! 
        self.class_names = class_names
        # self.backbone_conf = backbone_conf
        # self.head_conf = head_conf

        mmcv.mkdir_or_exist(default_root_dir)
        self.default_root_dir = default_root_dir
        self.evaluator = DetNuscEvaluator(class_names=self.class_names,
                                          output_dir=self.default_root_dir)
        self.fullmodel = BaseBEVDepth(self.backbone_conf,
                                  self.head_conf,
                                  is_train_depth=True)
        self.mode = 'valid'





    def forward(self, imgs, mats):
        return self.fullmodel(imgs, mats)

    def training_step(self, batch):
        (imgs, mats, _, _, gt_boxes, gt_labels, depth_labels) = batch
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            imgs = imgs.cuda()
            gt_boxes = [gt_box.cuda() for gt_box in gt_boxes]
            gt_labels = [gt_label.cuda() for gt_label in gt_labels]

        #* Forward
        preds, depth_preds = self(imgs, mats)

        #* Get Targets & Detection Loss
        if isinstance(self.fullmodel, torch.nn.parallel.DistributedDataParallel): #! 지워도?
            targets = self.fullmodel.module.get_targets(gt_boxes, gt_labels)
            detection_loss = self.fullmodel.module.loss(targets, preds)
        else:
            targets = self.fullmodel.get_targets(gt_boxes, gt_labels)
            detection_loss = self.fullmodel.loss(targets, preds)

        #* Depth Loss
        if len(depth_labels.shape) == 5:
            # only key-frame will calculate depth loss
            depth_labels = depth_labels[:, 0, ...]
        depth_loss = self.fullmodel.depth_loss(depth_labels.cuda(), depth_preds)

        self.log('detection_loss', detection_loss)
        self.log('depth_loss', depth_loss)

        return detection_loss + depth_loss

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')

    def predict_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'predict')

    def validation_epoch_end(self, validation_step_outputs):
        self.get_metrics(validation_step_outputs)

    def test_epoch_end(self, test_step_outputs):
        self.get_metrics(test_step_outputs)

    def configure_optimizers(self):
        lr = self.basic_lr_per_img * \
            self.batch_size_per_device * self.gpus
        optimizer = torch.optim.AdamW(self.fullmodel.parameters(),
                                      lr=lr,
                                      weight_decay=1e-7)
        scheduler = MultiStepLR(optimizer, [19, 23])
        return [[optimizer], [scheduler]]
             

    def eval_step(self, batch, batch_idx, prefix: str):
        (imgs, mats, _, img_metas, _, _) = batch
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            imgs = imgs.cuda()
            
        preds = self.fullmodel(imgs, mats)
        if isinstance(self.fullmodel, torch.nn.parallel.DistributedDataParallel):
            results = self.fullmodel.module.get_bboxes(preds, img_metas)
        else:
            results = self.fullmodel.get_bboxes(preds, img_metas)
        for i in range(len(results)):
            results[i][0] = results[i][0].detach().cpu().numpy()
            results[i][1] = results[i][1].detach().cpu().numpy()
            results[i][2] = results[i][2].detach().cpu().numpy()
            results[i].append(img_metas[i])
        return results

    def get_metrics(self, step_outputs):
        all_pred_results = list()
        all_img_metas = list()
        for step_output  in step_outputs:
            for i in range(len(step_output )):
                all_pred_results.append(step_output[i][:3])
                all_img_metas.append(step_output[i][3])
        synchronize()

        dataset_length = len(self.trainer.val_dataloaders.dataset) #! 여기 수정 필요
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:dataset_length]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:dataset_length]
        if get_rank() == 0:
            self.evaluator.evaluate(lm=self,  results=all_pred_results, img_metas=all_img_metas, logger=self.logger)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        return parent_parser
