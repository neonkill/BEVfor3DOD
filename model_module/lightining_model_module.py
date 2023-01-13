# -----------------------------------------------------------------------
# Copyright (C) 2020 Brady Zhou
# Released under the MIT license
# https://github.com/bradyz/cross_view_transformers/blob/master/LICENSE
# -----------------------------------------------------------------------
# Modified by yelin2
# -----------------------------------------------------------------------


import mmcv
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR


from torch.cuda.amp.autocast_mode import autocast
from model_module.model.detection.base_bev_depth import BaseBEVDepth
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes

from model_module.torch_dist import all_gather_object, get_rank, synchronize
from model_module.det_evaluators import DetNuscEvaluator

final_dim = (256, 704)

backbone_conf = {
    'x_bound': [-51.2, 51.2, 0.8],
    'y_bound': [-51.2, 51.2, 0.8],
    'z_bound': [-5, 3, 8],
    'd_bound': [2.0, 58.0, 0.5],
    'final_dim':
    final_dim,
    'output_channels':
    80,
    'downsample_factor':
    16,
    'img_backbone_conf':
    dict(
        type='ResNet',
        depth=50,
        frozen_stages=0,
        out_indices=[0, 1, 2, 3],
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    'img_neck_conf':
    dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128],
    ),
    'depth_net_conf':
    dict(in_channels=512, mid_channels=512)
}


bev_backbone = dict(
    type='ResNet',
    in_channels=80,
    depth=18,
    num_stages=3,
    strides=(1, 2, 2),
    dilations=(1, 1, 1),
    out_indices=[0, 1, 2],
    norm_eval=False,
    base_channels=160,
)

bev_neck = dict(type='SECONDFPN',
                in_channels=[80, 160, 320, 640],
                upsample_strides=[1, 2, 4, 8],
                out_channels=[64, 64, 64, 64])

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

TASKS = [
    dict(num_class=1, class_names=['car']),
    dict(num_class=2, class_names=['truck', 'construction_vehicle']),
    dict(num_class=2, class_names=['bus', 'trailer']),
    dict(num_class=1, class_names=['barrier']),
    dict(num_class=2, class_names=['motorcycle', 'bicycle']),
    dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
]

common_heads = dict(reg=(2, 2),
                    height=(1, 2),
                    dim=(3, 2),
                    rot=(2, 2),
                    vel=(2, 2))

bbox_coder = dict(
    type='CenterPointBBoxCoder',
    post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_num=500,
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.2, 0.2, 8],
    pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    code_size=9,
)

train_cfg = dict(
    point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    grid_size=[512, 512, 1],
    voxel_size=[0.2, 0.2, 8],
    out_size_factor=4,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
)

test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    max_pool_nms=False,
    min_radius=[4, 12, 10, 1, 0.85, 0.175],
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.2, 0.2, 8],
    nms_type='circle',
    pre_max_size=1000,
    post_max_size=83,
    nms_thr=0.2,
)

head_conf = {
    'bev_backbone_conf': bev_backbone,
    'bev_neck_conf': bev_neck,
    'tasks': TASKS,
    'common_heads': common_heads,
    'bbox_coder': bbox_coder,
    'train_cfg': train_cfg,
    'test_cfg': test_cfg,
    'in_channels': 256,  # Equal to bev_neck output_channels.
    'loss_cls': dict(type='GaussianFocalLoss', reduction='mean'),
    'loss_bbox': dict(type='L1Loss', reduction='mean', loss_weight=0.25),
    'gaussian_overlap': 0.1,
    'min_radius': 2,
}



class ModelModule(pl.LightningModule):
    def __init__(self, fullmodel=None, loss_func=None, metrics=None, optimizer_args=None, scheduler_args=None, cfg=None):
        super().__init__()

        self.save_hyperparameters(
            cfg,
            ignore=['fullmodel', 'loss_func', 'metrics', 'optimizer_args', 'scheduler_args'])

        self.backbone_conf = backbone_conf
        self.head_conf = head_conf
        self.fullmodel = BaseBEVDepth(self.backbone_conf,
                                        self.head_conf,
                                        is_train_depth=True)  

        self.downsample_factor = backbone_conf['downsample_factor']
        self.dbound = backbone_conf['d_bound']
        self.depth_channels = int(
            (self.dbound[1] - self.dbound[0]) / self.dbound[2])

        default_root_dir = './outputs/'
        mmcv.mkdir_or_exist(default_root_dir)
        self.default_root_dir = default_root_dir
        self.evaluator = DetNuscEvaluator(class_names=CLASSES,
                                          output_dir=self.default_root_dir)
        self.loss_func = loss_func
        self.metrics = metrics

        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args


    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return 3.0 * depth_loss


    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
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


    def forward(self, batch):
        return self.fullmodel(batch)

    def shared_step(self, batch, prefix='', on_step=False, return_output=True):
        
        preds, depth_preds = self(batch)

        gt_boxes = [gt_box.cuda() for gt_box in batch['gt_boxes']]
        gt_labels = [gt_label.cuda() for gt_label in batch['gt_labels']]
        
        targets = self.fullmodel.get_targets(gt_boxes, gt_labels)
        detection_loss = self.fullmodel.loss(targets, preds)

        depth_labels = batch['depths']
        if len(depth_labels.shape) == 5:
            # only key-frame will calculate depth loss
            depth_labels = depth_labels[:, 0, ...]
        depth_loss = self.get_depth_loss(depth_labels.cuda(), depth_preds)

        total_loss = detection_loss + depth_loss

        if self.trainer is not None:
            self.log(f'{prefix}/total_loss', 
                        total_loss.detach(), on_step=on_step, on_epoch=True)
            self.log(f'{prefix}/depth_loss', 
                        depth_loss.detach(), on_step=on_step, on_epoch=True)
            self.log(f'{prefix}/detection_loss', 
                        detection_loss.detach(), on_step=on_step, on_epoch=True)

        if prefix == 'val' or prefix == 'test':
            for img_meta in batch['img_metas']:
                img_meta['box_type_3d'] = LiDARInstance3DBoxes
            
            if isinstance(self.fullmodel, torch.nn.parallel.DistributedDataParallel):
                results = self.fullmodel.module.get_bboxes(preds, batch['img_metas'])
            else:
                results = self.fullmodel.get_bboxes(preds, batch['img_metas'])
            for i in range(len(results)):
                results[i][0] = results[i][0].detach().cpu().numpy()
                results[i][1] = results[i][1].detach().cpu().numpy()
                results[i][2] = results[i][2].detach().cpu().numpy()
                results[i].append(batch['img_metas'][i])
            return results

        elif prefix == 'train':
            return total_loss


    def training_step(self, batch, batch_idx):
        # print(self.hparams)
        return self.shared_step(batch, 'train', True,
                                batch_idx % self.hparams.experiment.log_image_interval == 0)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val', False,
                                batch_idx % self.hparams.experiment.log_image_interval == 0)

    def vis_step(self, batch, batch_idx):
        return self.shared_step(batch, 'vis', False, True)

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test', False,
                                batch_idx % self.hparams.experiment.log_image_interval == 0)
        
    def test_epoch_end(self, test_step_outputs):
        all_pred_results = list()
        all_img_metas = list()
        for test_step_output in test_step_outputs:
            for i in range(len(test_step_output)):
                all_pred_results.append(test_step_output[i][:3])
                all_img_metas.append(test_step_output[i][3])
        synchronize()
        # TODO: Change another way.
        dataset_length = len(self.trainer.test_dataloaders[0].dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:dataset_length]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:dataset_length]
        if get_rank() == 0:
            self.evaluator.evaluate(all_pred_results, all_img_metas)
        # self._log_epoch_metrics('test')

    def on_validation_start(self) -> None:
        # self._log_epoch_metrics('train')
        self._enable_dataloader_shuffle(self.trainer.val_dataloaders)

    def validation_epoch_end(self, validation_step_outputs):
        all_pred_results = list()
        all_img_metas = list()
        for validation_step_output in validation_step_outputs:
            for i in range(len(validation_step_output)):
                all_pred_results.append(validation_step_output[i][:3])
                all_img_metas.append(validation_step_output[i][3])
        synchronize()

        len_dataset = len(self.trainer.val_dataloaders[0].dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:len_dataset]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:len_dataset]
        if get_rank() == 0:
            self.evaluator.evaluate(all_pred_results, all_img_metas)
            # metrics = self.evaluator.evaluate(all_pred_results, all_img_metas)
            # if isinstance(metrics, list):
            #     for metric in metrics:
            #         for k, v in metric.items():
            #             self.log(k, v)
            # elif isinstance(metrics, dict):
            #     for k, v in metric.items():
            #         self.log(k, v)

    def _log_epoch_metrics(self, prefix: str):
        """
        on_validation_start에서 train 할 때 저장된 metric logging 후 reset
        val 하면서 metric update 하고 val 끝나면 metric logging 후 reset
        """
        metrics = self.metrics.compute()

        for key, value in metrics.items():
            if len(value)>1:
                # print(f'{prefix}/metrics/{key}{value[0][0]:.1f}', value[0][1])
                # print(f'{prefix}/metrics/{key}{value[1][0]:.1f}', value[1][1])
                self.log(f'{prefix}/metrics/{key}{value[0][0]:.1f}', value[0][1])
                self.log(f'{prefix}/metrics/{key}{value[1][0]:.1f}', value[1][1])
            else:
                # print(f'{prefix}/metrics/{key}', value)
                self.log(f'{prefix}/metrics/{key}', value)

        self.metrics.reset()

    def _enable_dataloader_shuffle(self, dataloaders):
        """
        HACK for https://github.com/PyTorchLightning/pytorch-lightning/issues/11054
        """
        for v in dataloaders:
            v.sampler.shuffle = True
            v.sampler.set_epoch(self.current_epoch)

    def configure_optimizers(self, disable_scheduler=False):

        lr = 2e-4
        optimizer = torch.optim.AdamW(self.fullmodel.parameters(),
                                      lr=lr,
                                      weight_decay=1e-7)
        scheduler = MultiStepLR(optimizer, [19, 23])
        return [[optimizer], [scheduler]]


            

            
