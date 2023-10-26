# -----------------------------------------------------------------------
# Copyright (C) 2020 Brady Zhou
# Released under the MIT license
# https://github.com/bradyz/cross_view_transformers/blob/master/LICENSE
# -----------------------------------------------------------------------
# Modified by neonkill
# -----------------------------------------------------------------------


import mmcv
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR


from torch.cuda.amp.autocast_mode import autocast
from model_module.model.detection.base_bev_depth import BaseBEVDepth
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
# from mmdet3d.core.bbox.structures.box_3d_mode import Box3DMode
from mmdet3d.core.bbox import get_box_type

from model_module.torch_dist import all_gather_object, get_rank, synchronize
from model_module.det_evaluators import DetNuscEvaluator
from model_module.metrics import DepthMetrics
from mmdet3d.core import bbox3d2result

 
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

 
class ModelModule(pl.LightningModule): 

    def __init__(self, fullmodel=None, loss_func=None, metrics=None, optimizer_args=None, scheduler_args=None, cfg=None):
        super().__init__()
        # print(cfg)
        self.save_hyperparameters(
            cfg,
            ignore=['fullmodel', 'loss_func', 'metrics', 'optimizer_args', 'scheduler_args'])
        # self.eval_interval = 1 #! 
        # self.batch_size_per_device = 4 #!
        # self.basic_lr_per_img = 2e-4 
        self.fullmodel = fullmodel
        self.class_names = CLASSES
        self.box_type_3d, self.box_mode_3d = get_box_type('LiDAR')

        default_root_dir = './outputs/'
        mmcv.mkdir_or_exist(default_root_dir)
        self.default_root_dir = default_root_dir
        self.evaluator = DetNuscEvaluator(class_names=CLASSES,
                                          output_dir=self.default_root_dir)
        self.mode = 'valid'
        # self.loss_func = loss_func
        # self.metrics = metrics
        self.depth_metrics = DepthMetrics()
        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args

    def forward(self, batch):
        return self.fullmodel(batch)

    def training_step(self, batch):
        # (imgs, mats, _, _, gt_boxes, gt_labels, depth_labels) = batch
        det_preds, depth_preds = self(batch)
        # gt_boxes = [LiDARInstance3DBoxes(
        #             gt_box,
        #             box_dim=len(gt_box[0]),
        #             origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d) for gt_box in batch['gt_boxes']]
        gt_boxes = [gt_box.cuda() for gt_box in batch['gt_boxes']]
        gt_labels = [gt_label.cuda() for gt_label in batch['gt_labels']]
        depth_labels = batch['depths']
        
        

        #* Get Targets & Detection Loss
        if isinstance(self.fullmodel, torch.nn.parallel.DistributedDataParallel): #! 지워도?
            # targets = self.fullmodel.module.get_targets(det_preds['all_cls_scores'][-1], det_preds['all_bbox_preds'][-1], gt_boxes, gt_labels)
            detection_loss = self.fullmodel.module.loss(gt_boxes,gt_labels, det_preds)
        else:
            # targets = self.fullmodel.get_targets(det_preds['all_cls_scores'][-1], det_preds['all_bbox_preds'][-1], gt_boxes, gt_labels)
            detection_loss = self.fullmodel.loss(gt_boxes,gt_labels, det_preds)

        #* Depth Loss
        if len(depth_labels.shape) == 5:
            # only key-frame will calculate depth loss
            depth_labels = depth_labels[:, 0, ...]
        depth_loss = self.fullmodel.depth_loss(depth_labels.cuda(), depth_preds)

        self.depth_metrics.update(depth_preds, batch)
        
        self.log('detection_loss', detection_loss)
        self.log('depth_loss', depth_loss)

        return detection_loss + depth_loss

    def on_validation_start(self) -> None:
        self._log_epoch_metrics('train')

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def validation_epoch_end(self, validation_step_outputs):
        self._log_epoch_metrics('val')
        self.get_metrics(validation_step_outputs, len(self.trainer.val_dataloaders[0].dataset))

    def test_epoch_end(self, test_step_outputs):
        self.get_metrics(test_step_outputs, len(self.trainer.test_dataloaders[0].dataset))

    def _log_epoch_metrics(self, prefix: str):
        """
        on_validation_start에서 train 할 때 저장된 metric logging 후 reset
        val 하면서 metric update 하고 val 끝나면 metric logging 후 reset
        """
        depth_metrics = self.depth_metrics.compute()
        for k, v in depth_metrics.items():
            self.log(f'{prefix}/depth_metrics/bin/{k}', v, sync_dist=True)
            
        self.depth_metrics.reset()
    
    
    def configure_optimizers(self, disable_scheduler=False):

        # Define optimizer
        if self.optimizer_args.dual_lr:
            bb_param, depth_param, nbb_param = [], [], []
            bb_keys, depth_keys, nbb_keys = set(), set(), set()

            for k, param in dict(self.fullmodel.named_parameters()).items():
                if any(part in k for part in self.optimizer_args.bb_keywords):
                    bb_param.append(param)
                    bb_keys.add(k)
                elif any(part in k for part in self.optimizer_args.depth_keywords):
                    depth_param.append(param)
                    depth_keys.add(k)
                else:
                    nbb_param.append(param)
                    nbb_keys.add(k)
        
            print('----------------------')
            print(bb_keys)
            print('----------------------')
            print(depth_keys)
            print('----------------------')
            print(nbb_keys)

            opt = torch.optim.AdamW(bb_param, 
                                        lr = self.optimizer_args.lr, 
                                        weight_decay = self.optimizer_args.weight_decay)
            
            opt.add_param_group({'params': nbb_param, 
                                'lr': self.optimizer_args.lr*self.optimizer_args.bb_mult})

            if len(depth_param) != 0:
                opt.add_param_group({'params': depth_param, 
                                'lr': self.optimizer_args.depth_lr})
            

        else:
            opt = torch.optim.AdamW(self.fullmodel.parameters(), 
                                        lr = self.optimizer_args.lr, 
                                        weight_decay = self.optimizer_args.weight_decay)


        # Define LR scheduler
        if self.scheduler_args.name == 'onecycle':
            if self.optimizer_args.dual_lr:
                if len(depth_keys)==0:
                    lr = [self.optimizer_args.lr, self.optimizer_args.lr*self.optimizer_args.bb_mult]
                else:
                    lr = [self.optimizer_args.lr, self.optimizer_args.lr*self.optimizer_args.bb_mult, self.optimizer_args.depth_lr]
            else:
                lr = self.optimizer_args.lr

            sch = torch.optim.lr_scheduler.OneCycleLR(opt, 
                                            max_lr=lr,
                                            total_steps=self.scheduler_args.total_steps,
                                            pct_start=self.scheduler_args.pct_start,
                                            div_factor=self.scheduler_args.div_factor,
                                            cycle_momentum=self.scheduler_args.cycle_momentum,
                                            final_div_factor=self.scheduler_args.final_div_factor)
            return [opt], [{'scheduler': sch, 'interval': 'step'}]

        elif self.scheduler_args.name == 'MultiStep':
            sch = MultiStepLR(opt, self.scheduler_args.down_step)
            return [opt], [{'scheduler': sch, 'interval': 'epoch'}]
            
            
            
        else:
            AssertionError('scheduler is not defined!')
            

    def eval_step(self, batch, batch_idx):

        preds, depth_preds = self(batch)
        
        self.depth_metrics.update(depth_preds, batch)
            
        for img_meta in batch['img_metas']:
            img_meta['box_type_3d'] = LiDARInstance3DBoxes

        if isinstance(self.fullmodel, torch.nn.parallel.DistributedDataParallel):
            results = self.fullmodel.module.get_bboxes(preds, batch['img_metas'])
        else:
            results = self.fullmodel.get_bboxes(preds, batch['img_metas'])
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in results
        ]
        for i in range(len(bbox_results)):
            results[i][0] = bbox_results[i]['boxes_3d']
            results[i][1] = bbox_results[i]['scores_3d']
            results[i][2] = bbox_results[i]['labels_3d']
            results[i].append(batch['img_metas'][i])
        return results

    def get_metrics(self, step_outputs, data_length):
        all_pred_results = list()
        all_img_metas = list()
        for step_output  in step_outputs:
            for i in range(len(step_output)):
                all_pred_results.append(step_output[i][:3])
                all_img_metas.append(step_output[i][3])
        synchronize() 
        print('data_length',data_length)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:data_length]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:data_length]
        if get_rank() == 0:
            self.evaluator.evaluate(lm=self,  results=all_pred_results, img_metas=all_img_metas, logger=self.logger)

    # def get_metrics(self, step_outputs, data_length):
    #     all_pred_results = list()
    #     all_img_metas = list()
    #     # all_img_lidar_calibrated_sensors = list()
    #     # all_img_cam_translation = list()
    #     for step_output  in step_outputs:
    #         for i in range(len(step_output)):   
    #             all_pred_results.append(step_output[i][:3])
    #             all_img_metas.append(step_output[i][3])
    #             # all_img_lidar_calibrated_sensors.append(step_output[i][4])
    #             # all_img_cam_translation.append(step_output[i][5])
    #     synchronize() 
    #     print('data_length',data_length)
    #     all_pred_results = sum(
    #         map(list, zip(*all_gather_object(all_pred_results))),
    #         [])[:data_length]
    #     all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
    #                         [])[:data_length]
    #     # all_img_lidar_calibrated_sensors = sum(map(list, zip(*all_gather_object(all_img_lidar_calibrated_sensors))),
    #     #                     [])[:data_length]
    #     # all_img_cam_translation = sum(map(list, zip(*all_gather_object(all_img_cam_translation))),
    #     #                     [])[:data_length]
    #     if get_rank() == 0:
    #         self.evaluator.evaluate(lm=self, results=all_pred_results, img_metas=all_img_metas, logger=self.logger)
    #         # self.evaluator.evaluate(lm=self, results=all_pred_results, img_metas=all_img_metas, lidar_calibrated = all_img_lidar_calibrated_sensors, logger=self.logger)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        return parent_parser
