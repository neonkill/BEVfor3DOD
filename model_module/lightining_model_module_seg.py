# -----------------------------------------------------------------------
# Copyright (C) 2020 Brady Zhou
# Released under the MIT license
# https://github.com/bradyz/cross_view_transformers/blob/master/LICENSE
# -----------------------------------------------------------------------
# Modified by neonkill
# -----------------------------------------------------------------------


import torch
import pytorch_lightning as pl
from .metrics import DepthMetrics, SegMetrics
from model_module.torch_dist import all_gather_object, get_rank, synchronize
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from model_module.det_evaluators_detr import DetNuscEvaluator
from mmdet3d.core import bbox3d2result
import mmcv

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
    def __init__(self, fullmodel, loss_func, metrics, optimizer_args, scheduler_args=None, cfg=None):
        super().__init__()

        self.save_hyperparameters(
            cfg,
            ignore=['fullmodel', 'loss_func', 'metrics', 'optimizer_args', 'scheduler_args'])

        self.fullmodel = fullmodel
        self.loss_func = loss_func
        self.metrics = metrics
        self.depth_metrics = DepthMetrics()
        self.num_cls = 13
        self.seg_metrics = SegMetrics(self.num_cls)
        
        default_root_dir = './outputs/'
        mmcv.mkdir_or_exist(default_root_dir)
        self.default_root_dir = default_root_dir
        self.evaluator = DetNuscEvaluator(class_names=CLASSES,
                                          output_dir=self.default_root_dir)


        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args
        self.depth_compute = 0
        self.seg_compute = 0
        
    def forward(self, batch):
        return self.fullmodel(batch)

    def shared_step(self, batch, prefix='', on_step=False, return_output=True):
        pred = self(batch)
        # print(pred.keys())
        loss, loss_details = self.loss_func(pred, batch)
        # print(batch.keys())
        # exit()

        #! 
        self.metrics.update(pred, batch)
        if 'depth_bin' in pred.keys():
            self.depth_compute = 1
            self.depth_metrics.update(pred, batch)
        if 'seg_logits' in pred.keys():
            self.seg_compute = 1
            self.seg_metrics.update(pred, batch)
        # for k, v in self.metrics.items():
        #     v.update(pred, batch)
        if prefix == 'val':
            for img_meta in batch['img_metas']:
                img_meta['box_type_3d'] = LiDARInstance3DBoxes
                
            results = self.loss_func['detection'].get_bboxes(pred['det_pred'], batch['img_metas'])
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in results
            ]
            for i in range(len(bbox_results)):
                results[i][0] = bbox_results[i]['boxes_3d']
                results[i][1] = bbox_results[i]['scores_3d']
                results[i][2] = bbox_results[i]['labels_3d']
                results[i].append(batch['img_metas'][i])
                results[i].append(batch['lidar_calibrated_sensors'][i])
            return results

        if self.trainer is not None:
            self.log(f'{prefix}/total_loss', loss.detach(), on_step=on_step, on_epoch=True, sync_dist=True)
            self.log_dict({f'{prefix}/loss/{k}': v.detach() for k, v in loss_details.items()}, on_step=on_step, on_epoch=True, sync_dist=True)

        # self.backward(loss, retain_graph=True)
        # Used for visualizations
        if return_output:
            return {'loss': loss, 'batch': batch, 'pred': pred}

        return {'loss': loss}

    def training_step(self, batch, batch_idx):
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
        
    def test_epoch_end(self, outputs):
        self._log_epoch_metrics('test')

    def on_validation_start(self) -> None:
        self._log_epoch_metrics('train')
        self._enable_dataloader_shuffle(self.trainer.val_dataloaders)

    def validation_epoch_end(self, outputs):
        self._log_epoch_metrics('val')
        self.get_metrics(outputs, len(self.trainer.val_dataloaders[0].dataset))

    # def on_backward(self, use_amp, loss, opt):
    #     loss.backward(retain_graph=True)

    # def backward(self, loss, *args, **kwargs) -> None:
    #     if self._fabric:
    #         self._fabric.backward(loss, *args, **kwargs)
    #     else:
    #         loss.backward(*args, **kwargs)

    def _log_epoch_metrics(self, prefix: str):
        """
        on_validation_start에서 train 할 때 저장된 metric logging 후 reset
        val 하면서 metric update 하고 val 끝나면 metric logging 후 reset
        """
        metrics = self.metrics.compute()
        if self.depth_compute:
            depth_metrics = self.depth_metrics.compute()
            for k, v in depth_metrics.items():
                self.log(f'{prefix}/metrics/bin/{k}', v, sync_dist=True)
                
        if self.seg_compute:
            seg_metrics = self.seg_metrics.compute()
            for k, v in seg_metrics.items():
                self.log(f'{prefix}/metrics/bin/{k}', v, sync_dist=True)
            
        for key, value in metrics.items():
            if len(value)>1:
                print(f'{prefix}/metrics/{key}{value[0][0]:.1f}', value[0][1])
                print(f'{prefix}/metrics/{key}{value[1][0]:.1f}', value[1][1])
                self.log(f'{prefix}/metrics/{key}{value[0][0]:.1f}', value[0][1], sync_dist=True)
                self.log(f'{prefix}/metrics/{key}{value[1][0]:.1f}', value[1][1], sync_dist=True)
            else:
                self.log(f'{prefix}/metrics/{key}', value[0][1], sync_dist=True)

        self.metrics.reset()
        self.depth_metrics.reset()

    def _enable_dataloader_shuffle(self, dataloaders):
        """
        HACK for https://github.com/PyTorchLightning/pytorch-lightning/issues/11054
        """
        for v in dataloaders:
            v.sampler.shuffle = False
            v.sampler.set_epoch(self.current_epoch)
            
    def get_metrics(self, step_outputs, data_length):
        all_pred_results = list()
        all_img_metas = list()
        all_lidar_cali = list()
        for step_output  in step_outputs:
            for i in range(len(step_output)):
                all_pred_results.append(step_output[i][:3])
                all_img_metas.append(step_output[i][3])
                all_lidar_cali.append(step_output[i][4])
        synchronize() 
        print('data_length',data_length)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:data_length]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:data_length]
        all_lidar_cali = sum(map(list, zip(*all_gather_object(all_lidar_cali))),
                    [])[:data_length]
        if get_rank() == 0:
            self.evaluator.evaluate(lm=self,  results=all_pred_results, img_metas=all_img_metas, lidar_cali=all_lidar_cali, logger=self.logger)

    def configure_optimizers(self, disable_scheduler=False):

        # Define optimizer
        if self.optimizer_args.dual_lr:
            bb_param, depth_param, seg_param, detection_param, nbb_param = [], [], [], [], []
            bb_keys, depth_keys, seg_keys, detection_keys, nbb_keys = set(), set(), set(), set(), set()

            for k, param in dict(self.fullmodel.named_parameters()).items():
                if any(part in k for part in self.optimizer_args.bb_keywords):
                    bb_param.append(param)
                    bb_keys.add(k)
                elif any(part in k for part in self.optimizer_args.depth_keywords):
                    depth_param.append(param)
                    depth_keys.add(k)
                elif any(part in k for part in self.optimizer_args.detection_keywords):
                    detection_param.append(param)
                    detection_keys.add(k)
                elif any(part in k for part in self.optimizer_args.seg_keywords):
                    seg_param.append(param)
                    seg_keys.add(k)
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
                                        weight_decay = self.optimizer_args.weight_decay,
                                         amsgrad=True)
            
            opt.add_param_group({'params': nbb_param, 
                                'lr': self.optimizer_args.lr*self.optimizer_args.bb_mult})

            if len(depth_param) != 0:
                opt.add_param_group({'params': depth_param, 
                                'lr': self.optimizer_args.depth_lr})
                
            if len(detection_param) != 0:
                opt.add_param_group({'params': detection_param, 
                                'lr': self.optimizer_args.detection_lr})
            if len(seg_param) != 0:
                opt.add_param_group({'params': seg_param, 
                                'lr': self.optimizer_args.seg_lr})
            

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
                    # lr = [self.optimizer_args.lr, self.optimizer_args.lr*self.optimizer_args.bb_mult, self.optimizer_args.depth_lr, self.optimizer_args.seg_lr]
                    lr = [self.optimizer_args.lr, self.optimizer_args.lr*self.optimizer_args.bb_mult, self.optimizer_args.depth_lr, self.optimizer_args.detection_lr, self.optimizer_args.seg_lr]
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


        elif self.scheduler_args.name == 'cosannealing':

            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 
                                                        T_max=self.scheduler_args.T_max,
                                                        eta_min=self.scheduler_args.eta_min)

            return [opt], [{'scheduler': sch, 'interval': 'step'}]

        
        else:
            AssertionError('scheduler is not defined!')
            


            

            
