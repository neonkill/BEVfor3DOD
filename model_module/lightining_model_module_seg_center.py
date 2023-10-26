# -----------------------------------------------------------------------
# Copyright (C) 2020 Brady Zhou
# Released under the MIT license
# https://github.com/bradyz/cross_view_transformers/blob/master/LICENSE
# -----------------------------------------------------------------------
# Modified by neonkill
# -----------------------------------------------------------------------


import torch
import pytorch_lightning as pl
from .metrics import DepthMetrics
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from model_module.det_evaluators_center import DetNuscEvaluator
from model_module.torch_dist import all_gather_object, get_rank, synchronize
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

        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args
        self.class_names = CLASSES 
        self.depth_compute = 0
        default_root_dir = './outputs/'
        mmcv.mkdir_or_exist(default_root_dir)
        self.default_root_dir = default_root_dir
        self.evaluator = DetNuscEvaluator(class_names=CLASSES,
                                          output_dir=self.default_root_dir)
        
    def forward(self, batch):
        return self.fullmodel(batch)

    def shared_step(self, batch, prefix='', on_step=False, return_output=True):
        pred = self(batch)

        loss, loss_details = self.loss_func(pred, batch)
        # print(batch.keys())
        # exit()
        
        gt_boxes = [gt_box.cuda() for gt_box in batch['gt_boxes']]
        gt_labels = [gt_label.cuda() for gt_label in batch['gt_labels']]
        
        if isinstance(self.fullmodel, torch.nn.parallel.DistributedDataParallel): #! 지워도?
            targets = self.fullmodel.module.det_head.get_targets(gt_boxes, gt_labels)
            detection_loss = self.fullmodel.module.det_head.loss(targets, pred['det_pred'])
        else:
            targets = self.fullmodel.det_head.get_targets(gt_boxes, gt_labels)
            detection_loss = self.fullmodel.det_head.loss(targets, pred['det_pred'])        
            
        self.metrics.update(pred, batch)
        if 'depth_bin' in pred.keys():
            self.depth_compute = 1
            self.depth_metrics.update(pred, batch)

        if prefix == 'val':
            for img_meta in batch['img_metas']:
                img_meta['box_type_3d'] = LiDARInstance3DBoxes
                
            if isinstance(self.fullmodel, torch.nn.parallel.DistributedDataParallel):
                results = self.fullmodel.module.det_head.get_bboxes(pred['det_pred'], batch['img_metas'])
            else:
                results = self.fullmodel.det_head.get_bboxes(pred['det_pred'], batch['img_metas'])
            for i in range(len(results)):
                results[i][0] = results[i][0].detach().cpu().numpy()
                results[i][1] = results[i][1].detach().cpu().numpy()
                results[i][2] = results[i][2].detach().cpu().numpy()
                results[i].append(batch['img_metas'][i])
            return results

        loss_details['detection_loss'] = detection_loss 
        loss = loss + detection_loss* 0.005
        #! 

        # for k, v in self.metrics.items():
            # v.update(pred, batch)

        if self.trainer is not None:
            self.log(f'{prefix}/total_loss', loss.detach(), on_step=on_step, on_epoch=True, sync_dist=True)
            self.log_dict({f'{prefix}/loss/{k}': v.detach() for k, v in loss_details.items()}, on_step=on_step, on_epoch=True, sync_dist=True)
            # self.log(f'{prefix}/loss/detection_loss', detection_loss.detach(), on_step=on_step, on_epoch=True, sync_dist=True)

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
            v.sampler.shuffle = True
            v.sampler.set_epoch(self.current_epoch)
            
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
                elif any(part in k for part in self.optimizer_args.seg_keywords):
                    seg_param.append(param)
                    seg_keys.add(k)
                elif any(part in k for part in self.optimizer_args.detection_keywords):
                    detection_param.append(param)
                    detection_keys.add(k)
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
                
            if len(seg_param) != 0:
                opt.add_param_group({'params': seg_param, 
                                'lr': self.optimizer_args.seg_lr})
                
            if len(detection_param) != 0:
                opt.add_param_group({'params': detection_param, 
                                'lr': self.optimizer_args.detection_lr})
                
            

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
                    lr = [self.optimizer_args.lr, self.optimizer_args.lr*self.optimizer_args.bb_mult, self.optimizer_args.depth_lr, self.optimizer_args.seg_lr, self.optimizer_args.detection_lr]
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
            


            

            
