# -----------------------------------------------------------------------
# Copyright (C) 2020 Brady Zhou
# Released under the MIT license
# https://github.com/bradyz/cross_view_transformers/blob/master/LICENSE
# -----------------------------------------------------------------------
# Modified by yelin2
# -----------------------------------------------------------------------


import torch
import pytorch_lightning as pl


class ModelModule(pl.LightningModule):
    def __init__(self, fullmodel, loss_func, metrics, optimizer_args, scheduler_args=None, cfg=None):
        super().__init__()

        self.save_hyperparameters(
            cfg,
            ignore=['fullmodel', 'loss_func', 'metrics', 'optimizer_args', 'scheduler_args'])

        self.fullmodel = fullmodel
        self.loss_func = loss_func
        self.metrics = metrics

        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args

    def forward(self, batch):
        return self.fullmodel(batch)

    def shared_step(self, batch, prefix='', on_step=False, return_output=True):
        pred = self(batch)
        loss, loss_details = self.loss_func(pred, batch)

        self.metrics.update(pred, batch)

        if self.trainer is not None:
            self.log(f'{prefix}/total_loss', loss.detach(), on_step=on_step, on_epoch=True)
            self.log_dict({f'{prefix}/loss/{k}': v.detach() for k, v in loss_details.items()}, on_step=on_step, on_epoch=True)

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

    def _log_epoch_metrics(self, prefix: str):
        """
        on_validation_start에서 train 할 때 저장된 metric logging 후 reset
        val 하면서 metric update 하고 val 끝나면 metric logging 후 reset
        """

        metrics = self.metrics.compute()

        for key, value in metrics.items():
            if isinstance(value, dict):
                for subkey, val in value.items():
                    self.log(f'{prefix}/metrics/{key}{subkey}', val)
            else:
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

        # Define optimizer
        opt = torch.optim.AdamW(self.fullmodel.parameters(), 
                                    lr = self.optimizer_args.lr, 
                                    weight_decay = self.optimizer_args.weight_decay)


        # Define LR scheduler
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 
                                        max_lr=self.optimizer_args.lr,
                                        total_steps=self.scheduler_args.total_steps,
                                        pct_start=self.scheduler_args.pct_start,
                                        div_factor=self.scheduler_args.div_factor,
                                        cycle_momentum=self.scheduler_args.cycle_momentum,
                                        final_div_factor=self.scheduler_args.final_div_factor)

        return [opt], [{'scheduler': sch, 'interval': 'step'}]
