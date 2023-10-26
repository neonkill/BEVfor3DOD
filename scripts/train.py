from pathlib import Path
from torch.utils.data import SequentialSampler
import torch

import logging
from torchinfo import summary
import pytorch_lightning as pl
import hydra
import os
# from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from utils import setup_config, setup_experiment, load_backbone
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

log = logging.getLogger(__name__)

# CONFIG_PATH = Path.cwd() / 'config'
CONFIG_PATH = '/usr/src/CV_For_Autonomous_Driving/config'
CONFIG_NAME = 'default_config.yaml'


def maybe_resume_training(experiment):
    save_dir = Path(experiment.save_dir).resolve()
    checkpoints = list(save_dir.glob(f'**/{experiment.uuid}/checkpoints/*.ckpt'))

    log.info(f'Searching {save_dir}.')

    if not checkpoints:
        return None

    log.info(f'Found {checkpoints[-1]}.')

    return checkpoints[-1]

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    # print(cfg)
    setup_config(cfg)

    pl.seed_everything(cfg.experiment.seed, workers=True)

    Path(cfg.experiment.save_dir).mkdir(exist_ok=True, parents=False)


    # Create and load model/data
    model_module, data_module = setup_experiment(cfg)

    # Optionally load model
    ckpt_path = maybe_resume_training(cfg.experiment)
    
    ckpt = torch.load('/usr/src/nuscenes/no_aux_10obj_binary_model.ckpt', map_location='cpu')

    state_dict = ckpt['state_dict']
    load_state_dict(model_module.fullmodel, state_dict, prefix='fullmodel.')
    
    if ckpt_path is not None:
        model_module.fullmodel = load_backbone(ckpt_path, prefix='fullmodel')


    # Loggers and callbacks
    # logger = None
    logger = pl.loggers.WandbLogger(project=cfg.experiment.project,
                                    save_dir=cfg.experiment.save_dir,
                                    name=cfg.experiment.name,
                                    id=cfg.experiment.uuid)

    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(filename='model',
                        every_n_train_steps=cfg.experiment.checkpoint_interval),
    ]

    # Train
    # summary(model_module.backbone)
    trainer = pl.Trainer(logger=logger,
                         callbacks=callbacks,
                         strategy=DDPStrategy(find_unused_parameters=True),     #! find_unsued_parameters False -> True
                        #  replace_sampler_ddp=False, 
                         **cfg.trainer)
    trainer.fit(model_module, datamodule=data_module, ckpt_path=ckpt_path)
    # trainer.fit(model_module)


if __name__ == '__main__':
    main()
