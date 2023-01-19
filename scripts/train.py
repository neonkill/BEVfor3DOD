from pathlib import Path


import logging
from torchinfo import summary
import pytorch_lightning as pl
import hydra

from pytorch_lightning.plugins import DDPPlugin
# from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from utils import setup_config, setup_experiment, load_backbone


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
    # trainer = pl.Trainer(logger=logger,
    #                      callbacks=callbacks,
    #                      strategy=DDPStrategy(find_unused_parameters=False),     #! find_unsued_parameters False -> True
    #                      **cfg.trainer)
    trainer = pl.Trainer(logger=logger,
                         callbacks=callbacks,
                         strategy=DDPPlugin(find_unused_parameters=False),     #! find_unsued_parameters False -> True
                         **cfg.trainer)
    trainer.fit(model_module, datamodule=data_module, ckpt_path=ckpt_path)
    # trainer.fit(model_module)


if __name__ == '__main__':
    main()
