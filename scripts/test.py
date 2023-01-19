


import logging
from pathlib import Path

import torch
import hydra
import pytorch_lightning as pl

from utils import setup_config, setup_experiment, remove_prefix


log = logging.getLogger(__name__)

CONFIG_PATH = '/usr/src/CV_For_Autonomous_Driving/config'
CONFIG_NAME = 'default_config.yaml'


def resume_training(experiment):
    save_dir = Path(experiment.save_dir).resolve()
    checkpoints = list(save_dir.glob(f'**/{experiment.uuid}/checkpoints/*.ckpt'))

    print(f'**/{experiment.uuid}/checkpoints/*.ckpt')
    log.info(f'Searching {save_dir}.')

    if not checkpoints:
        return None

    log.info(f'Found {checkpoints[-1]}.')

    return checkpoints[-1]


# hydra.main decorator를 사용해 experiment cfg 생성
@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):

    setup_config(cfg)

    pl.seed_everything(cfg.experiment.seed, workers=True)
    Path(cfg.experiment.save_dir).mkdir(exist_ok=True, parents=False)

    # Logger
    logger = pl.loggers.WandbLogger(project=cfg.experiment.project,
                                    save_dir=cfg.experiment.save_dir,
                                    id=cfg.experiment.uuid)
    
    # Create and load model/data
    model_module, data_module = setup_experiment(cfg)
    
    # eval_loader = data_module.eval_dataloader()

    # print(len(data_module.test_dataloader()))
    # exit()

    # load model
    ckpt_path = resume_training(cfg.experiment)
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        state_dict = remove_prefix(ckpt['state_dict'], 'fullmodel')
        model_module.fullmodel.load_state_dict(state_dict)
        
    model_module.to('cuda')

    print('loaded pretrained network! Start Evaluation!')
    # print(len(eval_loader))

    # evaluation (pl version)
    #! For Debug only 5 batch
    # trainer = pl.Trainer(logger=logger, accelerator='gpu', gpus=[0],fast_dev_run=True)
    trainer = pl.Trainer(logger=logger, accelerator='gpu', gpus=[0])
    # trainer = pl.Trainer(logger=logger, accelerator='gpu', gpus=[0])
    # trainer.test(model=model_module)
    # trainer.test(model=model_module,
    #             dataloaders=eval_loader)

    trainer.test(model=model_module, datamodule=data_module)



if __name__ == '__main__':
    main()
