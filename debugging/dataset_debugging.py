
import hydra
from typing import Optional
from collections.abc import Callable
from omegaconf import OmegaConf, DictConfig

from data_module.lightning_data_module import DataModule


def setup_config(cfg: DictConfig, override: Optional[Callable] = None):

    OmegaConf.set_struct(cfg, False)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, True)


def test_get_datasets(data_module):
    
    print(data_module.data_cfg)
    print(data_module.loader_cfg)

    split = 'train'
    data_cfg = data_module.data_cfg
    datasets = data_module.get_datasets(split=split,
                                        **data_module.data_cfg)

    print(f'In {split} dataset, {len(datasets)} scenes are included.')
    



CONFIG_PATH = '/usr/src/CV_For_Autonomous_Driving/config'
CONFIG_NAME = 'default_config.yaml'

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    
    setup_config(cfg)

    DM = DataModule(cfg.data, cfg.loader)

    split = 'train'
    data_cfg = DM.data_cfg
    datasets = DM.get_datasets(split=split,
                                **data_cfg)

    print(f'In {split} dataset, {len(datasets)} scenes are included.')


    total = 0
    for dataset in datasets:
        total += len(dataset)

    print(f'nuscenes train has {total} data samples')

if __name__ == '__main__':
    main()