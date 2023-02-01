# -----------------------------------------------------------------------
# Copyright (C) 2020 Brady Zhou
# Released under the MIT license
# https://github.com/bradyz/cross_view_transformers/blob/master/LICENSE
# -----------------------------------------------------------------------

from tokenize import group
import torch

from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig,SCMode
from torchmetrics import MetricCollection
from pathlib import Path
import copy
# from model_module.losses import MultipleLoss
from model_module.lightining_model_module import ModelModule
from data_module.lightning_data_module import DataModule


from collections.abc import Callable
from typing import Tuple, Dict, Optional



def setup_config(cfg: DictConfig, override: Optional[Callable] = None):
    OmegaConf.set_struct(cfg, False)

    if override is not None:
        override(cfg)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, True)

    save_dir = Path(cfg.experiment.save_dir)
    save_dir.mkdir(parents=False, exist_ok=True)


def setup_network(cfg: DictConfig):
    model_config = OmegaConf.to_container(cfg.model,structured_config_mode=SCMode.DICT)
    # print('ttt', type(model_config['backbone']['img_backbone_conf']))
    # print('ttt',type(OmegaConf.to_container(cfg.model, resolve=True)['head']))
    
    return instantiate(model_config)

def setup_compute_groups(cfg: DictConfig):
    # return [['road_iou'], ['lane_iou'], ['vehicle_iou']]
    groups = []
    for k, _ in instantiate(cfg.metrics).items():
        groups.append([k])

    return groups

def setup_model_module(cfg: DictConfig) -> ModelModule:
    fullmodel = setup_network(cfg) 
    
    model_module = ModelModule(fullmodel ,optimizer_args=cfg.optimizer ,scheduler_args=cfg.scheduler ,cfg=cfg) 

    return model_module



def setup_data_module(cfg: DictConfig) -> DataModule:
    return DataModule(cfg.data.dataset, cfg.data, cfg.loader)
    # return DataModule(cfg.data, cfg.loader)


def setup_experiment(cfg: DictConfig) -> Tuple[ModelModule, DataModule, Callable]:
    # model_module = setup_BEVDEPTH_model_module(cfg)
    model_module = setup_model_module(cfg)
    data_module = setup_data_module(cfg)

    return model_module, data_module


def load_backbone(checkpoint_path: str, prefix: str = 'backbone'):
    checkpoint = torch.load(checkpoint_path)

    cfg = DictConfig(checkpoint['hyper_parameters'])

    cfg = OmegaConf.to_object(checkpoint['hyper_parameters'])
    cfg = DictConfig(cfg)

    state_dict = remove_prefix(checkpoint['state_dict'], prefix)

    backbone = setup_network(cfg)
    backbone.load_state_dict(state_dict)

    return backbone


def remove_prefix(state_dict: Dict, prefix: str) -> Dict:
    result = dict()

    for k, v in state_dict.items():
        tokens = k.split('.')

        if tokens[0] == prefix:
            tokens = tokens[1:]

        key = '.'.join(tokens)
        result[key] = v

    return result
