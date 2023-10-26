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
from model_module.losses import MultipleLoss
from model_module.lightining_model_module_seg import ModelModule
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

# def setup_model_module(cfg: DictConfig) -> ModelModule:
#     fullmodel = setup_network(cfg) 
    
#     model_module = ModelModule(fullmodel ,optimizer_args=cfg.optimizer ,scheduler_args=cfg.scheduler ,cfg=cfg) 

#     return model_module

def setup_model_module(cfg: DictConfig) -> ModelModule:
    backbone = setup_network(cfg)
    loss_func = MultipleLoss(instantiate(cfg.loss))
    metrics = MetricCollection({k: v for k, v in instantiate(cfg.metrics).items()},compute_groups=setup_compute_groups(cfg))
    # metrics = MetricCollection({k: v for k, v in instantiate(cfg.metrics).items()})
    
    model_module = ModelModule(backbone, loss_func, metrics,
                               cfg.optimizer, cfg.scheduler,
                               cfg=cfg)

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
    # backbone.load_state_dict(state_dict)
    load_state_dict(backbone, state_dict, prefix='')

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