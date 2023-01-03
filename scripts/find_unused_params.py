'''
This script is used for find unused parameters in training phase

Author: yelin
Usage:
    python script/debugging/find_unused_params.py +experiment=[experiment_name]

'''



import torch

import hydra
from pathlib import Path
from utils import setup_config, setup_data_module, setup_model_module


CONFIG_PATH = '/usr/src/CV_For_Autonomous_Driving/config'
CONFIG_NAME = 'default_config.yaml'



@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    setup_config(cfg)

    data = setup_data_module(cfg)
    model = setup_model_module(cfg)
    device = torch.device("cuda")
    model.to(device)

    loader = data.train_dataloader()
    sample = next(iter(loader))

    for k, v in sample.items():
        if k == 'gt_box' or k == 'gt_label':
            continue
        sample[k] = v.cuda()

    out = model.training_step(sample, 0)

    out['loss'].backward()

    for name, param in model.named_parameters():
        if param.grad is None:
            print(name)


if __name__ == '__main__':
    main()