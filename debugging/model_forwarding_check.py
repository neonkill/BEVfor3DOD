
'''
This script is used for forwarding test

Author: yelin
Usage:
    
    python script/debugging/model_test.py +experiment=[experiment_name]


'''


import hydra
import torch
from pathlib import Path

from bev_transformer.common import setup_config, setup_network

CONFIG_PATH = '/usr/src/CV_For_Autonomous_Driving/config'
CONFIG_NAME = 'default_config.yaml'


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    # *  setup model  * #
    setup_config(cfg)
    model = setup_network(cfg)
    device = torch.device("cuda")
    model.to(device)
    print(cfg.experiment.save_dir)

    image = torch.randn(1, 6, 3, cfg.data.image.h, cfg.data.image.w, dtype=torch.float).to(device)
    I = torch.randn((1, 6, 3, 3), dtype=torch.float).to(device)
    E = torch.randn((1, 6, 4, 4), dtype=torch.float).to(device)

    dummy_input = {'image': image,
                    'intrinsics': I,
                    'extrinsics': E}


    # *  Full model test  * #
    _ = model(dummy_input)




if __name__ == '__main__':
    main()