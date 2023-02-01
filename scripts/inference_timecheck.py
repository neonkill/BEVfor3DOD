
'''
measure models inference time
'''

import hydra
import torch
import numpy as np
from tqdm import tqdm
# from pathlib import Path

from utils import setup_config, setup_network

CONFIG_PATH = '/usr/src/CV_For_Autonomous_Driving/config'
CONFIG_NAME = 'default_config.yaml'


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    setup_config(cfg)
    model = setup_network(cfg)
    device = torch.device("cuda")
    model.to(device)

    # h = cfg.data
    # image = torch.randn(1, 6, 3, cfg.data.image.h, cfg.data.image.w, dtype=torch.float).to(device)
    # I = torch.randn((1, 6, 3, 3), dtype=torch.float).to(device)
    # E = torch.randn((1, 6, 4, 4), dtype=torch.float).to(device)

    # dummy_input = {'image': image,
    #                 'intrinsics': I,
    #                 'extrinsics': E}

    image = torch.randn(1, 6, 3, cfg.data.image.h, cfg.data.image.w,dtype=torch.float).to(device)
    depth = torch.randn(1, 6, cfg.data.image.h, cfg.data.image.w, dtype=torch.float).to(device)
    I = torch.randn((1, 6, 3, 3), dtype=torch.float).to(device)
    E = torch.randn((1, 6, 4, 4), dtype=torch.float).to(device)
    sensor2sensor_mats = torch.randn((1, 6, 4, 4), dtype=torch.float).to(device)
    sensor2ego_mats = torch.randn((1, 6, 4, 4), dtype=torch.float).to(device)
    ida_mats = torch.randn((1, 6, 4, 4), dtype=torch.float).to(device)
    bda_mats = torch.randn((1, 4, 4), dtype=torch.float).to(device)

    dummy_input = {'image' : image,
                   'depths' : depth,
                    'sensor2ego_mats': sensor2ego_mats,
                    'intrinsics': I,
                    'extrinsics': E,
                    'ida_mats': ida_mats,
                    'sensor2sensor_mats': sensor2sensor_mats,
                    'bda_mats': bda_mats
                    }


    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1000
    timings=np.zeros((repetitions,1))

    #GPU-WARM-UP
    for _ in range(100):
        _ = model(dummy_input)

    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in tqdm(range(repetitions)):

            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            time = starter.elapsed_time(ender)
            timings[rep] = time

    bb = np.sum(timings) / repetitions

    print(f'inference time: {bb:.2f} ms  {1000/bb:.2f} fps')

if __name__ == '__main__':
    main()
