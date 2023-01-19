

from pathlib import Path
from functools import partial
import os
import torch
import pytorch_lightning as pl

from data_module.nusc_det_dataset import NuscDetDataset, collate_fn

from . import get_dataset_module_by_name


# def get_split(split):
#     path = Path(__file__).parent / 'splits' /  f'{split}.txt'
#     return path.read_text().strip().split('\n')


#! bev_depth_lss_r50_256x704_128x128_24e_2key의 conf 적용 완료
# H = 900
# W = 1600
# final_dim = (256, 704)
# img_conf = dict(img_mean=[123.675, 116.28, 103.53],
#                 img_std=[58.395, 57.12, 57.375],
#                 to_rgb=True)

# ida_aug_conf = {
#     'resize_lim': (0.386, 0.55),
#     'final_dim':
#     final_dim,
#     'rot_lim': (-5.4, 5.4),
#     'H':
#     H,
#     'W':
#     W,
#     'rand_flip':
#     True,
#     'bot_pct_lim': (0.0, 0.0),
#     'cams': [
#         'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
#         'CAM_BACK', 'CAM_BACK_RIGHT'
#     ],
#     'Ncams':
#     6,
# }

# bda_aug_conf = {
#     'rot_lim': (-22.5, 22.5),
#     'scale_lim': (0.95, 1.05),
#     'flip_dx_ratio': 0.5,
#     'flip_dy_ratio': 0.5
# }

# CLASSES = [
#     'car',
#     'truck',
#     'construction_vehicle',
#     'bus',
#     'trailer',
#     'barrier',
#     'motorcycle',
#     'bicycle',
#     'pedestrian',
#     'traffic_cone',
# ]


# class_names = CLASSES
# data_root='/usr/src/nuscenes'
# train_info_paths = os.path.join(data_root,'nuscenes_infos_train.pkl')
# val_info_paths = os.path.join(data_root,'nuscenes_infos_val.pkl')
# data_use_cbgs = False
# num_sweeps = 0
# sweep_idxes = list()
# key_idxes = list()
# data_return_depth = True
# use_fusion = False
# batch_size_per_device=4
# predict_info_paths = os.path.join(data_root,'nuscenes_infos_test.pkl')
'''
* dataset 공통 args
ida_aug_conf
bda_aug_conf
classes
data_root
num_sweeps
sweep_idxes
key_idxes
use_fusion
img_conf

* datalodaer 공통 args
batch_size
num_workers
shuffle -> arg로 해결
sampler
'''

class DataModule(pl.LightningDataModule):

    def __init__(self, dataset, data_cfg, loader_cfg):
        super().__init__()

        self.get_data = get_dataset_module_by_name(dataset).get_data
        self.data_cfg = data_cfg
        self.loader_cfg = loader_cfg


    def get_split(self, split, shuffle):
        self.dataset, return_depth, drop_last = self.get_data(split, **self.data_cfg)

        self.loader_config = dict(self.loader_cfg )

        if self.loader_config['num_workers'] == 0:
            self.loader_config['prefetch_factor'] = 2

         #! return torch.utils.data.DataLoader(dataset, shuffle=shuffle, collate_fn=collate_fn, **loader_config)

        return torch.utils.data.DataLoader(self.dataset,
                            drop_last=drop_last,
                            shuffle=shuffle,
                            collate_fn=partial(collate_fn,
                                    is_return_depth= return_depth),
                            **self.loader_config)
 

    def train_dataloader(self, shuffle=False):
        return self.get_split('train', shuffle=shuffle)

    def val_dataloader(self, shuffle=False):
        return self.get_split('val', shuffle=shuffle)

    def test_dataloader(self, shuffle=False):
        return self.get_split('val', shuffle=shuffle)

    # def predict_dataloader(self,shuffle=False):
    #     predict_dataset = NuscDetDataset(
    #                                     # ida_aug_conf=ida_aug_conf,
    #                                     #  bda_aug_conf=bda_aug_conf,
    #                                     #  classes=class_names,
    #                                     #  data_root=data_root,
    #                                      info_paths=predict_info_paths,
    #                                      is_train=False,
    #                                     #  img_conf=img_conf,
    #                                     #  num_sweeps=num_sweeps,
    #                                     #  sweep_idxes=sweep_idxes,
    #                                     #  key_idxes=key_idxes,
    #                                      return_depth=use_fusion,
    #                                     #  use_fusion=use_fusion
    #                                     )
    #     predict_loader = torch.utils.data.DataLoader(
    #         predict_dataset,
    #         shuffle=shuffle,
    #         collate_fn=partial(collate_fn, is_return_depth=use_fusion),
    #         **self.loader_config
    #     )
    #     return predict_loader

