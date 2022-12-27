

from pathlib import Path

import torch
import pytorch_lightning as pl

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

# from data_module.dataset.nuscenes_dataset import NuScenesDataset
from data_module.dataset.nuse_det_test import NuScenesDataset
# from data_module.dataset.nusc_det_dataset import NuScenesDataset

def get_split(split):
    path = Path(__file__).parent / 'dataset' / 'splits' / f'{split}.txt'
    return path.read_text().strip().split('\n')


class DataModule(pl.LightningDataModule):

    def __init__(self, data_cfg, loader_cfg):
        super().__init__()

        self.data_cfg = data_cfg
        self.loader_cfg = loader_cfg


    # 각 scene들의 dataset을 만들고, 이 dataset들의 list 반환
    def get_datasets(self,
                    dataset_dir,
                    version='v1.0-trainval',
                    split='train',
                    **kwargs
                    ):

        nusc = NuScenes(version=version, dataroot=dataset_dir)
        split_scenes = get_split(split)

        datasets = []
        for scene_record in nusc.scene:

            scene_name = scene_record['name']
            if scene_name not in split_scenes:
                continue
            
            map_name = nusc.get('log', scene_record['log_token'])['location']
            nusc_map = NuScenesMap(dataroot=dataset_dir, map_name=map_name)


            dataset = NuScenesDataset(nusc, 
                                        nusc_map, 
                                        dataset_dir,
                                        scene_name, 
                                        scene_record, 
                                        **kwargs)
            datasets.append(dataset)

        return datasets


    def get_loader(self, split, shuffle):
        
        datasets = self.get_datasets(split=split, **self.data_cfg)
        dataset = torch.utils.data.ConcatDataset(datasets)

        loader_config = dict(self.loader_cfg)

        if loader_config['num_workers'] == 0:
            loader_config['prefetch_factor'] = 2

        return torch.utils.data.DataLoader(dataset, shuffle=shuffle, **loader_config)


    def train_dataloader(self, shuffle=True):
        return self.get_loader('train', shuffle=shuffle)

    def val_dataloader(self, shuffle=True):
        return self.get_loader('val', shuffle=shuffle)

    def eval_dataloader(self, shuffle=False):
        return self.get_loader('eval', shuffle=shuffle)
    

    def get_dataset_test(self,
                    dataset_dir,
                    version='v1.0-trainval',
                    split='train',
                    **kwargs
                    ):

        nusc = NuScenes(version=version, dataroot=dataset_dir)
        split_scenes = get_split(split)

        datasets = []
        for scene_record in nusc.scene:

            scene_name = scene_record['name']
            if scene_name not in split_scenes:
                continue
            
            map_name = nusc.get('log', scene_record['log_token'])['location']
            nusc_map = NuScenesMap(dataroot=dataset_dir, map_name=map_name)


            # datasets.append(1)
            dataset = NuScenesDataset(nusc, 
                                        nusc_map, 
                                        dataset_dir,
                                        scene_name, 
                                        scene_record, 
                                        **kwargs)

            return dataset


    def get_dataset_bevdepth(self,
                    dataset_dir,
                    version='v1.0-trainval',
                    split='train',
                    **kwargs
                    ):

        H = 900
        W = 1600
        final_dim = (256, 704)
        img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                img_std=[58.395, 57.12, 57.375],
                to_rgb=True)

        ida_aug_conf = {
        'resize_lim': (0.386, 0.55),
        'final_dim':
        final_dim,
        'rot_lim': (-5.4, 5.4),
        'H':
        H,
        'W':
        W,
        'rand_flip':
        True,
        'bot_pct_lim': (0.0, 0.0),
        'cams': [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
            'CAM_BACK', 'CAM_BACK_RIGHT'
        ],
        'Ncams':
        6,
        }

        bda_aug_conf = {
        'rot_lim': (-22.5, 22.5),
        'scale_lim': (0.95, 1.05),
        'flip_dx_ratio': 0.5,
        'flip_dy_ratio':
         0.5
        }
        CLASSES = [
        'car',
        'truck',
        'construction_vehicle',
        'bus',
        'trailer',
        'barrier',
        'motorcycle',
        'bicycle',
        'pedestrian',
        'traffic_cone',
        ]

        nusc = NuScenes(version=version, dataroot=dataset_dir)
        split_scenes = get_split(split)

        datasets = []
        for scene_record in nusc.scene:

            scene_name = scene_record['name']
            if scene_name not in split_scenes:
                continue
            
            map_name = nusc.get('log', scene_record['log_token'])['location']
            nusc_map = NuScenesMap(dataroot=dataset_dir, map_name=map_name)


            # datasets.append(1)
            train_dataset = NuscDetDataset(ida_aug_conf=ida_aug_conf,
                                       bda_aug_conf=bda_aug_conf,
                                       classes=CLASSES,
                                       data_root=dataset_dir,
                                       info_paths='/usr/src/CV_For_Autonomous_Driving/BEVDepth/scripts/data/nuScenes/nuscenes_infos_train.pkl',
                                       is_train=True,
                                       use_cbgs=False,
                                       img_conf=img_conf,
                                       num_sweeps=1,
                                       sweep_idxes=list(),
                                       key_idxes=list(),
                                       return_depth=True,
                                       use_fusion=False)

            return dataset


    


    




