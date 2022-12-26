

from pathlib import Path

import torch
import pytorch_lightning as pl

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

from data_module.dataset.nuscenes_dataset import NuScenesDataset

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


            # datasets.append(1)
            dataset = NuScenesDataset(nusc, 
                                        nusc_map, 
                                        dataset_dir,
                                        scene_name, 
                                        scene_record, 
                                        **kwargs)
            datasets.append(dataset)

        return datasets


    def get_loader(self, split, shuffle):
        
        version = 'v1.0-' + split
        loader_cfg = dict(self.loader_cfg)

        dataset = NuimagesDataset(tf_cfg=self.data_cfg, version=version)

        return torch.utils.data.DataLoader(dataset, 
                                        shuffle=shuffle, 
                                        **loader_cfg)


    def train_dataloader(self, shuffle=True):
        return self.get_loader('train', shuffle=shuffle)

    def val_dataloader(self, shuffle=True):
        return self.get_loader('val', shuffle=shuffle)

    def eval_dataloader(self, shuffle=False):
        return self.get_loader('eval', shuffle=shuffle)