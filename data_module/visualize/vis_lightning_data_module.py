

from pathlib import Path

import torch
import pytorch_lightning as pl

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

from .. import get_dataset_module_by_name


def get_split(split):
    
    path = Path(__file__).parent / 'splits' /  f'{split}.txt'
    return path.read_text().strip().split('\n')


class DataModule(pl.LightningDataModule):

    def __init__(self, data_cfg, loader_cfg):
        super().__init__()

        self.get_data = get_dataset_module_by_name('vis_nuscenes_dataset').get_data
        self.data_cfg = data_cfg
        self.loader_cfg = loader_cfg


    def get_split(self, split, shuffle):

        datasets = self.get_data(split=split, **self.data_cfg)
        dataset = torch.utils.data.ConcatDataset(datasets)

        loader_config = dict(self.loader_cfg)

        if loader_config['num_workers'] == 0:
            loader_config['prefetch_factor'] = 2

        return torch.utils.data.DataLoader(dataset, shuffle=shuffle, collate_fn=collate_fn, **loader_config)


    def train_dataloader(self, shuffle=True):
        return self.get_split('train', shuffle=shuffle)

    def val_dataloader(self, shuffle=True):
        return self.get_split('val', shuffle=shuffle)

    def eval_dataloader(self, shuffle=False):
        return self.get_split('eval', shuffle=shuffle)



def collate_fn(batchs):

    bevs, views = [], []
    centers, visibilitys = [], []
    depths, cam_idxs, images = [], [], []
    intrinsics, extrinsics = [], []
    gt_boxs, gt_labels = [], []

    for batch in batchs:
        bevs.append(batch['bev'])
        views.append(batch['view'])
        centers.append(batch['center'])
        visibilitys.append(batch['visibility'])
        depths.append(batch['depth'])
        cam_idxs.append(batch['cam_idx'])
        images.append(batch['image'])
        intrinsics.append(batch['intrinsics'])
        extrinsics.append(batch['extrinsics'])
        gt_boxs.append(batch['gt_box'])
        gt_labels.append(batch['gt_label'])

    results = {}
    results['bev'] = torch.stack(bevs, dim=0)
    results['view'] = torch.stack(views, dim=0)
    results['center'] = torch.stack(centers, dim=0)
    results['visibility'] = torch.stack(visibilitys, dim=0)
    results['depth'] = torch.stack(depths, dim=0)
    results['cam_idx'] = torch.stack(cam_idxs, dim=0)
    results['image'] = torch.stack(images, dim=0)
    results['intrinsics'] = torch.stack(intrinsics, dim=0)
    results['extrinsics'] = torch.stack(extrinsics, dim=0)

    results['gt_box'] = gt_boxs
    results['gt_label'] = gt_labels

    return results



