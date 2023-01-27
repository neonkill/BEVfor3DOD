

from pathlib import Path

import torch
import pytorch_lightning as pl

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

from . import get_dataset_module_by_name


# def get_split(split):
#     path = Path(__file__).parent / 'splits' /  f'{split}.txt'
#     return path.read_text().strip().split('\n')


class DataModule(pl.LightningDataModule):

    def __init__(self, dataset, data_cfg, loader_cfg):
        super().__init__()

        self.get_data = get_dataset_module_by_name(dataset).get_data
        self.data_cfg = data_cfg
        self.loader_cfg = loader_cfg


    def get_split(self, split, shuffle):

        datasets = self.get_data(split=split, **self.data_cfg)
        dataset = torch.utils.data.ConcatDataset(datasets)

        loader_config = dict(self.loader_cfg)

        if loader_config['num_workers'] == 0:
            loader_config['prefetch_factor'] = 2

        #! return torch.utils.data.DataLoader(dataset, shuffle=shuffle, collate_fn=collate_fn, **loader_config)
        return torch.utils.data.DataLoader(dataset, shuffle=shuffle, collate_fn=collate_fn, **loader_config)


    def train_dataloader(self, shuffle=True):
        return self.get_split('train', shuffle=shuffle)

    def val_dataloader(self, shuffle=True):
        return self.get_split('val', shuffle=shuffle)

    def test_dataloader(self, shuffle=False):
        return self.get_split('eval', shuffle=shuffle)



def collate_fn(batchs):

    bevs, views = [], []
    centers, visibilitys = [], []
    depths, cam_idxs, images = [], [], []
    intrinsics, extrinsics = [], []
    gt_boxes, gt_labels = [], []
    sensor2sensor_mats = []
    sensor2ego_mats = []
    ida_mats = []
    img_metas = []
    images_before_crop = [] #!


    for batch in batchs:
        bevs.append(batch['bev'])
        views.append(batch['view'])
        centers.append(batch['center'])
        visibilitys.append(batch['visibility'])
        depths.append(batch['depths'])
        cam_idxs.append(batch['cam_idx'])
        images.append(batch['image'])
        intrinsics.append(batch['intrinsics'])
        extrinsics.append(batch['extrinsics'])
        gt_boxes.append(batch['gt_boxes'])
        gt_labels.append(batch['gt_labels'])
        sensor2sensor_mats.append(batch['sensor2sensor_mats'])
        sensor2ego_mats.append(batch['sensor2ego_mats'])
        ida_mats.append(batch['ida_mats'])
        img_metas.append(batch['img_metas'])
        images_before_crop.append(batch['images_before_crop'])  #!


    results = {}
    results['bev'] = torch.stack(bevs, dim=0)
    results['view'] = torch.stack(views, dim=0)
    results['center'] = torch.stack(centers, dim=0)
    results['visibility'] = torch.stack(visibilitys, dim=0)
    results['depths'] = torch.stack(depths, dim=0)
    results['cam_idx'] = torch.stack(cam_idxs, dim=0)
    results['image'] = torch.stack(images, dim=0)
    results['intrinsics'] = torch.stack(intrinsics, dim=0)
    results['extrinsics'] = torch.stack(extrinsics, dim=0)
    results['sensor2sensor_mats'] = torch.stack(sensor2sensor_mats, dim=0)
    results['sensor2ego_mats'] = torch.stack(sensor2ego_mats, dim=0)
    results['ida_mats'] = torch.stack(ida_mats, dim=0)
    results['images_before_crop'] = torch.stack(images_before_crop, dim=0)      #!

    results['img_metas'] = img_metas
    results['gt_boxes'] = gt_boxes
    results['gt_labels'] = gt_labels

    return results



