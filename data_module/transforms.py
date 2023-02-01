# -----------------------------------------------------------------------
# Copyright (C) 2020 Brady Zhou
# Released under the MIT license
# https://github.com/bradyz/cross_view_transformers/blob/master/LICENSE
# -----------------------------------------------------------------------

import os
import pathlib
import mmcv
import torch
import torchvision
import numpy as np

from PIL import Image
from .common import encode, decode, map_pointcloud_to_image, depth_transform
from .augmentations import StrongAug, GeometricAug

from pyquaternion import Quaternion

from nuscenes.utils.data_classes import Box, LidarPointCloud
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes

'''
Used in 3D O.D

image_aug: 0~1 normalization + top crop
bda_aug: GT Box in BEV Frame Augmentation (BEVDepth type)
'''

# np.random.seed(45)

class Sample(dict):
    def __init__(
        self,
        token,
        scene,
        intrinsics,
        extrinsics,
        images,
        view,
        bev,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Used to create path in save/load
        self.token = token      # token name
        self.scene = scene      # scene name

        self.view = view    # global coordinate to ego vehicle coordinate
        self.bev = bev      # bev segmentation map

        self.images = images            

        self.intrinsics = intrinsics    # camera parameters
        self.extrinsics = extrinsics

    def __getattr__(self, key):
        return super().__getitem__(key)

    def __setattr__(self, key, val):
        self[key] = val

        return super().__setattr__(key, val)


class SaveDataTransform:
    """
    All data to be saved to .json must be passed in as native Python lists
    """
    def __init__(self, labels_dir):
        self.labels_dir = pathlib.Path(labels_dir)

    def get_cameras(self, batch: Sample):
        return {
            'images': batch.images,
            'intrinsics': batch.intrinsics,
            'extrinsics': batch.extrinsics
        }

    def get_bev(self, batch: Sample):
        result = {
            'view': batch.view,
        }

        scene_dir = self.labels_dir / batch.scene

        bev_path = f'bev_{batch.token}.png'
        Image.fromarray(encode(batch.bev)).save(scene_dir / bev_path)

        result['bev'] = bev_path

        # Auxilliary labels
        if batch.get('aux') is not None:
            aux_path = f'aux_{batch.token}.npz'
            np.savez_compressed(scene_dir / aux_path, aux=batch.aux)

            result['aux'] = aux_path

        # Visibility mask
        if batch.get('visibility') is not None:
            visibility_path = f'visibility_{batch.token}.png'
            Image.fromarray(batch.visibility).save(scene_dir / visibility_path)

            result['visibility'] = visibility_path

        return result

    def __call__(self, batch):
        """
        Save sensor/label data and return any additional info to be saved to json
        """
        result = {}
        result.update(self.get_cameras(batch))
        result.update(self.get_bev(batch))
        result.update({k: v for k, v in batch.items() if k not in result})

        return result


class LoadDataTransform(torchvision.transforms.ToTensor):
    #! TODO: is_train
    def __init__(self, dataset_dir, labels_dir, image_config, num_classes, augment='none', is_train=False):
        super().__init__()

        self.dataset_dir = pathlib.Path(dataset_dir)
        self.labels_dir = pathlib.Path(labels_dir)
        self.image_config = image_config
        self.num_classes = num_classes

        xform = {
            'none': [],
            'strong': [StrongAug()],
            'geometric': [StrongAug(), GeometricAug()],
        }[augment] + [torchvision.transforms.ToTensor()]

        self.img_trans_vision = torchvision.transforms.Compose(xform)
        self.to_tensor = super().__call__

        #! BEVDepth aug vars
        self.is_train = is_train 
        self.bda_aug_conf= { 'rot_lim': [-22.5, 22.5],
                        'scale_lim': [0.95, 1.05],
                        'flip_dx_ratio': 0.5,
                        'flip_dy_ratio': 0.5 } 
        self.img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                        img_std=[58.395, 57.12, 57.375],
                        to_rgb=True)
        self.img_mean = np.array(self.img_conf['img_mean'], np.float32)
        self.img_std = np.array(self.img_conf['img_std'], np.float32)
        self.to_rgb = self.img_conf['to_rgb']

    def get_sensor2sensor_mat(self):
        sensor2sensor_mat = np.full((4, 4), 1e-9, dtype=np.float32)
        np.fill_diagonal(sensor2sensor_mat, 1.0)
        return sensor2sensor_mat

    #resize: [0.44, 0.344] / resize_dim[704, 310] / crop [0, 54, 704, 310]
    def get_img_transform(self, resize, crop):
        '''
        resize: [w, h]
        '''
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)

        ida_rot[0,:] *= resize[0]
        ida_rot[1,:] *= resize[1]

        ida_tran -= torch.Tensor(crop[:2])

        ida_mat = ida_rot.new_zeros(4, 4)
        ida_mat[3, 3] = 1
        ida_mat[2, 2] = 1
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 3] = ida_tran
        
        return ida_mat


    def get_lidar_depth(self, lidar_points, img, 
                        lidar_calibrated_sensor, 
                        lidar_ego_pose, 
                        cam_calibrated_sensor, 
                        cam_ego_pose):
        
        pts_img, depth = map_pointcloud_to_image(
                                    lidar_points.copy(), img, 
                                    lidar_calibrated_sensor.copy(),
                                    lidar_ego_pose.copy(), 
                                    cam_calibrated_sensor.copy(), 
                                    cam_ego_pose.copy())

        return np.concatenate([pts_img[:2, :].T, depth[:, None]],axis=1).astype(np.float32)


    def get_cameras(self, sample: Sample, h, w, top_crop):

    
        """
        Note: we invert I and E here for convenience.
        """
        images = list()
        # images_before_crop = list() #!
        intrinsics = list()
        sensor2sensor_mats = list()
        sensor2ego_mats = list()
        ida_mats = list()
        depths = list()

        lidar_path = sample.lidar_path
        lidar_points = np.fromfile(os.path.join(self.dataset_dir, lidar_path),
                                    dtype=np.float32,count=-1).reshape(-1, 5)[..., :4]

        ego2global_translation = torch.tensor(np.float32(sample.ego2global_translation))
        ego2global_rotation = torch.tensor(np.float32(sample.ego2global_rotation))
         #! depth aug
        

        # resize: [0.44, 0.344] / resize_dim[704, 310] / crop [0, 54, 704, 310]
        for i, (image_path, I_original, sensor2ego_mat) in enumerate(zip(sample.images, sample.intrinsics, sample.sensor2ego_mats)):

            # RGB images
            h_resize = h + top_crop
            w_resize = w

            image = Image.open(self.dataset_dir / image_path)

            image_new = image.resize((w_resize, h_resize), resample=Image.BILINEAR)
            image_new = image_new.crop((0, top_crop, image_new.width, image_new.height))

            # images_before_crop.append(self.img_trans_vision(image))    #!

            resize = [w_resize/1600, h_resize/900]
            crop = [0, top_crop, image_new.width, image_new.height]
            ida_mat = self.get_img_transform(resize, crop)
           
            image_new = mmcv.imnormalize(np.array(image_new), self.img_mean,
                                       self.img_std, self.to_rgb)
            image_new = torch.from_numpy(image_new).permute(2, 0, 1)

            # depth 
            point_depth = self.get_lidar_depth(lidar_points, 
                                                image, 
                                                sample.lidar_calibrated_sensor, 
                                                sample.lidar_ego_pose, 
                                                sample.cam_calibrated_sensors[i], 
                                                sample.cam_ego_poses[i])
            point_depth_new = point_depth
            point_depth_new = depth_transform(cam_depth=point_depth, 
                                            resize=resize,
                                            resize_dims=(h, w), #! (256,704)
                                            crop=crop)


            # intrinsic
            I = np.float32(I_original)

            sensor2ego_mat = np.float32(sensor2ego_mat)

            images.append(image_new)
            intrinsics.append(torch.tensor(I))
            sensor2ego_mats.append(torch.tensor(sensor2ego_mat))
            sensor2sensor_mats.append(torch.tensor(self.get_sensor2sensor_mat()))
            ida_mats.append(ida_mat)
            depths.append(torch.tensor(point_depth_new))

        img_metas = dict(
                token=sample.token,
                # box_type_3d=LiDARInstance3DBoxes,
                ego2global_translation=ego2global_translation,
                ego2global_rotation=ego2global_rotation,
            )

        return {
            'cam_idx': torch.LongTensor(sample.cam_ids),
            'image': torch.stack(images, 0),
            # 'images_before_crop': torch.stack(images_before_crop, 0), #! 
            'intrinsics': torch.stack(intrinsics, 0),
            'extrinsics': torch.tensor(np.float32(sample.extrinsics)),
            'sensor2sensor_mats': torch.stack(sensor2sensor_mats, 0),
            'sensor2ego_mats': torch.stack(sensor2ego_mats, 0),
            'ida_mats': torch.stack(ida_mats, 0),
            'img_metas': img_metas,
            'depths': torch.stack(depths, 0)
        }
    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def get_bev(self, sample: Sample):
        
        scene_dir = self.labels_dir / sample.scene
        bev = None

        if sample.bev is not None:
            bev = Image.open(scene_dir / sample.bev)
            bev = decode(bev, self.num_classes)
            bev = (255 * bev).astype(np.uint8)
            bev = self.to_tensor(bev)

        assert bev.shape == (12, 200, 200), f'{scene_dir}/{sample.bev} get different shape'
        result = {
            'bev': bev,
            'view': torch.tensor(sample.view),
        }

        if 'visibility' in sample:
            visibility = Image.open(scene_dir / sample.visibility)
            result['visibility'] = torch.tensor(np.array(visibility, dtype=np.uint8))

        if 'aux' in sample:
            aux = np.load(scene_dir / sample.aux)['aux']
            result['center'] = self.to_tensor(aux[..., 1])

        if 'pose' in sample:
            result['pose'] = torch.tensor(np.float32(sample['pose']))

        return result
    
    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx, flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        gt_boxes = gt_boxes.type(torch.float32)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1) 
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1) 
        return gt_boxes, rot_mat

    def get_3d_det(self, batch):
        gt_boxes = torch.tensor(np.array(batch.gt_boxes))
        gt_labels = torch.tensor(np.array(batch.gt_labels))

        #! BD
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                          flip_dx, flip_dy)

        bda_mat[:3, :3] = bda_rot
        return {'gt_boxes': gt_boxes, 'gt_labels': gt_labels, 'bda_mat':bda_mat}

        
    def __call__(self, batch):
        if not isinstance(batch, Sample):
            batch = Sample(**batch)

        result = dict()
        result.update(self.get_cameras(batch, **self.image_config))
        result.update(self.get_bev(batch))
        result.update(self.get_3d_det(batch))

        return result
