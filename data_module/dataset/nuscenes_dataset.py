# -----------------------------------------------------------------------
# Copyright (C) 2020 Brady Zhou
# Released under the MIT license
# https://github.com/bradyz/cross_view_transformers/blob/master/LICENSE
# -----------------------------------------------------------------------
# modified by yelin2
# -----------------------------------------------------------------------


import torch
import numpy as np
import cv2

from pathlib import Path
from functools import lru_cache

from data_module.dataset.utils import *
# from pyquaternion import Quaternion
# from shapely.geometry import MultiPolygon

# from .common import INTERPOLATION, get_view_matrix, get_pose, get_split
# from .transforms import Sample, SaveDataTransform


STATIC = ['lane', 'road_segment']
DIVIDER = ['road_divider', 'lane_divider']
DYNAMIC = [
    'car', 'truck', 'bus',
    'trailer', 'construction',
    'pedestrian',
    'motorcycle', 'bicycle',
]

CLASSES = STATIC + DIVIDER + DYNAMIC
NUM_CLASSES = len(CLASSES)



class NuScenesDataset(torch.utils.data.Dataset):
    CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
               'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    def __init__(self, 
                nusc, 
                nusc_map, 
                scene_name, 
                scene_record,
                **kwargs):
        '''
        kwargs
            num_classes: 12
            cameras: [[0, 1, 2, 3, 4, 5]]
            bev: {'h': 200, 'w': 200, 
                    'h_meters': 100.0, 'w_meters': 100.0, 'offset': 0.0}
            image: {'h': 224, 'w': 480, 'top_crop': 46}
        '''

        self.scene_name = scene_name
        self.scene_record = scene_record

        self.nusc = nusc
        self.nusc_map = nusc_map

        self.bh = kwargs['bev']['h']
        self.bw = kwargs['bev']['w']
        self.meter2pix = get_bev_meter2pix_matrix(kwargs['bev'])

        self.samples = self.get_scene_samples(scene_record, kwargs['cameras'])
        

    def get_scene_samples(self, scene_record, cameras):
        samples = []
        sample_token = scene_record['first_sample_token']

        while sample_token:
            sample_record = self.nusc.get('sample', sample_token)

            samples.append(self.get_sample_record(sample_record, cameras[0]))
            sample_token = sample_record['next']

        return samples


    def get_sample_record(self, sample_record, cameras):
        lidar_record = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        egolidar = self.nusc.get('ego_pose', lidar_record['ego_pose_token'])

        egolidarflat2world = get_pose(egolidar, flat=True)
        world2egolidarflat = get_pose(egolidar, flat=True, inv=True)

        cam_channels = []
        images = []
        intrinsics = []
        extrinsics = []

        for cam_idx in cameras:
            cam_ch = self.CAMERAS[cam_idx]
            cam_token = sample_record['data'][cam_ch]

            cam_record = self.nusc.get('sample_data', cam_token)
            egocam = self.nusc.get('ego_pose', cam_record['ego_pose_token'])
            cam = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])

            egocam2cam = get_pose(cam, inv=True)
            world2egocam = get_pose(egocam, inv=True)

            E = egocam2cam @ world2egocam @ egolidarflat2world  # egolidar2cam
            I = cam['camera_intrinsic']

            full_path = Path(self.nusc.get_sample_data_path(cam_token))
            image_path = str(full_path.relative_to(self.nusc.dataroot))

            cam_channels.append(cam_ch)
            intrinsics.append(I)
            extrinsics.append(E.tolist())
            images.append(image_path)

        return {
            'scene': self.scene_name,
            'token': sample_record['token'],

            'pose': egolidarflat2world.tolist(),
            'pose_inverse': world2egolidarflat.tolist(),

            'cam_ids': list(cameras),
            'cam_channels': cam_channels,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'images': images,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        
        return 20
