# -----------------------------------------------------------------------
# Copyright (C) 2020 Brady Zhou
# Released under the MIT license
# https://github.com/bradyz/cross_view_transformers/blob/master/LICENSE
# -----------------------------------------------------------------------
# modified by yelin2
# -----------------------------------------------------------------------


import torch
import torchvision

import cv2
import numpy as np
from PIL import Image
INTERPOLATION = cv2.LINE_8

from pathlib import Path
from functools import lru_cache

from data_module.dataset.utils import *
from pyquaternion import Quaternion
from shapely.geometry import MultiPolygon


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
                dataset_dir,
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
        self.dataset_dir = dataset_dir

        self.img_cfg = kwargs['image']
        self.img_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

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


    def get_category_index(self, name, categories):
        """
        human.pedestrian.adult
        """
        tokens = name.split('.')

        for i, category in enumerate(categories):
            if category in tokens:
                return i

        return None


    def get_annotations_by_category(self, sample, categories):
        result = [[] for _ in categories]

        for ann_token in self.nusc.get('sample', sample['token'])['anns']:
            a = self.nusc.get('sample_annotation', ann_token)
            idx = self.get_category_index(a['category_name'], categories)

            if idx is not None:
                result[idx].append(a)

        return result


    def convert_to_box(self, sample, annotations):
        # Import here so we don't require nuscenes-devkit unless regenerating labels
        from nuscenes.utils import data_classes

        V = self.meter2pix
        M_inv = np.array(sample['pose_inverse'])
        S = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

        # bbox ann 각각에 대해서 
        for a in annotations:
            # 3D bbox in LidarSeg data
            #   a['translation']: bbox center location in meters (x, y, z)
            #   a['size']: bbox size in meters (width, length, height)
            #   a['rotation']: bbox rotation (queternion)
            box = data_classes.Box(a['translation'], a['size'], Quaternion(a['rotation']))

            # 3D bbox에서 four bottom corners return
            # 처음 두개는 forward 방향에 있는 corners, 다음 두개는 backward 방향에 있는 corners
            corners = box.bottom_corners()                                              # 3 4

            # 3D bbox의 center
            center = corners.mean(-1)                                                   # 3

            # 
            front = (corners[:, 0] + corners[:, 1]) / 2.0                               # 3
            left = (corners[:, 0] + corners[:, 3]) / 2.0                                # 3

            p = np.concatenate((corners, np.stack((center, front, left), -1)), -1)      # 3 7
            p = np.pad(p, ((0, 1), (0, 0)), constant_values=1.0)                        # 4 7
            p = V @ S @ M_inv @ p                                                       # 3 7

            yield p                                                                     # 3 7

    def get_static_layers(self, sample, layers, patch_radius=150):
        h, w = self.bh, self.bw
        V = self.meter2pix
        M_inv = np.array(sample['pose_inverse'])
        S = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

        # map에서 ego pose 기준 300m x 300m bbox left_top, right_bottom
        box_coords = (sample['pose'][0][-1] - patch_radius, 
                        sample['pose'][1][-1] - patch_radius,
                        sample['pose'][0][-1] + patch_radius, 
                        sample['pose'][1][-1] + patch_radius)
        
        # box_coords에 해당하는 layers class 정보 얻음
        #   records_in_patch(dict)
        #         key       |     value
        #     ----------------------------------
        #     'lane'          | list of tokens
        #     'road_segment'  | list of tokens

        records_in_patch = self.nusc_map.get_records_in_patch(
                                box_coords, layers, 'intersect')

        result = []

        for layer in layers:
            render = np.zeros((h, w), dtype=np.uint8)

            for r in records_in_patch[layer]:
                
                # polygon tokens = polygon 정보를 불러오기 위한 token list
                polygon_token = self.nusc_map.get(layer, r)
                if layer == 'drivable_area': 
                    polygon_tokens = polygon_token['polygon_tokens']
                else: 
                    polygon_tokens = [polygon_token['polygon_token']]

                for p in polygon_tokens:
                    # polygon (x, y) 좌표들 얻기
                    polygon = self.nusc_map.extract_polygon(p)
                    polygon = MultiPolygon([polygon])

                    # polygon exterior (1로 채움)
                    exteriors = [np.array(poly.exterior.coords).T for poly in polygon.geoms]            # 2 n
                    exteriors = [np.pad(p, ((0, 1), (0, 0)), constant_values=0.0) for p in exteriors]   # 3 n
                    exteriors = [np.pad(p, ((0, 1), (0, 0)), constant_values=1.0) for p in exteriors]   # 4 n
                    # world -> egolidar -> z축 삭제 -> meter2pix
                    exteriors = [V @ S @ M_inv @ p for p in exteriors]                                  # 4 n
                    exteriors = [p[:2].round().astype(np.int32).T for p in exteriors]
                    cv2.fillPoly(render, exteriors, 1, INTERPOLATION)

                    # polygon interior (interior 부분 0으로 채움)
                    interiors = [np.array(pi.coords).T for poly in polygon.geoms for pi in poly.interiors]  # 2 n
                    interiors = [np.pad(p, ((0, 1), (0, 0)), constant_values=0.0) for p in interiors]       # 3 n
                    interiors = [np.pad(p, ((0, 1), (0, 0)), constant_values=1.0) for p in interiors]       # 4 n
                    interiors = [V @ S @ M_inv @ p for p in interiors]
                    interiors = [p[:2].round().astype(np.int32).T for p in interiors]

                    cv2.fillPoly(render, interiors, 0, INTERPOLATION)

            result.append(render)
        
        return 255 * np.stack(result, -1)


    def get_line_layers(self, sample, layers, patch_radius=150, thickness=1):
        h, w = self.bh, self.bw
        V = self.meter2pix
        M_inv = np.array(sample['pose_inverse'])
        S = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

        box_coords = (sample['pose'][0][-1] - patch_radius, 
                        sample['pose'][1][-1] - patch_radius,
                        sample['pose'][0][-1] + patch_radius, 
                        sample['pose'][1][-1] + patch_radius)

        records_in_patch = self.nusc_map.get_records_in_patch(
                                    box_coords, layers, 'intersect')

        result = []

        for layer in layers:
            render = np.zeros((h, w), dtype=np.uint8)

            for r in records_in_patch[layer]:
                polygon_token = self.nusc_map.get(layer, r)
                line = self.nusc_map.extract_line(polygon_token['line_token'])

                p = np.float32(line.xy)                                     # 2 n
                p = np.pad(p, ((0, 1), (0, 0)), constant_values=0.0)        # 3 n
                p = np.pad(p, ((0, 1), (0, 0)), constant_values=1.0)        # 4 n
                p = V @ S @ M_inv @ p                                       # 3 n
                p = p[:2].round().astype(np.int32).T                        # n 2

                cv2.polylines(render, [p], False, 1, thickness=thickness)

            result.append(render)

        return 255 * np.stack(result, -1)


    def get_dynamic_layers(self, sample, anns_by_category):
        h, w = self.bh, self.bw
        result = list()

        # DYNAMIC category들 별로
        for anns in anns_by_category:
            render = np.zeros((h, w), dtype=np.uint8)

            for p in self.convert_to_box(sample, anns):
                p = p[:2, :4]

                cv2.fillPoly(render, [p.round().astype(np.int32).T], 1, INTERPOLATION)

            result.append(render)

        return 255 * np.stack(result, -1)


    def get_dynamic_objects(self, sample, annotations):
        h, w = self.bh, self.bw

        center_score = np.zeros((h, w), dtype=np.float32)
        center_offset = np.zeros((h, w, 2), dtype=np.float32)
        buf = np.zeros((h, w), dtype=np.uint8)

        visibility = np.full((h, w), 255, dtype=np.uint8)

        coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float32)

        for ann, p in zip(annotations, self.convert_to_box(sample, annotations)):
            box = p[:2, :4]
            center = p[:2, 4]
            front = p[:2, 5]
            left = p[:2, 6]

            buf.fill(0)
            cv2.fillPoly(buf, [box.round().astype(np.int32).T], 1, INTERPOLATION)
            mask = buf > 0

            if not np.count_nonzero(mask):
                continue

            sigma = 1
            center_offset[mask] = center[None] - coords[mask]
            center_score[mask] = np.exp(-(center_offset[mask] ** 2).sum(-1) / (sigma ** 2))

            visibility[mask] = ann['visibility_token']

        center_score = center_score[..., None]

        return center_score, visibility


    def get_images(self, sample, h, w, top_crop):
        images = list()
        intrinsics = list()

        for image_path, I_original in zip(sample['images'], sample['intrinsics']):
            h_resize = h + top_crop
            w_resize = w

            image = Image.open(self.dataset_dir + '/' + image_path)

            image_new = image.resize((w_resize, h_resize), resample=Image.BILINEAR)
            image_new = image_new.crop((0, top_crop, image_new.width, image_new.height))

            I = np.float32(I_original)
            I[0, 0] *= w / image.width
            I[0, 2] *= w / image.width
            I[1, 1] *= h / image.height
            I[1, 2] *= h / image.height
            I[1, 2] -= top_crop

            images.append(self.img_transform(image_new))
            intrinsics.append(torch.tensor(I))

        return {
            'cam_idx': torch.LongTensor(sample['cam_ids']),
            'image': torch.stack(images, 0),
            'intrinsics': torch.stack(intrinsics, 0),
            'extrinsics': torch.tensor(np.float32(sample['extrinsics'])),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        '''
                        shape           range
        bev         [200, 200, 12]      [0, 255]
        view        [3, 3]        
        center      [200, 200, 1]       [0, 1]
        visibility  [200, 200]          [1,2,3,4,255]
        '''
        sample = self.samples[idx]

        # map-view segmentation target
        anns_dynamic = self.get_annotations_by_category(sample, DYNAMIC)
        anns_vehicle = self.get_annotations_by_category(sample, ['vehicle'])[0]

        static = self.get_static_layers(sample, STATIC)
        dividers = self.get_line_layers(sample, DIVIDER, thickness=2)
        dynamic = self.get_dynamic_layers(sample, anns_dynamic) 
        bev = np.concatenate((static, dividers, dynamic), -1)

        center, visibility = self.get_dynamic_objects(sample, anns_vehicle)

        result = dict()
        result.update(self.get_images(sample, **self.img_cfg))

        result.update({'bev': torch.tensor(bev),
                        'view': torch.tensor(self.meter2pix),
                        'center': torch.tensor(center),
                        'visibility': torch.tensor(visibility)})


        return result

