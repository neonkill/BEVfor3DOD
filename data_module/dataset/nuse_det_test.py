
# -----------------------------------------------------------------------
# Copyright (C) 2020 Brady Zhou
# Released under the MIT license
# https://github.com/bradyz/cross_view_transformers/blob/master/LICENSE
# -----------------------------------------------------------------------
# modified by yelin2
# -----------------------------------------------------------------------


import torch
import torchvision
import os

import cv2
import numpy as np
from PIL import Image
INTERPOLATION = cv2.LINE_8

from pathlib import Path
from functools import lru_cache

from data_module.dataset.utils import *
from pyquaternion import Quaternion
from shapely.geometry import MultiPolygon
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import view_points


STATIC = ['lane', 'road_segment']
DIVIDER = ['road_divider', 'lane_divider']
DYNAMIC = [
    'car', 'truck', 'bus',
    'trailer', 'construction',
    'pedestrian',
    'motorcycle', 'bicycle',
]

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.constructi    on_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}

CLASSES = STATIC + DIVIDER + DYNAMIC
NUM_CLASSES = len(CLASSES)


def map_pointcloud_to_image(
        lidar_points,
        img,
        lidar_calibrated_sensor,
        lidar_ego_pose,
        cam_calibrated_sensor,
        cam_ego_pose,
        min_dist: float = 0.0,
    ):

        # Points live in the point sensor frame. So they need to be
        # transformed via global to the image plane.
        # First step: transform the pointcloud to the ego vehicle
        # frame for the timestamp of the sweep.

        lidar_points = LidarPointCloud(lidar_points.T)
        lidar_points.rotate(
            Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
        lidar_points.translate(np.array(lidar_calibrated_sensor['translation']))

        # Second step: transform from ego to the global frame.
        lidar_points.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
        lidar_points.translate(np.array(lidar_ego_pose['translation']))

        # Third step: transform from global into the ego vehicle
        # frame for the timestamp of the image.
        lidar_points.translate(-np.array(cam_ego_pose['translation']))
        lidar_points.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        lidar_points.translate(-np.array(cam_calibrated_sensor['translation']))
        lidar_points.rotate(
            Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = lidar_points.points[2, :]
        coloring = depths

        # Take the actual picture (matrix multiplication with camera-matrix
        # + renormalization).
        points = view_points(lidar_points.points[:3, :],
                             np.array(cam_calibrated_sensor['camera_intrinsic']),
                             normalize=True)

        # Remove points that are either outside or behind the camera.
        # Leave a margin of 1 pixel for aesthetic reasons. Also make
        # sure points are at least 1m in front of the camera to avoid
        # seeing the lidar points on the camera casing for non-keyframes
        # which are slightly out of sync.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < img.size[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < img.size[1] - 1)
        points = points[:, mask]
        coloring = coloring[mask]

        return points, coloring



class NuScenesDataset(torch.utils.data.Dataset):
    CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
               'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    CLASSES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

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
    



    def get_lidar_depth(self, lidar_points, img, lidar_record, cam_record):
        
        lidar_calibrated_sensor = self.nusc.get('calibrated_sensor', lidar_record['calibrated_sensor_token'])        
        lidar_ego_pose = self.nusc.get('ego_pose', lidar_record['ego_pose_token'])
        cam_calibrated_sensor = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
        cam_ego_pose = self.nusc.get('ego_pose', cam_record['ego_pose_token'])
        
        pts_img, depth = map_pointcloud_to_image(
        lidar_points.copy(), img, lidar_calibrated_sensor.copy(),
        lidar_ego_pose.copy(), cam_calibrated_sensor.copy(), cam_ego_pose.copy())

        return np.concatenate([pts_img[:2, :].T, depth[:, None]],axis=1).astype(np.float32)

    def get_boxgt(self, anns, egocams, cams):
        """Generate gt labels from info.
        Args:
            info(dict): Infos needed to generate gt labels.
            cams(list): Camera names.
        Returns:
            Tensor: GT bboxes.
            Tensor: GT labels.
        """
        
        ego2global_rotation = np.mean(
            [egocams[cam]['rotation'] for cam in range(len(cams))],
            0)
        ego2global_translation = np.mean([
            egocams[cam]['translation'] for cam in range(len(cams))
        ], 0)
        trans = -np.array(ego2global_translation)
        rot = Quaternion(ego2global_rotation).inverse
        gt_boxes = list()
        gt_labels = list()
        for ann_info in anns:
            # Use ego coordinate.
            if (map_name_from_general_to_detection[ann_info['category_name']]
                    not in self.CLASSES
                    or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <=
                    0):
                continue
            box = Box(
                ann_info['translation'],
                ann_info['size'],
                Quaternion(ann_info['rotation']),
                velocity=ann_info['velocity'],
            )
            box.translate(trans)
            box.rotate(rot)
            box_xyz = np.array(box.center)
            box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
            box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
            box_velo = np.array(box.velocity[:2])
            gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
            gt_boxes.append(gt_box)
            gt_labels.append(
                self.CLASSES.index(map_name_from_general_to_detection[
                    ann_info['category_name']]))

        return gt_boxes, gt_labels

    def get_sample_record(self, sample_record, cameras):
        lidar_record = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        lidar_path = lidar_record['filename']
        lidar_points = np.fromfile(os.path.join(self.dataset_dir, lidar_path),dtype=np.float32,count=-1).reshape(-1, 5)[..., :4]
        egolidar = self.nusc.get('ego_pose', lidar_record['ego_pose_token'])
        egolidar_calibrated = self.nusc.get('calibrated_sensor', lidar_record['calibrated_sensor_token'])

        egolidarflat2world = get_pose(egolidar, flat=True)
        world2egolidarflat = get_pose(egolidar, flat=True, inv=True)

        cam_channels = []
        images = []
        intrinsics = []
        extrinsics = []
        egocams = []
        cam_records = []

        for cam_idx in cameras:
            cam_ch = self.CAMERAS[cam_idx]
            cam_token = sample_record['data'][cam_ch]

            cam_record = self.nusc.get('sample_data', cam_token)
            egocam = self.nusc.get('ego_pose', cam_record['ego_pose_token'])    #global2egocam
            cam = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])  #egocam2sensor

            egocam2cam = get_pose(cam, inv=True)
            world2egocam = get_pose(egocam, inv=True)

            E = egocam2cam @ world2egocam @ egolidarflat2world  # egolidar2cam
            I = cam['camera_intrinsic']

            full_path = Path(self.nusc.get_sample_data_path(cam_token))
            image_path = str(full_path.relative_to(self.nusc.dataroot))
            
            
            egocams.append(egocam)
            cam_channels.append(cam_ch)
            intrinsics.append(I)
            extrinsics.append(E.tolist())
            images.append(image_path)
            cam_records.append(cam_record)
            

            #! move to getitem
            # image = Image.open(self.dataset_dir + '/' + image_path)
            # point_depth = self.get_lidar_depth(lidar_points, image, lidar_record, cam_record)
            # point_depths.append(point_depth)
            


        return {
            'scene': self.scene_name,
            'token': sample_record['token'],

            'lidar_record': lidar_record,
            'cam_records': cam_records,
            'lidar_points': lidar_points,

            'pose': egolidarflat2world.tolist(),
            'pose_inverse': world2egolidarflat.tolist(),

            'cam_ids': list(cameras),
            'cam_channels': cam_channels,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'images': images,
            'egocams':egocams,
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
    
    def get_annotations_box(self, sample):
        ann_infos = list()
        for ann_token in self.nusc.get('sample', sample['token'])['anns']:
            a = self.nusc.get('sample_annotation', ann_token)
            velocity = self.nusc.box_velocity(a['token'])
            if np.any(np.isnan(velocity)):
                velocity = np.zeros(3)
            a['velocity'] = velocity
            ann_infos.append(a)

        return ann_infos


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
        depths = list()

        intrinsics = list()

        for i, (image_path, I_original) in enumerate(zip(sample['images'], sample['intrinsics'])):
            h_resize = h + top_crop
            w_resize = w

            # read image & depth
            image = Image.open(self.dataset_dir + '/' + image_path)

            point_depth = self.get_lidar_depth(sample['lidar_points'], 
                                                image, 
                                                sample['lidar_record'], 
                                                sample['cam_records'][i])

            # image resize & crop
            image_new = image.resize((w_resize, h_resize), resample=Image.BILINEAR)
            image_new = image_new.crop((0, top_crop, image_new.width, image_new.height))

            #! depth resize & crop
            # point_depth_new = point_depth
            point_depth_new = depth_transform(cam_depth=point_depth, 
                                            resize=(1,1),
                                            resize_dims=(h, w),
                                            crop=[0, 0],
                                            flip=False,
                                            rotate=0)

            I = np.float32(I_original)
            I[0, 0] *= w / image.width
            I[0, 2] *= w / image.width
            I[1, 1] *= h / image.height
            I[1, 2] *= h / image.height
            I[1, 2] -= top_crop

            images.append(self.img_transform(image_new))
            intrinsics.append(torch.tensor(I))
            depths.append(torch.tensor(point_depth_new))

        return {
            'depth': torch.stack(depths, 0),
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
        anns_box = self.get_annotations_box(sample)

        static = self.get_static_layers(sample, STATIC)
        dividers = self.get_line_layers(sample, DIVIDER, thickness=2)
        dynamic = self.get_dynamic_layers(sample, anns_dynamic) 
        bev = np.concatenate((static, dividers, dynamic), -1)

        center, visibility = self.get_dynamic_objects(sample, anns_vehicle)


        # 3D object detection target
        gt_box, gt_label = self.get_boxgt(anns_box, sample['egocams'], self.CAMERAS)
        

        # input images & depth target
        result = dict()
        result.update(self.get_images(sample, **self.img_cfg))

        result.update({'bev': torch.tensor(bev),
                        'view': torch.tensor(self.meter2pix),
                        'center': torch.tensor(center),
                        'visibility': torch.tensor(visibility),
                        'gt_box':torch.tensor(gt_box),
                        'gt_label':torch.tensor(gt_label)})


        return result

