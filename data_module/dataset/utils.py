
import numpy as np
from functools import lru_cache
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points





# 원점이 bev map의 중심이고 meter 단위로 표현된 맵을 원점이 left-top이고 pixel 단위로 변환
def get_bev_meter2pix_matrix(bev):
    w = bev['h']
    h = bev['w']
    offset = bev['offset']

    sh = h / bev['h_meters']
    sw = w / bev['w_meters']

    return np.float32([
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ])


# 4x4 transform matrix 만들어줌
def get_pose(transform, inv=False, flat=False):
    
    rotation = transform['rotation']
    translation = transform['translation']

    if flat:
        yaw = Quaternion(rotation).yaw_pitch_roll[0]
        R = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).rotation_matrix
    else:
        R = Quaternion(rotation).rotation_matrix

    t = np.array(translation, dtype=np.float32)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R if not inv else R.T
    pose[:3, -1] = t if not inv else R.T @ -t

    return pose


# from BEVDepth
def depth_transform(cam_depth, resize, resize_dims, crop, flip, rotate):
    """Transform depth based on ida augmentation configuration.

    Args:
        cam_depth (np array): Nx3, 3: x,y,d.
        resize (float): Resize factor.
        resize_dims (list): Final dimension.
        crop (list): x1, y1, x2, y2
        flip (bool): Whether to flip.
        rotate (float): Rotation value.

    Returns:
        np array: [h/down_ratio, w/down_ratio, d]
    """

    H, W = resize_dims
    rh, rw = resize
    # cam_depth[:, :2] = cam_depth[:, :2] * resize
    cam_depth[:, 0] = cam_depth[:, 0] * rw
    cam_depth[:, 1] = cam_depth[:, 1] * rh
    cam_depth[:, 0] -= crop[0]
    cam_depth[:, 1] -= crop[1]
    if flip:
        cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

    cam_depth[:, 0] -= W / 2.0
    cam_depth[:, 1] -= H / 2.0

    h = rotate / 180 * np.pi
    rot_matrix = [
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ]
    cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T

    cam_depth[:, 0] += W / 2.0
    cam_depth[:, 1] += H / 2.0

    depth_coords = cam_depth[:, :2].astype(np.int16)

    depth_map = np.zeros(resize_dims)
    valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                  & (depth_coords[:, 0] < resize_dims[1])
                  & (depth_coords[:, 1] >= 0)
                  & (depth_coords[:, 0] >= 0))
    depth_map[depth_coords[valid_mask, 1],
              depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]

    return depth_map



class NuScenesSingleton:
    """
    Wraps both nuScenes and nuScenes map API

    This was an attempt to sidestep the 30 second loading time in a "clean" manner
    """
    def __init__(self, dataset_dir, version):
        """
        dataset_dir: /path/to/nuscenes/
        version: v1.0-trainval
        """
        self.dataroot = str(dataset_dir)
        self.nusc = self.lazy_nusc(version, self.dataroot)

    @classmethod
    def lazy_nusc(cls, version, dataroot):
        # Import here so we don't require nuscenes-devkit unless regenerating labels
        from nuscenes.nuscenes import NuScenes

        if not hasattr(cls, '_lazy_nusc'):
            cls._lazy_nusc = NuScenes(version=version, dataroot=dataroot)

        return cls._lazy_nusc

    def get_scenes(self):
        for scene_record in self.nusc.scene:
            yield scene_record['name'], scene_record

    @lru_cache(maxsize=16)
    def get_map(self, log_token):
        # Import here so we don't require nuscenes-devkit unless regenerating labels
        from nuscenes.map_expansion.map_api import NuScenesMap

        map_name = self.nusc.get('log', log_token)['location']
        nusc_map = NuScenesMap(dataroot=self.dataroot, map_name=map_name)

        return nusc_map

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_singleton'):
            obj = super(NuScenesSingleton, cls).__new__(cls)
            obj.__init__(*args, **kwargs)

            cls._singleton = obj

        return cls._singleton

        
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