
import numpy as np
from pyquaternion import Quaternion



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
