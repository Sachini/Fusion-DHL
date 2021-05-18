import json
import math
import os.path as osp
import sys

import h5py
import matplotlib.pyplot as plt

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
from util.math_utils import *

_sec2nano = 1e09
_traj_freq = 200


def change_cf(ori, vectors):
    """
    Euler-Rodrigous formula v'=v+2s(rxv)+2rx(rxv)
    :param ori: quaternion [n]x4
    :param vectors: vector nx3
    :return: rotated vector nx3
    """
    assert ori.shape[-1] == 4
    assert vectors.shape[-1] == 3

    if len(ori.shape) == 1:
        ori = np.repeat([ori], vectors.shape[0], axis=0)
    q_s = ori[:, :1]
    q_r = ori[:, 1:]

    tmp = np.cross(q_r, vectors)
    vectors = np.add(np.add(vectors, np.multiply(2, np.multiply(q_s, tmp))), np.multiply(2, np.cross(q_r, tmp)))
    return vectors


def load_trajectory(data_path, init_heading=0, type='ronin', visualize=False):
    with open(osp.join(data_path, 'info.json'), 'r') as f:
        info = json.load(f)
        start_frame = info.get('start_frame', 0)

    with h5py.File(osp.join(data_path, 'data.hdf5'), 'r') as f:
        if type == 'ronin':
            traj = np.copy(f['computed/ronin_traj'])
            if traj.shape[-1] == 2:
                traj = np.concatenate([traj, np.zeros([traj.shape[0], 1])], axis=-1)
        elif type == 'gt':
            traj = np.copy(f['pose/tango_pos'])[start_frame:]
            traj -= traj[0]
        ts = np.copy(f['synced/time'])[start_frame:]

    if init_heading:
        r = np.asarray([math.cos(init_heading / 2), math.sin(init_heading / 2), 0, 0])
        traj = change_cf(r, traj)

    if visualize:
        plt.figure()
        plt.plot(traj[:, 0], traj[:, 1])
        plt.axis('equal')
        plt.show()

    return ts, traj[:, :2]


def trajectory_as_speed_angle_rate(timestamp, trajectory, interp_freq=200):
    assert _traj_freq % interp_freq == 0, "Interpolate frequency is invalid"
    w = int(_traj_freq / interp_freq)
    dt = np.mean(timestamp[w::w] - timestamp[:-w:w])

    glob_v = (trajectory[w::w] - trajectory[:-w:w]) / dt
    vel_magnitude = np.linalg.norm(glob_v, axis=1)

    yaw = np.arctan2(glob_v[:, 1], glob_v[:, 0])
    yaw = adjust_angle_arr(yaw)
    yaw_diff = (yaw[1:] - yaw[:-1])

    return timestamp[::w], vel_magnitude, yaw_diff, \
           {'pos': trajectory[0], 'yaw': yaw[0], 'yaw_xy': np.asarray([np.sin(yaw[0]), np.cos(yaw[0])])}


def trajectory_as_polar_velocity(trajectory, interp_freq=50, visualize=False, timestamp=None, traj_freq=_traj_freq):
    # specify timestmp or traj frequency
    if timestamp is None:
        assert traj_freq is not None, 'Must specify timestamp or traj_freq'
        traj_freq = 1 / np.mean(np.diff(timestamp))

    assert traj_freq % interp_freq == 0, "Interpolate frequency is invalid"
    w = int(traj_freq / interp_freq)
    dt = 1 / traj_freq * w

    glob_v = (trajectory[w::w] - trajectory[:-w:w]) / dt
    vel_norm = np.linalg.norm(glob_v, axis=1)

    yaw_traj = np.arctan2(glob_v[:, 1], glob_v[:, 0])
    yaw_traj = adjust_angle_arr(yaw_traj)
    vel_yaw = yaw_traj - yaw_traj[0]

    reconstruct_vel = np.stack([np.cos(vel_yaw), np.sin(vel_yaw)], axis=1) * vel_norm[:, None] * dt
    reconstruct_traj = np.zeros([reconstruct_vel.shape[0] + 1, 2])
    reconstruct_traj[1:] = np.cumsum(reconstruct_vel, axis=0)
    if timestamp is None:
        ts = np.arange(0, len(reconstruct_traj), 1) * dt
    else:
        ts = timestamp[::w]

    if visualize:
        plt.figure()
        plt.plot(trajectory[:, 0], trajectory[:, 1])
        plt.plot(reconstruct_traj[:, 0], reconstruct_traj[:, 1], color='r')
        plt.axis('equal')
        plt.show()
    return ts, vel_norm, vel_yaw, reconstruct_traj, {'pos': trajectory[0], 'yaw': vel_yaw[0],
                                                     'yaw_xy': np.asarray([np.sin(vel_yaw[0]), np.cos(vel_yaw[0])])}


def load_trajectory_as_speed_angle_rate(data_path, interp_freq=200, type='ronin', visualize=False, init_heading=0):
    ts, traj = load_trajectory(data_path, init_heading=init_heading, type=type, visualize=False)

    ts, vel_norm, yaw_diff, _ = trajectory_as_speed_angle_rate(ts, np.flip(traj[:, :2], axis=1), interp_freq)

    dt = np.mean(ts[1:] - ts[:-1])
    yaw = np.insert(yaw_diff, 0, [0])
    yaw = np.cumsum(yaw)

    reconstruct_vel = np.stack([np.cos(yaw), np.sin(yaw)], axis=1) * vel_norm[:, None] * dt
    reconstruct_traj = np.zeros([reconstruct_vel.shape[0] + 1, 2])
    reconstruct_traj[1:] = np.cumsum(reconstruct_vel, axis=0)

    if visualize:
        plt.figure()
        plt.plot(traj[:, 0], traj[:, 1])
        plt.plot(reconstruct_traj[:, 0], reconstruct_traj[:, 1], color='r')
        plt.axis('equal')
        plt.show()
    return ts, vel_norm, yaw_diff, reconstruct_traj


def load_trajectory_as_polar_velocity(data_path, interp_freq=200, type='ronin', visualize=False, init_heading=0):
    ts, traj = load_trajectory(data_path, init_heading=init_heading, type=type, visualize=False)
    ts, vel_norm, vel_yaw, reconstruct_traj, _ = trajectory_as_polar_velocity(np.flip(traj[:, :2], axis=1), interp_freq,
                                                                              visualize=visualize, timestamp=ts)

    return ts, vel_norm, vel_yaw, reconstruct_traj
