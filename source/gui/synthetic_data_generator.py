import argparse
import math
import os
import os.path as osp
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button
from numpy.random import randn
from scipy.interpolate import interp1d

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
from optim.residual_functions import PositionPriorFunctor
from util.data_loader import trajectory_as_speed_angle_rate

matplotlib.use('TkAgg')
font = {'family': 'DejaVu Sans',
        'size': 4}

matplotlib.rc('font', **font)

_traj_freq = 200


def draw_walkable_path(fmap, r=2):
    walkable_map = np.zeros(fmap.shape)
    fig, ax = plt.subplots(3, 1, figsize=(25, 25), gridspec_kw={'height_ratios': [20, 1, 1]})

    draw = drawPath(fig, ax, walkable_map, r)
    draw.connect()
    ax[0].matshow(fmap[:, :], cmap='gray_r')
    plt.tight_layout()
    plt.show()
    draw.disconnect()

    return draw.walkable_map, np.asarray(draw.traj)


class drawPath:
    def __init__(self, fig, ax, walkable_map, r):
        self.traj = []
        self.fig = fig
        self.ax = ax
        self.press = False
        self.walkable_map = walkable_map
        self.r = r
        self.slider = Slider(ax[1], 'Radius', self.r // 2, self.r * 10, valinit=self.r, valstep=2)
        self.slider.on_changed(self._update_point_radius)
        self.button = Button(ax[2], 'Reset radius', color='0.85', hovercolor='0.95')
        self.button.on_clicked(self._reset_slider)

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.fig.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.fig.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.fig.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def _update_point_radius(self, value):
        self.r = value

    def _reset_slider(self, event):
        self.slider.reset()

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.ax[0]: return

        self.press = not self.press
        print("[On press] x:{} y:{}".format(event.xdata, event.ydata))

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if not self.press: return
        if event.inaxes != self.ax[0]: return

        x, y = event.xdata, event.ydata
        circle = plt.Circle((x, y), self.r, color='red')
        self.traj.append([x, y])
        self.ax[0].add_patch(circle)
        self.fig.canvas.draw()

        # update the states
        self.walkable_map[int(y - self.r):int(y + self.r), int(x - self.r):int(x + self.r)] = 1

    def on_release(self, event):
        'on release we reset the press data'
        pass

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.fig.canvas.mpl_disconnect(self.cidpress)
        self.fig.canvas.mpl_disconnect(self.cidrelease)
        self.fig.canvas.mpl_disconnect(self.cidmotion)


def adjust_to_uniform_speed(sparse_points, avg_speed=1.4, freq=_traj_freq):
    """
    Uniform speed trajectory
    :param sparse_points: 2-d array [nx2]
    :param avg_speed: average walking speed of person (m/s)
    :param freq: target trajectory frequency
    :return: new timestamp and trajectory, timestamps of input
    """
    dist = np.linalg.norm(sparse_points[1:] - sparse_points[:-1], axis=1)
    idx = np.where(dist < 1)[0]
    dist = np.cumsum(np.delete(dist, idx))
    sparse_points = np.delete(sparse_points, idx, 0)
    dist = np.insert(dist, 0, [0])

    # intermediate dense trajectory
    const_dist = np.arange(0, dist[-1] + 1)
    trajectory = interp1d(dist, sparse_points, fill_value='extrapolate', kind='cubic', axis=0)(const_dist)

    # trajectory at given frequency
    dist2 = np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1)
    dist2 = np.insert(np.cumsum(dist2), 0, [0])
    total_time = dist2[-1] / avg_speed
    target_ts = np.arange(0, total_time + 1 / freq, 1 / freq)
    uniform_dist = np.arange(0, len(target_ts)) * avg_speed / freq

    input_ts = interp1d(uniform_dist, target_ts, fill_value='extrapolate')(dist2)
    target_traj = interp1d(input_ts, trajectory, fill_value='extrapolate', axis=0)(target_ts)

    return target_ts, target_traj


def synthesise_imu_errors(ts, traj, args):
    _, vel, ang, start_param = trajectory_as_speed_angle_rate(ts, traj)

    # scale - scale factor for person + scale noise
    # factor for person is usally < 1 but allowed +/- here
    idx_scale = np.arange(0, int(math.ceil(len(ts) / args.scale_interval))) * args.scale_interval
    scale = (1 + randn(1) * args.std_scale) + randn(len(idx_scale)) * args.std_scale_noise

    # bias - long term bias (change with orientation or calibration) + bias noise
    idx_bias = np.arange(0, int(math.ceil(len(ts) / args.bias_interval))) * args.bias_interval
    idx_bias_lt = np.sort(np.random.choice(idx_bias, len(idx_bias) // 5, replace=False))
    bias_lt = interp1d(idx_bias_lt, randn(len(idx_bias_lt)) * args.std_bias, kind='previous', fill_value='extrapolate')(
        idx_bias)
    bias = bias_lt + randn(len(idx_bias)) * args.std_bias_noise

    noise_idx, noise = None, None
    if args.add_noise:
        # add random angle errors at sprase points
        noise_idx = np.random.choice(int(len(ang) * 0.8), len(idx_bias) // 10, replace=False) + int(len(ang) * 0.1)
        noise = np.random.randn(len(noise_idx)) * np.pi / 4

    if args.verbose:
        plt.figure()
        plt.subplot(211)
        plt.plot(idx_scale, scale)
        plt.title('scale')
        plt.subplot(212)
        plt.plot(idx_bias, bias)
        plt.title('bias')
        plt.show()

    func = PositionPriorFunctor(vel, ang, ts, bias_index=idx_bias, scale_index=idx_scale, noise_index=noise_idx,
                                correct_angle=False)
    result_traj = func.get_modified_trajectory(start_param['pos'], start_param['yaw_xy'], bias=bias, scale=scale,
                                               noise=noise)
    return result_traj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('map_path', type=str)
    parser.add_argument('--map_dpi', type=float, default=2.5)

    # time series
    parser.add_argument('--interpolate', type=int, default=_traj_freq,
                        help='Target frequency of trajectory data (default: 200Hz)')
    parser.add_argument('--avg_speed', type=float, default=1.4, help='Average walking speed of person')

    # smooth trajectory
    parser.add_argument('--smooth', action='store_true', help='Enable to smooth the trajectory')
    parser.add_argument('--sigma_smooth_vel', type=float, default=5, help='Sigma to smooth velocity')
    parser.add_argument('--sigma_smooth_angle', type=float, default=5, help='Sigma to smooth angle')

    # error params
    parser.add_argument('--scale_interval', type=int, default=5000,
                        help='Variable interval for piecewise linear function of scale')
    parser.add_argument('--std_scale', type=float, default=0.01, help='Sigma for constant scale offset')
    parser.add_argument('--std_scale_noise', type=float, default=0.005, help='Sigma for scale noise')

    parser.add_argument('--bias_interval', type=int, default=1000,
                        help='Variable interval for piecewise linear function of bias')
    parser.add_argument('--std_bias', type=float, default=0.00001, help='Sigma for long-term bias')
    parser.add_argument('--std_bias_noise', type=float, default=0.000005, help='Sigma for bias noise')
    parser.add_argument('--add_noise', action='store_true')

    parser.add_argument('--verbose', action='store_true', help='When set progress of optimization is printed')

    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--out_name', type=str, default='generated')

    args = parser.parse_args()

    map_name = os.path.splitext(os.path.basename(os.path.normpath(args.map_path)))[0].lower()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if not os.path.exists(os.path.join(args.out_dir, map_name)):
        os.mkdir(os.path.join(args.out_dir, map_name))

    map = plt.imread(args.map_path)

    _, sparse_points = draw_walkable_path(map, r=2)
    np.savetxt(os.path.join(args.out_dir, map_name, args.out_name + '_raw.txt'), sparse_points)

    sparse_points = sparse_points / args.map_dpi  # convert to meters
    ts, traj = adjust_to_uniform_speed(sparse_points, avg_speed=args.avg_speed, freq=args.interpolate)
    imu_traj = synthesise_imu_errors(ts, traj, args)

    if args.verbose:
        fig = plt.figure(figsize=((map.shape[1] / 200), (map.shape[0] / 200)), dpi=200, frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(map, extent=[0, map.shape[1] / args.map_dpi, map.shape[0] / args.map_dpi, 0])
        ax.scatter(sparse_points[:, 0], sparse_points[:, 1], color='k', s=1)
        ax.scatter(traj[:, 0], traj[:, 1], color='b', s=1 / 10)
        ax.scatter(imu_traj[:, 0], imu_traj[:, 1], color='g', s=1 / 10)
        plt.savefig(os.path.join(args.out_dir, map_name, args.out_name + '_opt.png'))
        plt.show()

    np.savetxt(os.path.join(args.out_dir, map_name, args.out_name + '_mod.txt'),
               np.concatenate([ts[:, None], traj, imu_traj], axis=1))
    print('Done: generated trajectory')
