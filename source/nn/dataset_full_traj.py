import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave
from skimage.transform import resize
from torch.utils.data import Dataset

matplotlib.use('TkAgg')
_dpi = 2.5
_is = 1 / 1000

"""
For evaluation only. Load data for a full trajectory and segment into a specific patch size. Input is a single 
trajectory path
"""


class FlowEvalDataset(Dataset):
    def __init__(self, data_path, floorplan_path, result_dpi=_dpi, image_dpi=_dpi, patch_size=250, buffer=10,
                 time_seg=120 * 50):
        self.data_path = data_path
        self.patch_size, self.tp, self.td = patch_size, patch_size // 2, patch_size / 2 - buffer
        self.time_seg = time_seg

        self.result_dpi = result_dpi
        self.floorplan = plt.imread(floorplan_path)[:, :, :3]
        if image_dpi != result_dpi:
            resized_size = (np.asarray(self.floorplan.shape[:2]) / image_dpi * result_dpi).astype(np.int)
            self.floorplan = resize(self.floorplan, resized_size)
        else:
            resized_size = np.asarray(self.floorplan.shape[:2])

        self.traj = np.loadtxt(self.data_path)  # format - timestamp, pos (x, y) - final_trajectory.txt from optimizer
        self.ts = np.copy(self.traj[:, 0])
        self.traj = self.traj[:, 1:3] * self.result_dpi  # trajectory in pixels

        i, j = 0, 1
        self.data_list = []
        idx = [0, 0]
        while idx[-1] < len(self.traj) - 1:
            center, idx = self.find_patch(self.traj, i, resized_size)
            # start, stop, center of trajectory (x, y), map start indices
            self.data_list.append([*idx, *center, round(center[1]) - self.tp, round(center[0]) - self.tp])
            j += 1
            traj_length = idx[1] - idx[0]
            i = max(idx[-1] - int(traj_length // 4), i + int(traj_length // 4))

        self.data_list = np.asarray(self.data_list).astype(np.int)
        print('Loaded {} segments'.format(len(self.data_list)))

    def save_image(self, fname, array):
        imsave(fname, array)

    def find_patch(self, traj, index, shape):
        """
        find patch of size traj_patch centered on index
        :param traj_idx: trajectory index in trajectories list
        :param index: frame number of trajectory
        :param shape: map shape
        :return: trajectory, center of patch containing trajectory in pixels, [start, end idx of trajectory]
        """
        # IMPORTANT - matrix vs image indexing  is different
        c = np.asarray([max(self.tp, min((shape[1] - 1 - self.tp), traj[index, 0])),
                        max(self.tp, min((shape[0] - 1 - self.tp), traj[index, 1]))])
        # start index
        s = np.where(np.bitwise_or(np.bitwise_or(traj[:index, 0] < c[0] - self.td, traj[:index, 0] > c[0] + self.td),
                                   np.bitwise_or(traj[:index, 1] < c[1] - self.td, traj[:index, 1] > c[1] + self.td)))
        s = max(s[0]) if len(s[0]) > 0 else 0
        if self.time_seg > 0: s = max(s, index - self.time_seg)
        # end index
        e = np.where(np.bitwise_or(np.bitwise_or(traj[index:, 0] < c[0] - self.td, traj[index:, 0] > c[0] + self.td),
                                   np.bitwise_or(traj[index:, 1] < c[1] - self.td, traj[index:, 1] > c[1] + self.td)))
        e = min(e[0]) + index if len(e[0]) > 0 else len(traj)
        if self.time_seg > 0: e = min(e, index + self.time_seg)

        return c, [s, e]

    def get_traj_segment(self, index):
        info = self.data_list[index]
        traj_segment = self.traj[info[0]:info[1]] - (info[2:4] - self.tp)
        return np.round(traj_segment).astype(np.int)

    def plot_trajectory(self, traj, res=200):
        fig = plt.figure(figsize=(self.patch_size / res, self.patch_size / res), dpi=res, frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.set_xlim(0, self.patch_size)
        ax.set_ylim(self.patch_size, 0)
        cmap = matplotlib.cm.rainbow(np.linspace(0, 1, len(traj)))
        ax.scatter(traj[:, 0], traj[:, 1], color=cmap, s=_is)

        fig.canvas.draw()
        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()

        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
        buf.shape = (h, w, 4)

        buf = np.roll(buf, 3, axis=2)
        buf = buf.astype('float32')
        buf = buf / 255.0
        plt.close(fig)
        return buf

    def __getitem__(self, index):
        info = self.data_list[index]
        traj_segment = self.traj[info[0]:info[1]] - (info[2:4] - self.tp)
        raw = 1 - self.plot_trajectory(traj_segment)[:, :, :3]

        floor = 1 - self.floorplan[info[4]:info[4] + self.patch_size, info[5]:info[5] + self.patch_size]
        return np.concatenate([floor, raw], axis=-1), index

    def __len__(self):
        return len(self.data_list)
