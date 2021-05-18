"""
Adopted from official implementation https://github.com/Sachini/ronin
"""


import json
import sys
from os import path as osp

import h5py
import numpy as np
import quaternion
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset, DataLoader

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
from preprocessing.model_resnet1d import *


class StridedSequenceData(Dataset):
    """ Load a single data sequence"""
    def __init__(self, data_path, step_size=10, window_size=200, **kwargs):
        super(StridedSequenceData, self).__init__()
        self.feature_dim = 6
        self.target_dim = 2
        self.window_size = window_size
        self.step_size = step_size
        self.interval = kwargs.get('interval', window_size)

        self.data_path = data_path
        self.index_map = []
        self.ts = []

        with open(osp.join(data_path, 'info.json')) as f:
            self.info = json.load(f)

        with h5py.File(osp.join(data_path, 'data.hdf5'), 'r') as f:
            # use uncalibrated data
            ts = np.copy(f['synced/time'])
            gyro = np.copy(f['synced/gyro'])
            acce = np.copy(f['synced/acce'])
            ori = np.copy(f['synced/game_rv'])

        # Compute the IMU orientation in the Tango coordinate frame.
        ori_q = quaternion.from_float_array(ori)
        gyro_q = quaternion.from_float_array(np.concatenate([np.zeros([gyro.shape[0], 1]), gyro], axis=1))
        acce_q = quaternion.from_float_array(np.concatenate([np.zeros([acce.shape[0], 1]), acce], axis=1))
        glob_gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:]
        glob_acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]

        start_frame = self.info.get('start_frame', 0)
        self.ts = ts[start_frame:]
        self.features = np.concatenate([glob_gyro, glob_acce], axis=1)[start_frame:]
        feat_sigma = kwargs.get('feature_sigma,', -1)
        if feat_sigma > 0:
            self.features = gaussian_filter1d(self.features, sigma=feat_sigma, axis=0)
        self.index_map = np.arange(0, self.features.shape[0] - self.interval, step_size)

    def __getitem__(self, item):
        frame_id = self.index_map[item]
        feat = self.features[frame_id:frame_id + self.window_size]

        return feat.astype(np.float32).T, frame_id

    def __len__(self):
        return len(self.index_map)


def test_sequence(test_path, **kwargs):
    if not torch.cuda.is_available() or kwargs.get('cpu', False):
        device = torch.device('cpu')
        checkpoint = torch.load(kwargs.get('model_path'), map_location=lambda storage, location: storage)
    else:
        device = torch.device('cuda:0')
        checkpoint = torch.load(kwargs.get('model_path'))

    seq_dataset = StridedSequenceData(test_path, **kwargs)
    input_channel, output_channel = seq_dataset.feature_dim, seq_dataset.target_dim
    seq_loader = DataLoader(seq_dataset, batch_size=1024, shuffle=False)

    fc_config = {'fc_dim': 512, 'in_dim': kwargs.get('window_size') // 32 + 1, 'dropout': 0.5, 'trans_planes': 128}
    network = ResNet1D(input_channel, output_channel, BasicBlock1D, [2, 2, 2, 2],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **fc_config)

    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval().to(device)
    print('Model {} loaded to device {}.'.format(kwargs.get('model_path'), device))

    predictions = []
    network.eval()
    for bid, (feat, _) in enumerate(seq_loader):
        pred = network(feat.to(device)).cpu().detach().numpy()
        predictions.append(pred)
    predictions = np.concatenate(predictions, axis=0)

    # Reconstruct trajectory with predicted global velocities.
    ts = seq_dataset.ts
    ind = seq_dataset.index_map.astype(np.int)
    dts = np.mean(ts[ind[1:]] - ts[ind[:-1]])
    pos = np.zeros([predictions.shape[0] + 2, 2])
    pos[1:-1] = np.cumsum(predictions[:, :2] * dts, axis=0) + pos[0]
    pos[-1] = pos[-2]
    ts_ext = np.concatenate([[ts[0] - 1e-06], ts[ind], [ts[-1] + 1e-06]], axis=0)
    pos = interp1d(ts_ext, pos, axis=0)(ts)[:, :2]

    return pos
