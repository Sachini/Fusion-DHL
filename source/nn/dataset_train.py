import json
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from skimage.transform import resize
from torch.utils.data import Dataset

# matplotlib.use('TkAgg')

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
from nn.data_generator_train_real import pad_function

torch.manual_seed(1)
np.random.seed(1)

"""
For training. Load patch images and stack with corresponding floorplan patches
"""

_patch_size = 250


class FlowTrainDataset(Dataset):
    def __init__(self, real_dataset_dir, real_floorplan_dir, real_datalist_dir,
                 syn_dataset_dir=None, syn_floorplan_dir=None, syn_datalist_dir=None,
                 real_dscale_factor=1, syn_dscale_factor=1, correct_size_map=True, door_color='brown', test_phase=False,
                 transform=None):
        self.real_dataset_dir = real_dataset_dir
        self.real_floorplan_dir = real_floorplan_dir
        self.real_datalist_dir = real_datalist_dir
        self.syn_dataset_dir = syn_dataset_dir
        self.syn_floorplan_dir = syn_floorplan_dir
        self.syn_datalist_dir = syn_datalist_dir
        self.real_dscale_factor = real_dscale_factor
        self.syn_dscale_factor = syn_dscale_factor
        self.correct_size_map = correct_size_map
        self.door_color = door_color
        self.test_phase = test_phase
        self.transform = transform

        self.real_original_psize = int(real_dscale_factor * _patch_size)
        self.syn_original_psize = int(syn_dscale_factor * _patch_size)
        self.patchs_new = _patch_size

        print('loading maps and padding them')
        with open(osp.join(self.real_floorplan_dir, 'map_info.json'), 'r') as f:
            self.real_map_config = json.load(f)
        if 'tasc1_7000' in self.real_map_config:
            del (self.real_map_config['tasc1_7000'])

        if self.door_color == 'brown':
            for key in self.real_map_config:
                self.real_map_config[key]['image'] = plt.imread(osp.join(self.real_floorplan_dir, os.path.splitext(
                    self.real_map_config[key]['map'])[0] + '_brown.png'))
        else:
            for key in self.real_map_config:
                self.real_map_config[key]['image'] = plt.imread(
                    osp.join(self.real_floorplan_dir, os.path.splitext(self.real_map_config[key]['map'])[0] + '.png'))

        if self.correct_size_map:
            for key in self.real_map_config:
                if self.real_map_config[key]["image"].shape[0] < self.patchs_new:
                    self.real_map_config[key]["ver_pad"], real_image_pad = pad_function(
                        self.real_map_config[key]["image"], self.patchs_new, "ver")
                else:
                    real_image_pad = self.real_map_config[key]["image"][:, :, 0:3]
                    self.real_map_config[key]["ver_pad"] = 0

                if real_image_pad.shape[1] < self.patchs_new:
                    self.real_map_config[key]["hor_pad"], self.real_map_config[key]["padded"] = pad_function(
                        real_image_pad, self.patchs_new, "hor")
                else:
                    self.real_map_config[key]["padded"] = real_image_pad
                    self.real_map_config[key]["hor_pad"] = 0
        else:
            for key in self.real_map_config:
                if self.real_map_config[key]["image"].shape[0] < self.real_original_psize:
                    self.real_map_config[key]["ver_pad"], real_image_pad = pad_function(
                        self.real_map_config[key]["image"], self.real_original_psize, "ver")
                else:
                    real_image_pad = self.real_map_config[key]["image"][:, :, 0:3]
                    self.real_map_config[key]["ver_pad"] = 0

                if real_image_pad.shape[1] < self.real_original_psize:
                    self.real_map_config[key]["hor_pad"], self.real_map_config[key]["padded"] = pad_function(
                        real_image_pad, self.real_original_psize, "hor")
                else:
                    self.real_map_config[key]["padded"] = real_image_pad
                    self.real_map_config[key]["hor_pad"] = 0

        if syn_datalist_dir:
            print('loading mall maps and padding them')
            with open(osp.join(self.syn_floorplan_dir, 'map_info.json'), 'r') as f:
                self.syn_map_config = json.load(f)

            for key in self.syn_map_config:
                self.syn_map_config[key]['image'] = plt.imread(
                    osp.join(self.syn_floorplan_dir, self.syn_map_config[key]['map']))

            if self.correct_size_map:
                for key in self.syn_map_config:
                    if self.syn_map_config[key]["image"].shape[0] < self.patchs_new:
                        self.syn_map_config[key]["ver_pad"], syn_image_pad = pad_function(
                            self.syn_map_config[key]["image"], self.patchs_new, "ver")
                    else:
                        syn_image_pad = self.syn_map_config[key]["image"][:, :, 0:3]
                        self.syn_map_config[key]["ver_pad"] = 0

                    if syn_image_pad.shape[1] < self.patchs_new:
                        self.syn_map_config[key]["hor_pad"], self.syn_map_config[key]["padded"] = pad_function(
                            syn_image_pad, self.patchs_new, "hor")
                    else:
                        self.syn_map_config[key]["padded"] = syn_image_pad
                        self.syn_map_config[key]["hor_pad"] = 0
            else:
                for key in self.syn_map_config:
                    if self.syn_map_config[key]["image"].shape[0] < self.syn_original_psize:
                        self.syn_map_config[key]["ver_pad"], syn_image_pad = pad_function(
                            self.syn_map_config[key]["image"], self.syn_original_psize, "ver")
                    else:
                        syn_image_pad = self.syn_map_config[key]["image"][:, :, 0:3]
                        self.syn_map_config[key]["ver_pad"] = 0

                    if syn_image_pad.shape[1] < self.syn_original_psize:
                        self.syn_map_config[key]["hor_pad"], self.syn_map_config[key]["padded"] = pad_function(
                            syn_image_pad, self.syn_original_psize, "hor")
                    else:
                        self.syn_map_config[key]["padded"] = syn_image_pad
                        self.syn_map_config[key]["hor_pad"] = 0

        self.traj_data = {}
        self.traj_data['patches'] = []
        if syn_datalist_dir:
            print('loading synthetic trajectory info')
            with open(self.syn_datalist_dir, 'r') as f:
                for line in f:
                    if line[0] == '#' or len(line.strip()) == 0:
                        continue
                    params = line.strip().split()
                    self.traj_data['patches'].append({
                        'syn': True,
                        'data': params[0],
                        'map_name': params[1],
                        'start_v': int(params[2]),
                        'start_h': int(params[3]),
                    })

        print('loading trajectory info')
        with open(self.real_datalist_dir, 'r') as f:
            for line in f:
                if line[0] == '#' or len(line.strip()) == 0:
                    continue
                params = line.strip().split()
                self.traj_data['patches'].append({
                    'syn': False,
                    'data': params[0],
                    'map_name': params[1],
                    'start_v': int(params[2]),
                    'start_h': int(params[3]),
                })

    def trans(self, floor, traj, opt_flow, occ):

        if "flip" in self.transform:
            a = torch.rand(6)
            if a[0] > 0.5:
                floor = np.copy(np.flipud(floor))
                traj = np.copy(np.flipud(traj))
                opt_flow = np.copy(np.flipud(opt_flow))
                occ = np.copy(np.flipud(occ))
                opt_flow[:, :, 0] = -opt_flow[:, :, 0]
            if a[1] >= 0.5:
                floor = np.copy(np.fliplr(floor))
                traj = np.copy(np.fliplr(traj))
                opt_flow = np.copy(np.fliplr(opt_flow))
                occ = np.copy(np.fliplr(occ))
                opt_flow[:, :, 1] = -opt_flow[:, :, 1]
        if "rot90" in self.transform:
            if a[2] > 0.5:
                # rotate 90 degree to right
                floor = np.copy(np.rot90(floor, k=3))
                traj = np.copy(np.rot90(traj, k=3))
                opt_flow = np.copy(np.rot90(opt_flow, k=3))
                occ = np.copy(np.rot90(occ, k=3))
                temp0 = np.copy(opt_flow[:, :, 0])
                opt_flow[:, :, 0] = np.copy(opt_flow[:, :, 1])
                opt_flow[:, :, 1] = -np.copy(temp0)
            if a[3] >= 0.5:
                # rotate 90 degree to left
                floor = np.copy(np.rot90(floor))
                traj = np.copy(np.rot90(traj))
                opt_flow = np.copy(np.rot90(opt_flow))
                occ = np.copy(np.rot90(occ))
                temp0 = np.copy(opt_flow[:, :, 0])
                opt_flow[:, :, 0] = -np.copy(opt_flow[:, :, 1])
                opt_flow[:, :, 1] = np.copy(temp0)
        if "rot" in self.transform:
            if a[4] > 0.6:
                angle = a[5] * 90

                # rotate angle degree to the left, interpolate map and trajectory and clip
                floor = np.copy(
                    scipy.ndimage.rotate(floor, angle, reshape=False, order=3, mode='constant', cval=0.52549016,
                                         prefilter=True))
                floor[floor < 0] = 0
                floor[floor > 1] = 1
                traj = np.copy(scipy.ndimage.rotate(traj, angle, reshape=False, order=3, mode='constant', cval=0.0,
                                                    prefilter=True))
                traj[traj < 0] = 0
                traj[traj > 1] = 1

                # rotate flow and occupancy, don't interpolate (nearest neighbour)
                temp0 = np.copy(opt_flow[:, :, 0])
                opt_flow[:, :, 0] = np.copy(-np.sin(angle / 180 * np.pi) * opt_flow[:, :, 1]) + np.copy(
                    np.cos(angle / 180 * np.pi) * opt_flow[:, :, 0])
                opt_flow[:, :, 1] = np.copy(np.cos(angle / 180 * np.pi) * opt_flow[:, :, 1]) + np.copy(
                    np.sin(angle / 180 * np.pi) * temp0)

                opt_flow = np.copy(
                    scipy.ndimage.rotate(opt_flow, angle, reshape=False, order=0, mode='constant', cval=0.0,
                                         prefilter=False))
                occ = np.copy(scipy.ndimage.rotate(occ, angle, reshape=False, order=0, mode='constant', cval=0.0,
                                                   prefilter=False))

        return floor, traj, opt_flow, occ

    def __getitem__(self, index):
        # initializing

        if self.traj_data['patches'][index]['syn']:
            floorplan_old = self.syn_map_config[self.traj_data['patches'][index]['map_name']]["padded"]
            raw_old = plt.imread(osp.join(self.syn_dataset_dir, self.traj_data['patches'][index]['data']) + '_raw.png')
            flow_old = np.load(
                osp.join(self.syn_dataset_dir, 'flows', self.traj_data['patches'][index]['data'] + '_flow.npy'))

        else:
            floorplan_old = self.real_map_config[self.traj_data['patches'][index]['map_name']]["padded"]
            raw_old = plt.imread(osp.join(self.real_dataset_dir, self.traj_data['patches'][index]['data']) + '_raw.png')
            flow_old = np.load(
                osp.join(self.real_dataset_dir, 'flows', self.traj_data['patches'][index]['data'] + '_flow.npy'))

        start_ver = self.traj_data['patches'][index]['start_v']
        start_hor = self.traj_data['patches'][index]['start_h']

        # making the floorplan colors look right
        floorplan = 1 - floorplan_old

        # removing fourth channel of trajectory, removing occupancy channel from flow
        # if in test phase, also pass the transparency for plotting purpose
        if self.test_phase:
            raw = 1 - np.copy(raw_old)
        else:
            raw = 1 - np.copy(raw_old[:, :, 0:3])
        flow = np.copy(flow_old[:, :, 0:2])
        occupancy = np.copy(flow_old[:, :, 2])

        # resizing the floormap image and making it binary (trajectory and flow are already resized)
        if self.correct_size_map:
            if self.traj_data['patches'][index]['syn']:
                f = np.copy(floorplan[start_ver:start_ver + self.patchs_new, start_hor:start_hor + self.patchs_new])
            else:
                f = np.copy(floorplan[start_ver:start_ver + self.patchs_new, start_hor:start_hor + self.patchs_new])
        else:
            if self.traj_data['patches'][index]['syn']:
                f = resize(np.copy(floorplan[start_ver:start_ver + self.syn_original_psize,
                                   start_hor:start_hor + self.syn_original_psize]), (self.patchs_new, self.patchs_new))
            else:
                f = resize(np.copy(floorplan[start_ver:start_ver + self.real_original_psize,
                                   start_hor:start_hor + self.real_original_psize]), (self.patchs_new, self.patchs_new))

        # applying transformations
        f_transform, raw_transform, flow_transform, occupancy_transform = self.trans(f, raw, flow, occupancy)

        occupancy_transform[occupancy_transform < 0.6] = 0
        occupancy_transform[occupancy_transform != 0] = 1

        return np.concatenate([f_transform, raw_transform], axis=-1), flow_transform, occupancy_transform

    def __len__(self):
        return len(self.traj_data['patches'])
