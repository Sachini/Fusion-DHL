import argparse
import json
import os
import os.path as osp
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
from nn.dataset_full_traj import FlowEvalDataset
from nn.unet import UNetBiliner


def get_test_list(args):
    data_list = []
    if args.test_path is not None:
        # last two words are the map name. use test_list if not
        if osp.isdir(args.test_path):
            dname = osp.split(args.test_path)[1]
            fname = osp.join(args.test_path, args.file_name)
            mname = '_'.join(dname.rsplit("_")[-2:])
        else:
            dname = osp.split(osp.split(args.test_path)[0])[1]
            fname = args.test_path
            mname = '_'.join(dname.rsplit("_")[-2:])
        data_list.append([dname, fname, mname])
    elif args.test_list is not None:
        root_dir = args.data_dir if args.data_dir is not None else ''
        with open(args.test_list, 'r') as f:
            for line in f:
                if line[0] == '#' or len(line.strip()) == 0:
                    continue
                params = line.strip().split()
                dname = osp.split(params[0])[1]
                fname = osp.join(root_dir, params[0], args.file_name)
                data_list.append([dname, fname, params[1]])
    else:
        raise ValueError('Must specify test_list or test_path')

    return data_list


def test(args):
    # change the patch size, buffer (margin) and padding for unet, timecut (max segment length = timecut*2) here
    in_patch, buffer, padding, timecut = args.patch_size, 10, [1, 0, 1, 0], 2 * 60
    batch_size = 16
    sw = 101  # smoothing_window

    test_list = get_test_list(args)

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else 'cpu')
    MyNetwork = UNetBiliner(out_channels=2, in_channels=6, pad=padding)
    network = MyNetwork.to(device)

    assert osp.exists(args.model_path), "Checkpoint does not exist"

    checkpoints = torch.load(args.model_path, map_location=device)
    network.load_state_dict(checkpoints.get('model_state_dict'))

    out_dir = osp.join(args.out_dir, args.test_name)
    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    if not osp.exists(osp.join(out_dir, 'output')):
        os.makedirs(osp.join(out_dir, 'output'))
    if not osp.exists(osp.join(out_dir, 'plots')):
        os.makedirs(osp.join(out_dir, 'plots'))
    if not osp.exists(osp.join(out_dir, 'full_result')):
        os.makedirs(osp.join(out_dir, 'full_result'))

    network.eval()
    flow_factor = 4  # defined in tranining
    for data in test_list:
        print('Processing', data[0])
        dataset = FlowEvalDataset(data[1], osp.join(args.floorplan_dir, data[2] + '.png'),
                                  result_dpi=args.input_dpi, image_dpi=args.floorplan_dpi,
                                  patch_size=in_patch, buffer=buffer, time_seg=args.traj_freq * timecut)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        full_result = np.zeros([len(dataset.traj), 3])
        i = 0
        for bid, (feat, index) in enumerate(data_loader):
            index = index.numpy().astype(np.int)
            feat = feat.permute(0, 3, 1, 2)

            pred = network(feat[:, :6].to(device))
            feat = feat.permute(0, 2, 3, 1)
            pred = pred.permute(0, 2, 3, 1)

            for k in range(feat.shape[0]):
                map = feat[k].detach().cpu().numpy()
                pred_flow = pred[k].detach().cpu().numpy()
                flow_input = dataset.get_traj_segment(index[k])
                idx = dataset.data_list[index[k], :2]

                x = np.flip(flow_input, 1).astype(np.int)
                flow_result = np.stack(
                    [pred_flow[x[:, 0], x[:, 1], 1] / flow_factor, pred_flow[x[:, 0], x[:, 1], 0] / flow_factor],
                    axis=1)
                np.savetxt(osp.join(out_dir, 'output', '{}_{}.txt'.format(data[0], i)),
                           np.concatenate([flow_input, flow_result], axis=1))

                if not args.fast_test:
                    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                    color = map[x[:, 0], x[:, 1], 3:6]

                    ax[0].imshow(map[:, :, :3])
                    ax[0].scatter(flow_input[:, 0], flow_input[:, 1], color=color, s=0.1)
                    ax[0].set_title('input')
                    ax[1].imshow(map[:, :, :3])
                    ax[1].scatter(flow_input[:, 0] + flow_result[:, 0], flow_input[:, 1] + flow_result[:, 1],
                                  color=color, s=0.1)
                    ax[1].set_title('prediction')
                    plt.savefig(osp.join(out_dir, 'plots', '{}_{}.png'.format(data[0], i)))
                    plt.close(fig)
                i += 1

                # weighted average of result
                l = len(flow_input)
                w = np.arange(l, dtype=np.float) - l / 2
                w = 1 / (l / 4 * np.sqrt(2 * np.pi)) * np.exp(-w ** 2 / (2 * (l / 4) ** 2))
                w = w / max(w)
                full_result[idx[0]:idx[1], :2] += flow_result * w[:, None]
                full_result[idx[0]:idx[1], 2] += w

            del pred, feat

        avg_traj = full_result[:, :2] / full_result[:, 2:] + np.round(dataset.traj)
        if not args.no_smoothing:
            # mean filter
            avg_traj[:, 0] = np.convolve(np.pad(avg_traj[:, 0], sw // 2, mode='edge'), np.ones((sw,)) / sw,
                                         mode='valid')
            avg_traj[:, 1] = np.convolve(np.pad(avg_traj[:, 1], sw // 2, mode='edge'), np.ones((sw,)) / sw,
                                         mode='valid')
        np.savetxt(osp.join(out_dir, 'full_result', '{}.txt'.format(data[0], i)),
                   np.concatenate([dataset.ts[:, None], avg_traj / dataset.result_dpi], axis=1))
        if not args.fast_test:
            map = dataset.floorplan
            fig, ax = plt.subplots(1, 2, figsize=(20, 10))
            color = matplotlib.cm.rainbow(np.linspace(0, 1, len(avg_traj)))
            ax[0].imshow(1 - map)
            ax[0].scatter(dataset.traj[:, 0], dataset.traj[:, 1], color=color, s=0.1)
            ax[0].set_title('input')
            ax[1].imshow(1 - map)
            ax[1].scatter(avg_traj[:, 0], avg_traj[:, 1], color=color, s=0.1)
            ax[1].set_title('prediction')
            plt.savefig(osp.join(out_dir, 'full_result', '{}.png'.format(data[0], i)))
            plt.close(fig)


if __name__ == '__main__':
    with open(osp.join('../data_paths.json'), 'r') as f:
        default_config = json.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument('--floorplan_dir', type=str, default=default_config['floorplan_dir'])
    parser.add_argument('--floorplan_dpi', type=float, default=2.5)
    parser.add_argument('--input_dpi', type=float, default=2.5)

    # use test_path or (test_list, file_name and data_dir)
    parser.add_argument('--test_path', type=str, default=None, help='path to input file')
    parser.add_argument('--test_list', type=str, default=None, help='list of [folder_name map_name]')
    parser.add_argument('--file_name', type=str, default='final_trajectory.txt', help='file_name of input trajectory')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--traj_freq', type=float, default=50, help='Input data frequency (Hz)')

    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--test_name', type=str, default='output')
    parser.add_argument('--patch_size', type=int, default=250)
    parser.add_argument('--no_smoothing', action='store_true')
    parser.add_argument('--fast_test', action='store_true')

    args = parser.parse_args()

    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})
    test(args)
