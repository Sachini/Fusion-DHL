import argparse
import json
import os
import os.path as osp
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
from nn.dataset_train import FlowTrainDataset
from nn.unet import UNetBiliner

TRAIN_CONFIG = {'lr': 0.001, 'batch_size_train': 16, 'batch_size_val': 16, 's_factor': 0.1, 's_patience': 10,
                'epochs': 100000, 'save_interval': 10}
_e = 1e-12

_patchsize = 250


# write parser arguments
def write_config(args):
    if args.out_dir:
        with open(osp.join(args.out_dir, 'config.json'), 'w') as f:
            values = vars(args)
            values['config'] = TRAIN_CONFIG
            json.dump(values, f)


# compute loss for the whole dataset at any step during training
def run_test(network, data_loader, device, criterion, eval_mode=True):
    loss_all = []
    if eval_mode:
        network.eval()
    for bid, (feat, targ, occupancy) in enumerate(data_loader):
        feat = feat.permute(0, 3, 1, 2)
        targ = targ.permute(0, 3, 1, 2)
        pred = network(feat.to(device))
        loss = criterion(torch.stack((occupancy.to(device), occupancy.to(device)), 1) * pred, targ.to(device))
        loss_all.append(loss.cpu().detach().numpy())
    loss_all = np.stack(loss_all, axis=0)
    return loss_all


# training the network
def train(args, config):
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else 'cpu')
    print('Load network: UNet bilinear| device {}'.format(device))
    MyNetwork = UNetBiliner(out_channels=2, in_channels=6)
    network = MyNetwork.to(device)

    start_t = time.time()
    patch_size = _patchsize

    train_dataset = FlowTrainDataset(args.real_dataset, args.real_floorplans, args.real_train_list,
                                     args.syn_dataset, args.syn_floorplans, args.syn_train_list,
                                     args.real_dscale_factor, args.syn_dscale_factor,
                                     correct_size_map=args.corrsize_floorplans, test_phase=False,
                                     transform=["flip", "rot90", "rot"])
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG['batch_size_train'], shuffle=True, num_workers=2)
    end_t = time.time()
    print('Training set loaded. Length: {}, Time usage: {:.3f}s'.format(
        len(train_dataset), end_t - start_t))

    val_dataset, val_loader = None, None
    if args.real_val_list is not None:
        val_dataset = FlowTrainDataset(args.real_dataset, args.real_floorplans, args.real_val_list,
                                       args.syn_dataset, args.syn_floorplans, args.syn_val_list,
                                       args.real_dscale_factor, args.syn_dscale_factor,
                                       correct_size_map=args.corrsize_floorplans, test_phase=False,
                                       transform=["flip", "rot90", "rot"])
        val_loader = DataLoader(val_dataset, batch_size=TRAIN_CONFIG['batch_size_val'], shuffle=True, num_workers=2)

    summary_writer = None
    if args.out_dir is not None:
        if not osp.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        write_config(args)
        if not osp.isdir(osp.join(args.out_dir, 'checkpoints')):
            os.makedirs(osp.join(args.out_dir, 'checkpoints'))
        if not osp.isdir(osp.join(args.out_dir, 'logs')):
            os.makedirs(osp.join(args.out_dir, 'logs'))

    criterion = torch.nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(network.parameters(), TRAIN_CONFIG['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=TRAIN_CONFIG['s_factor'],
                                                           patience=TRAIN_CONFIG['s_patience'],
                                                           verbose=True,
                                                           eps=1e-12)

    print('Number of train samples: {}'.format(len(train_dataset)))
    if val_dataset:
        print('Number of val samples: {}'.format(len(val_dataset)))
    total_params = network.get_num_params()
    print('Total number of parameters: ', total_params)

    step = 0
    best_val_loss = np.inf

    start_epoch = 0
    if args.continue_from is not None and osp.exists(args.continue_from):
        checkpoints = torch.load(args.continue_from, map_location='cpu')
        start_epoch = checkpoints.get('epoch', 0)
        network.load_state_dict(checkpoints.get('model_state_dict'))
        optimizer.load_state_dict(checkpoints.get('optimizer_state_dict'))
    if args.out_dir is not None and osp.exists(osp.join(args.out_dir, 'logs')):
        summary_writer = SummaryWriter(osp.join(args.out_dir, 'logs'))
        summary_writer.add_text('info', 'total_param: {}'.format(total_params))

    print('Start from epoch {}'.format(start_epoch))
    total_epoch = start_epoch
    train_losses_all, val_losses_all = [], []

    init_train_loss = run_test(network, train_loader, device, criterion, eval_mode=False)
    train_losses_all.append(np.mean(init_train_loss))
    print('-------------------------')
    print('Init: average loss: {:.6f}'.format(train_losses_all[-1]))
    if summary_writer is not None:
        add_summary(summary_writer, init_train_loss, 0, 'train')

    if val_loader is not None:
        init_val_loss = run_test(network, val_loader, device, criterion)
        val_losses_all.append(np.mean(init_val_loss))
        print('Validation loss: {:.6f}'.format(val_losses_all[-1]))
        if summary_writer is not None:
            add_summary(summary_writer, init_val_loss, 0, 'val')

    try:
        for epoch in range(start_epoch, TRAIN_CONFIG['epochs']):
            saved = False
            start_t = time.time()
            network.train()
            train_losses = []
            for batch_id, (feat, targ, occupancy) in enumerate(train_loader):
                feat = feat.permute(0, 3, 1, 2)
                targ = targ.permute(0, 3, 1, 2)

                feat, targ, occupancy = feat.to(device), targ.to(device), occupancy.to(device)
                optimizer.zero_grad()
                pred = network(feat)
                loss = criterion(torch.stack((occupancy, occupancy), 1) * pred, targ)
                train_losses.append(loss.cpu().detach().numpy())
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                step += 1
            train_losses = np.stack(train_losses, axis=0)

            end_t = time.time()
            train_losses_all.append(np.average(train_losses))
            print('-------------------------')
            print('Epoch {}, time usage: {:.3f}s, average loss: {:.6f}'.format(
                epoch, end_t - start_t, train_losses_all[-1]))

            if summary_writer is not None:
                add_summary(summary_writer, train_losses, epoch + 1, 'train')
                summary_writer.add_scalar('optimizer/lr', optimizer.param_groups[0]['lr'], epoch)

            if val_loader is not None:
                network.eval()
                val_losses = run_test(network, val_loader, device, criterion)
                avg_loss = np.average(val_losses)
                print('Validation loss: {:.6f}'.format(avg_loss))
                scheduler.step(avg_loss)
                if summary_writer is not None:
                    add_summary(summary_writer, val_losses, epoch + 1, 'val')
                val_losses_all.append(avg_loss)

                # save the checkpoint with the best validation loss
                if avg_loss < best_val_loss:
                    best_val_loss = avg_loss
                    if args.out_dir and osp.isdir(args.out_dir):
                        saved = True
                        model_path = osp.join(args.out_dir, 'checkpoints', 'best_checkpoint_%d.pt' % epoch)
                        torch.save({'model_state_dict': network.state_dict(),
                                    'epoch': epoch,
                                    'optimizer_state_dict': optimizer.state_dict()}, model_path)
                        print('Model saved to ', model_path)
            if not saved and epoch % TRAIN_CONFIG['save_interval'] == 0 and args.out_dir is not None and osp.isdir(
                    args.out_dir):
                model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_%d.pt' % epoch)
                torch.save({'model_state_dict': network.state_dict(),
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict()}, model_path)
                print('Model saved to ', model_path)

            total_epoch = epoch

    except KeyboardInterrupt:
        print('-' * 60)
        print('Early terminate')

    print('Training complete')
    if args.out_dir:
        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_latest.pt')
        torch.save({'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': total_epoch}, model_path)
        print('Checkpoint saved to ', model_path)

    return train_losses_all, val_losses_all


def add_summary(writer, loss, step, mode):
    writer.add_scalar('{}_loss/avg'.format(mode), np.mean(loss), step)


def test_plot_flows(args, config):
    start_t = time.time()
    test_dataset = FlowTrainDataset(args.real_dataset, args.real_floorplans, args.real_test_list,
                                    args.syn_dataset, args.syn_floorplans, args.syn_test_list, args.real_dscale_factor,
                                    args.syn_dscale_factor,
                                    correct_size_map=args.corrsize_floorplans, test_phase=True,
                                    transform=["flip", "rot90", "rot"])

    test_loader = DataLoader(test_dataset, batch_size=TRAIN_CONFIG['batch_size_train'], shuffle=False)
    end_t = time.time()
    print('Test set loaded. Length: {}, Time usage: {:.3f}s'.format(
        len(test_dataset), end_t - start_t))

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else 'cpu')
    MyNetwork = UNetBiliner(out_channels=2, in_channels=6)
    network = MyNetwork.to(device)

    if osp.exists(args.continue_from):
        print('Loading saved model ...')
        checkpoints = torch.load(args.continue_from, map_location=device)
        network.load_state_dict(checkpoints.get('model_state_dict'))
    else:
        raise ValueError("Checkpoint does not exist")

    if not osp.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not osp.exists(osp.join(args.out_dir, 'output')):
        os.makedirs(osp.join(args.out_dir, 'output'))
    if not osp.exists(osp.join(args.out_dir, 'plots')):
        os.makedirs(osp.join(args.out_dir, 'plots'))
    summary_writer = SummaryWriter(osp.join(args.out_dir, 'output'))

    criterion = torch.nn.SmoothL1Loss()
    loss_all = []
    network.eval()
    target_all, pred_all = [], []
    i = 1
    for bid, (feat, targ, occupancy) in enumerate(test_loader):

        feat = feat.permute(0, 3, 1, 2)
        targ = targ.permute(0, 3, 1, 2)

        pred = network(feat[:, 0:6, :, :].to(device))
        loss = criterion(torch.stack((occupancy.to(device), occupancy.to(device)), 1) * pred, targ.to(device))
        loss_all.append(loss.cpu().detach().numpy())

        feat = feat.permute(0, 2, 3, 1)
        targ = targ.permute(0, 2, 3, 1)
        pred = pred.permute(0, 2, 3, 1)

        for k in range(feat.shape[0]):
            # get the floorplan (map), trajectory(raw), gt flows(flow), and trajectory's occupancy map(occupancy)
            map_traj = feat[k]
            flow = targ[k].numpy()
            occupancyy = occupancy[k].numpy()
            map = map_traj[:, :, 0:3]
            raw = map_traj[:, :, 3:]

            fig, ax = plt.subplots(1, 3, frameon=False)
            ax[0].set_axis_off()
            ax[1].set_axis_off()
            ax[2].set_axis_off()
            ax[0].title.set_text('groundtruth flow')
            ax[1].title.set_text('network predicted flow')
            ax[2].title.set_text('input trajectory')
            ax[0].imshow(raw)

            # plot the groundtruth flows
            a = np.transpose(np.nonzero(occupancyy))
            for j, x in enumerate(a):
                if j:
                    ax[0].plot([x[1], x[1] + flow[x[0], x[1], 1] / 4], [x[0], x[0] + flow[x[0], x[1], 0] / 4],
                               color='white', linewidth=1)

            # get the flow from network
            flow = pred[k].cpu().detach().numpy()

            ax[1].imshow(raw)
            # plot the network output flows
            a = np.transpose(np.nonzero(occupancyy))
            for j, x in enumerate(a):
                if j:
                    ax[1].plot([x[1], x[1] + flow[x[0], x[1], 1] / (4)], [x[0], x[0] + flow[x[0], x[1], 0] / (4)],
                               color='white', linewidth=1)

            # plot the input trajectory on the map
            ax[2].imshow(map)
            ax[2].imshow(1 - raw)
            plt.savefig(osp.join(args.out_dir, 'plots', '%d_flows.png' % i))
            plt.close(fig)
            i += 1

    loss_all = np.stack(loss_all, axis=0)

    print(loss_all)

    summary_writer.close()


def test_plot_traj(args, config):
    start_t = time.time()
    test_dataset = FlowTrainDataset(args.real_dataset, args.real_floorplans, args.real_test_list,
                                    args.syn_dataset, args.syn_floorplans, args.syn_test_list, args.real_dscale_factor,
                                    args.syn_dscale_factor,
                                    correct_size_map=args.corrsize_floorplans, test_phase=True,
                                    transform=["flip", "rot90", "rot"])
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    end_t = time.time()
    print('Test set loaded. Length: {}, Time usage: {:.3f}s'.format(
        len(test_dataset), end_t - start_t))

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else 'cpu')
    MyNetwork = UNetBiliner(out_channels=2, in_channels=6)
    network = MyNetwork.to(device)

    if osp.exists(args.continue_from):
        print('Loading saved model ...')
        checkpoints = torch.load(args.continue_from, map_location=device)
        network.load_state_dict(checkpoints.get('model_state_dict'))
    else:
        raise ValueError("Checkpoint does not exist")

    if not osp.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not osp.exists(osp.join(args.out_dir, 'output')):
        os.makedirs(osp.join(args.out_dir, 'output'))
    if not osp.exists(osp.join(args.out_dir, 'plots')):
        os.makedirs(osp.join(args.out_dir, 'plots'))
    summary_writer = SummaryWriter(osp.join(args.out_dir, 'output'))

    criterion = torch.nn.SmoothL1Loss()
    loss_all = []
    network.eval()
    target_all, pred_all = [], []
    i = 1
    for bid, (feat, targ, occupancy) in enumerate(test_loader):

        feat = feat.permute(0, 3, 1, 2)
        targ = targ.permute(0, 3, 1, 2)

        pred = network(feat[:, 0:6, :, :].to(device))
        loss = criterion(torch.stack((occupancy.to(device), occupancy.to(device)), 1) * pred, targ.to(device))
        loss_all.append(loss.cpu().detach().numpy())

        feat = feat.permute(0, 2, 3, 1)
        targ = targ.permute(0, 2, 3, 1)
        pred = pred.permute(0, 2, 3, 1)

        for k in range(feat.shape[0]):
            # get the floorplan (map), trajectory(raw), gt flows(flow), and trajectory's occupancy map(occupancy)
            map_traj = feat[k].detach().cpu().numpy()
            flow = targ[k].numpy()
            occupancyy = occupancy[k].numpy()
            pred_flow = pred[k].cpu().detach().numpy()

            map = map_traj[:, :, 0:3]
            raw = map_traj[:, :, 3:]

            fig, ax = plt.subplots(1, 3, frameon=False)
            ax[0].set_axis_off()
            ax[1].set_axis_off()
            ax[2].set_axis_off()
            ax[0].title.set_text('groundtruth flow')
            ax[1].title.set_text('network predicted flow')
            ax[2].title.set_text('input trajectory')

            # plot the corrected trajectory using groundtruth flows
            a = np.transpose(np.nonzero(occupancyy))
            results, flow_result, color = [], [], []
            for j, x in enumerate(a):
                if j:
                    results.append([x[1], x[0]])
                    flow_result.append(
                        [flow[x[0], x[1], 1] / (4), flow[x[0], x[1], 0] / (4), pred_flow[x[0], x[1], 1] / (4),
                         pred_flow[x[0], x[1], 0] / (4)])
                    color.append(raw[x[0], x[1], :])

            results = np.array(results)
            flow_result = np.array(flow_result)
            color = np.array(color)

            ax[0].imshow(map)
            ax[0].scatter(results[:, 0] + flow_result[:, 0], results[:, 1] + flow_result[:, 1], color=1 - color, s=0.5)

            # plot the corrected trajectory using network flows
            ax[1].imshow(map)
            ax[1].scatter(results[:, 0] + flow_result[:, 2], results[:, 1] + flow_result[:, 3], color=1 - color, s=0.5)

            # plot the input trajectory on the map
            ax[2].imshow(map)
            ax[2].imshow(1 - raw)
            plt.savefig(osp.join(args.out_dir, 'plots', '%d_traj.png' % i))
            plt.close(fig)
            i += 1

    loss_all = np.stack(loss_all, axis=0)

    print(loss_all)

    summary_writer.close()


if __name__ == '__main__':
    with open(osp.join('../data_paths.json'), 'r') as f:
        default_config = json.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument('--corrsize_floorplans', type=bool, default=True,
                        help='whether your floorplan image has the correct resolution')
    parser.add_argument('--real_dscale_factor', type=int, default=1,
                        help='the factor you want to downscale your floorplan image(of real dataset) by')
    parser.add_argument('--real_floorplans', type=str, help='path to floorplans used in real dataset')
    parser.add_argument('--real_train_list', type=str,
                        help='path to datalist of real train dataset, generated by data_generation script')
    parser.add_argument('--real_val_list', type=str, default=None,
                        help='path to datalist of real validation dataset, generated by data_generation script')
    parser.add_argument('--real_test_list', type=str, default=None,
                        help='path to datalist of real test dataset, generated by data_generation script')
    parser.add_argument('--real_dataset', type=str, default=None,
                        help='path to real dataset, generated by data_generation script')
    parser.add_argument('--syn_dscale_factor', type=int, default=1,
                        help='the factor you want to downscale your floorplan image(of synthetic dataset) by')
    parser.add_argument('--syn_floorplans', type=str)
    parser.add_argument('--syn_train_list', type=str,
                        help='path to datalist of synthetic train dataset, generated by data_generation script')
    parser.add_argument('--syn_val_list', type=str, default=None,
                        help='path to datalist of synthetic validation dataset, generated by data_generation script')
    parser.add_argument('--syn_test_list', type=str, default=None,
                        help='path to datalist of synthetic test dataset, generated by data_generation script')
    parser.add_argument('--syn_dataset', type=str, default=None,
                        help='path to synthetic dataset, generated by data_generation script')
    parser.add_argument('--out_dir', type=str, default=None, help='path to output files')

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test_plot_flow', 'test_plot_traj'])

    parser.add_argument('--continue_from', type=str, default=None, help='path to saved checkpoint')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})

    if args.mode == 'train':
        train(args, default_config)
    elif args.mode == 'test_plot_flow':
        test_plot_flows(args, default_config)
    elif args.mode == 'test_plot_traj':
        test_plot_traj(args, default_config)
    else:
        raise ValueError('Undefined mode')
