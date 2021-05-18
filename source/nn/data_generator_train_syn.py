import argparse
import json
import math
import os
import os.path as osp
import random
import sys

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
from scipy.interpolate import interp1d

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
from util.math_utils import *

random.seed(0)
np.random.seed(0)

"""
For training. Load all synthetic mall data as full trajectories and segment each data into multiple random patches 
using a specific patch size
"""

_ali = 1  # align interval for computing starting angle
_dpi = 2.5  # floorplan resolution

_is = 1 / 1000  # trajectory_thickness in image

_interpolate = 200  # we'll use 1 in every 200/_interpolate synthetic data points

patch_per_traj = 25  # number of generated patches from each trajectory


def save_plot_image(startv, starth, patchs, image_size, traj_comp, filename, show_fig=False, save_fig=True,
                    give_array=False, dpi=5, res=200):
    traj = np.copy(traj_comp)
    traj[:, 0] = traj_comp[:, 0] - starth / _dpi
    traj[:, 1] = traj_comp[:, 1] - startv / _dpi

    fig = plt.figure(figsize=((patchs / res), (patchs / res)), dpi=res, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.set_xlim(0, (patchs / dpi))
    ax.set_ylim((patchs / dpi), 0)

    cmap = matplotlib.cm.rainbow(np.linspace(0, 1, len(traj)))
    ax.scatter(traj[:, 0], traj[:, 1], color=cmap, s=_is)

    if save_fig:
        plt.savefig(filename, transparent=True)
    if show_fig:
        plt.show()
    if give_array:
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

    plt.close(fig)
    pass


def save_mask_image(startv, starth, patchs, image_size, raw_comp, traj_comp, filename, give_array=True,
                    save_array=False):
    traj = np.copy(traj_comp)
    traj[:, 0] = traj_comp[:, 0] - starth / _dpi
    traj[:, 1] = traj_comp[:, 1] - startv / _dpi

    raw = np.copy(raw_comp)
    raw[:, 0] = raw_comp[:, 0] - starth / _dpi
    raw[:, 1] = raw_comp[:, 1] - startv / _dpi

    # flow is 4x the real amount of displacement (flow factor)
    flow = (traj - raw) * 4
    buf = np.zeros((image_size[0], image_size[1], 3))
    for i, (y, x) in enumerate(raw):
        if int(x * _dpi) < image_size[0] and int(y * _dpi) < image_size[1] and int(x * _dpi) >= 0 and int(
                y * _dpi) >= 0:
            buf[int(x * _dpi), int(y * _dpi), 0] = flow[i, 1] * _dpi  # vertical displacement
            buf[int(x * _dpi), int(y * _dpi), 1] = flow[i, 0] * _dpi  # horizontal displacement
            buf[int(x * _dpi), int(y * _dpi), 2] = 1  # the occupancy mask of the trajectory
    if save_array:
        np.save(filename, buf)
    if give_array:
        return np.copy(buf[:int(patchs), :int(patchs), :])
    pass


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


def open_data(gt_path, raw_path, init_heading=np.pi):
    gt_data = np.loadtxt(gt_path)
    raw_data = np.loadtxt(raw_path)

    opt = np.copy(gt_data)
    raw = np.copy(raw_data)

    if init_heading:
        a = np.asarray([math.cos(init_heading / 2), math.sin(init_heading / 2), 0, 0])
        if raw.shape[-1] == 2:
            raw = np.concatenate([raw, np.zeros([raw.shape[0], 1])], axis=1)
        raw = change_cf(a, raw)[:, :2]

    # align start point - good approximate alignment
    angle = np.arctan2(opt[_ali, 1] - opt[0, 1], opt[_ali, 0] - opt[0, 0])

    glob_v = raw[1::1] - raw[:-1:1]
    vel_magnitude = np.linalg.norm(glob_v, axis=1)

    yaw = np.arctan2(glob_v[:, 1], glob_v[:, 0])
    yaw = adjust_angle_arr(yaw)
    yaw_diff = (yaw[1:] - yaw[:-1])

    yaw = np.insert(yaw_diff, 0, [angle])
    yaw = np.cumsum(yaw)

    reconstruct_vel = np.stack([np.cos(yaw), np.sin(yaw)], axis=1) * vel_magnitude[:, None]
    reconstruct_traj = np.zeros([reconstruct_vel.shape[0] + 1, 2])
    reconstruct_traj[1:] = np.cumsum(reconstruct_vel, axis=0)
    reconstruct_traj = reconstruct_traj + opt[0]
    return opt, raw_data, reconstruct_traj


def get_processed_scale(sc, sc_ind, feat_ind):
    if sc_ind is None:
        return sc
    else:
        return interp1d(sc_ind, sc, axis=0, bounds_error=False, fill_value=1)(feat_ind)[:, None]


def calc_start_index(shape, pos, patch_size):
    pos = max(0, pos - patch_size // 2)
    i = max(0, min(shape - patch_size, pos))
    return int(round(i))


def pad_function(mapp, patch_size, orientation):
    if orientation == "ver":
        l = patch_size - mapp.shape[0]
        l_prev = l // 2
        l_after = l - l_prev
        color = [0.7372549, 0.74509805, 0.7529412]
        padded_map = np.stack(
            [np.pad(mapp[:, :, c], ((l_prev, l_after), (0, 0)), 'constant', constant_values=color[c]) for c in
             range(3)], axis=2)
        pre = l_prev

    if orientation == "hor":
        l = patch_size - mapp.shape[1]
        l_prev = l // 2
        l_after = l - l_prev
        color = [0.7372549, 0.74509805, 0.7529412]
        padded_map = np.stack(
            [np.pad(mapp[:, :, c], ((0, 0), (l_prev, l_after)), 'constant', constant_values=color[c]) for c in
             range(3)], axis=2)
        pre = l_prev

    # return the amount of padding and the padded map
    return pre, padded_map


def remove_intersection(opt, raw):
    flag = 0
    comm_dist = 0
    x0 = opt[0, 0]
    y0 = opt[0, 1]
    points_passed = np.array([[x0, y0]])

    for i, (x, y) in enumerate(opt[1:], 1):
        if i % 200 == 1:
            comm_dist = comm_dist + np.sqrt(np.power((x - x0), 2) + np.power((y - y0), 2))
            dist = np.sqrt(np.power((x - x0), 2) + np.power((y - y0), 2))
            x0 = x
            y0 = y
            if min(np.array([np.linalg.norm(np.array([x, y]) - j) for j in points_passed])) > 1:
                if [x, y] not in points_passed:
                    points_passed = np.append(points_passed, [[x, y]], axis=0)

            else:
                if comm_dist > 100 and dist > 0:
                    flag = flag + 1
                if [x, y] not in points_passed:
                    points_passed = np.append(points_passed, [[x, y]], axis=0)

            if flag > 10:
                break
    ind1 = i - 1

    comm_dist = 0
    flag = 0
    x0 = raw[0, 0]
    y0 = raw[0, 1]
    points_passed = np.array([[x0, y0]])

    for i, (x, y) in enumerate(raw[1:], 1):
        if i % 200 == 1:
            comm_dist = comm_dist + np.sqrt(np.power((x - x0), 2) + np.power((y - y0), 2))
            dist = np.sqrt(np.power((x - x0), 2) + np.power((y - y0), 2))
            x0 = x
            y0 = y
            if min(np.array([np.linalg.norm(np.array([x, y]) - j) for j in points_passed])) > 1:
                if [x, y] not in points_passed:
                    points_passed = np.append(points_passed, [[x, y]], axis=0)

            else:
                if comm_dist > 100 and dist > 0:
                    flag = flag + 1
                if [x, y] not in points_passed:
                    points_passed = np.append(points_passed, [[x, y]], axis=0)
            if flag > 10:
                break
    ind2 = i - 1

    index_end = min(ind1, ind2)

    return index_end


def get_patch_data_pad(have_intersect, start_ind, opt_0, raw_0, img_size, pre_ver_pad, pre_hor_pad, patch_size,
                       init_heading=np.pi):
    ## cut from start index
    opt = np.copy(opt_0[::int(200 / _interpolate)][start_ind:])
    raw = np.copy(raw_0[::int(200 / _interpolate)][start_ind:])

    raw -= raw[0]

    # align starting position and yaw
    angle = np.arctan2(opt[_ali, 1] - opt[0, 1], opt[_ali, 0] - opt[0, 0])

    glob_v = raw[1:] - raw[:-1]
    vel_magnitude = np.linalg.norm(glob_v, axis=1)

    yaw = np.arctan2(glob_v[:, 1], glob_v[:, 0])
    yaw = adjust_angle_arr(yaw)
    yaw_diff = (yaw[1:] - yaw[:-1])

    yaw = np.insert(yaw_diff, 0, [angle])
    yaw = np.cumsum(yaw)

    reconstruct_vel = np.stack([np.cos(yaw), np.sin(yaw)], axis=1) * vel_magnitude[:, None]
    reconstruct_traj = np.zeros([reconstruct_vel.shape[0] + 1, 2])
    reconstruct_traj[1:] = np.cumsum(reconstruct_vel, axis=0)

    raw_align = reconstruct_traj + np.asarray([opt[0]])

    #### getting the (x,y)s to the correct scale so we can shift
    opt = opt * _dpi
    raw_align = raw_align * _dpi

    #### padding the image, shifting the paths
    raw_align[:, 1] = raw_align[:, 1] + pre_ver_pad
    opt[:, 1] = opt[:, 1] + pre_ver_pad
    raw_align[:, 0] = raw_align[:, 0] + pre_hor_pad
    opt[:, 0] = opt[:, 0] + pre_hor_pad

    #### fnding the patch, and where to stop (where we get out of the patch)
    l = len(opt)
    margin = 5
    start_patch_ver = calc_start_index(img_size[0] - 1, opt[0, 1], patch_size)
    start_patch_horiz = calc_start_index(img_size[1] - 1, opt[0, 0], patch_size)
    a = np.argwhere(opt[:, 0] < start_patch_horiz)
    b = np.argwhere(opt[:, 0] > start_patch_horiz + patch_size - 1 - margin)
    c = np.argwhere(opt[:, 1] < start_patch_ver)
    d = np.argwhere(opt[:, 1] > start_patch_ver + patch_size - 1 - margin)
    if np.concatenate([a, b, c, d]).size == 0:
        end_ind = l - 1
    else:
        end_ind = min(np.concatenate([a, b, c, d]))
    a = np.argwhere(raw_align[:, 0] < start_patch_horiz)
    b = np.argwhere(raw_align[:, 0] > start_patch_horiz + patch_size - 1 - margin)
    c = np.argwhere(raw_align[:, 1] < start_patch_ver)
    d = np.argwhere(raw_align[:, 1] > start_patch_ver + patch_size - 1 - margin)
    if np.concatenate([a, b, c, d]).size == 0:
        end_ind = min(end_ind, l - 1)
    else:
        end_ind = min(end_ind, min(np.concatenate([a, b, c, d])))
    end_ind = max(0, end_ind - 1)
    end_ind = np.asarray(end_ind).item()

    #### cutting the rest of the trajectory
    opt_cut = np.copy(opt[:end_ind + 1, :])
    raw_align_cut = np.copy(raw_align[:end_ind + 1, :])

    #####removing intersections
    if have_intersect == False:
        if end_ind == 0:
            end_ind_final = np.copy(end_ind)
        else:
            end_ind_final = remove_intersection(opt_cut, raw_align_cut)
    else:
        end_ind_final = np.copy(end_ind)

    #### start and end of the optimized trajectory as vertical,horizontal locations in image
    start_traj_loc = np.array([int(round(opt[0, 1])), int(round(opt[0, 0]))])
    end_traj_loc = np.array([int(round(opt[end_ind_final, 1])), int(round(opt[end_ind_final, 0]))])

    #### cutting the rest of the trajectory, (/20 for correct dpi in saving image)
    opt_final = np.copy(opt_cut[:end_ind_final + 1, :]) / _dpi
    raw_align_final = np.copy(raw_align_cut[:end_ind_final + 1, :]) / _dpi

    ###  shifting the locations to match the patch image
    start_traj_loc = start_traj_loc - np.array([start_patch_ver, start_patch_horiz])
    end_traj_loc = end_traj_loc - np.array([start_patch_ver, start_patch_horiz])

    return opt_final, raw_align_final, start_traj_loc, end_traj_loc, start_patch_ver, start_patch_horiz


def shift_traj(traj, raw, s_v, s_h, p_size):
    if args.fix_style == 'avg':
        ### aligning the trajectory by the average of start and end points

        end_flow_horiz = traj[-1, 0] - raw[-1, 0]
        end_flow_ver = traj[-1, 1] - raw[-1, 1]
        glob_horiz_shift = end_flow_horiz / 2
        glob_ver_shift = end_flow_ver / 2

        raw_shifted = np.copy(raw)
        raw_shifted[:, 0] = np.copy(raw_shifted[:, 0]) + glob_horiz_shift
        raw_shifted[:, 1] = np.copy(raw_shifted[:, 1]) + glob_ver_shift

    elif args.fix_style == 'start':
        # trajectory is already aligned by its start point
        raw_shifted = np.copy(raw)

    if args.both_end_shift == True:
        # shifting the positions of both ends and geo-localizing the trajectory with new positions
        start_old = raw_shifted[0]
        end_old = raw_shifted[-1]
        norm_old = np.linalg.norm(end_old - start_old)
        angle_old = np.arctan2(end_old[1] - start_old[1], end_old[0] - start_old[0])

        r = args.mean_shift_radius_in_pixels / (np.sqrt(2) * (_dpi * 2))
        start_shift = r * np.random.randn(2) + 0
        end_shift = r * np.random.randn(2) + 0
        start_new = raw_shifted[0] + start_shift
        end_new = raw_shifted[-1] + end_shift

        start_new[0] = min(start_new[0], p_size + s_h)
        start_new[0] = max(start_new[0], 0 + s_h)
        start_new[1] = min(start_new[1], p_size + s_v)
        start_new[1] = max(start_new[1], 0 + s_v)
        end_new[0] = min(end_new[0], p_size + s_h)
        end_new[0] = max(end_new[0], 0 + s_h)
        end_new[1] = min(end_new[1], p_size + s_v)
        end_new[1] = max(end_new[1], 0 + s_v)

        norm_new = np.linalg.norm(end_new - start_new)
        angle_new = np.arctan2(end_new[1] - start_new[1], end_new[0] - start_new[0])

        raw_shifted -= start_old
        if norm_old != 0:
            raw_shifted *= (norm_new / norm_old)
        angle_dif = angle_new - angle_old
        raw_new = np.copy(raw_shifted)
        raw_new[:, 0] = np.copy(raw_shifted[:, 0]) * np.cos(angle_dif) - np.copy(raw_shifted[:, 1]) * np.sin(angle_dif)
        raw_new[:, 1] = np.copy(raw_shifted[:, 0]) * np.sin(angle_dif) + np.copy(raw_shifted[:, 1]) * np.cos(angle_dif)

        raw_new += start_new
    else:
        raw_new = np.copy(raw_shifted)

    return raw_new


def save_all_figs(raw, traj, flow, map, outdir, dataname, index):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(raw)

    a = np.transpose(np.nonzero(flow[:, :, 2]))  # indices of occupancy mask
    # print(a.shape)
    # print(len(traj_patch-raw_patch_shifted))
    for i, x in enumerate(a):
        if i >= 0:
            plt.plot([x[1], x[1] + flow_patch[x[0], x[1], 1]], [x[0], x[0] + flow_patch[x[0], x[1], 0]], color='black',
                     linewidth=1)
    plt.savefig(osp.join(outdir, dataname, "_(%d)" % index + "_rawflow.png"))
    # plt.show()
    plt.close(fig)

    ### trajectories on maps plots

    map = np.copy(map)[start_v:start_v + out_patch_size, start_h:start_h + out_patch_size]
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(map)
    plt.imshow(traj)
    plt.savefig(osp.join(args.out_dir, data, "_(%d)" % j + "_(%d,%d)" % (start_v, start_h) + "_opt.png"))
    # plt.show()
    plt.close(fig)

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(map)
    plt.imshow(raw)
    plt.savefig(osp.join(args.out_dir, data, "_(%d)" % j + "_(%d,%d)" % (start_v, start_h) + "_raw.png"))
    # plt.show()
    plt.close(fig)


if __name__ == '__main__':

    np.random.seed(1)

    out_patch_size = 250

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_all_figs', type=bool, default=True, help='True for saving extra figures for inspection')

    parser.add_argument('--data_dir', type=str, help='Path to synthetic data directory for specific mall floor')
    parser.add_argument('--datalist_file', type=str, help='Path to .txt data list for specific mall floor')
    parser.add_argument('--floorplans_dir', type=str,
                        help='Path to resized floorplan directory with dpi compatible with network input')
    parser.add_argument('--out_dir', type=str, help='directory of the final dataset')
    parser.add_argument('--scale_action', type=str, default='raw_scale', choices=['fix_scale', 'raw_scale'],
                        help='raw_scale for network input with real, errored scales')
    parser.add_argument('--fix_style', type=str, default='start', choices=['start', 'avg'],
                        help='align the trajectory by startpoint or average of start&end points')
    parser.add_argument('--both_end_shift', type=bool, default=True,
                        help='shift both ends of trajectory to simulate flp error')
    parser.add_argument('--mean_shift_radius_in_pixels', type=float, default=45, help='amount of shift')
    parser.add_argument('--has_intersections', type=bool, default=True,
                        help='allow the trajectory to have intersections')
    parser.add_argument('--which_mall', type=str, default='metrotown_0g', choices=['metrotown_0g', 'metrotown_01'])

    args = parser.parse_args()

    if not osp.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if not osp.exists(osp.join(args.out_dir, "flows")):
        os.mkdir(osp.join(args.out_dir, "flows"))

    with open(osp.join(args.floorplans_dir, "map_info.json"), 'r') as f:
        map_config = json.load(f)

    for key in map_config:
        if key == args.which_mall:
            print(key)
            map_config[key]['image'] = plt.imread(osp.join(args.floorplans_dir, map_config[key]['map']))
            if map_config[key]["image"].shape[0] < out_patch_size:
                map_config[key]["ver_pad"], image_pad = pad_function(map_config[key]["image"], out_patch_size, "ver")
            else:
                image_pad = map_config[key]["image"]
                map_config[key]["ver_pad"] = 0

            if image_pad.shape[1] < out_patch_size:
                map_config[key]["hor_pad"], map_config[key]["padded"] = pad_function(image_pad, out_patch_size, "hor")
            else:
                map_config[key]["padded"] = image_pad
                map_config[key]["hor_pad"] = 0

    with open(args.datalist_file, 'r') as f:
        data_list = [s.strip().split()[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']

    total_number_trajectories = len(data_list)
    k = 1
    map_name = args.which_mall
    for data in data_list:
        if not osp.exists(osp.join(args.data_dir, data + '_modified.txt')):
            print("a data not found!")
            print(osp.join(args.data_dir, data + '_modified.txt'))
            continue
        if not osp.exists(osp.join(args.out_dir, data)):
            os.mkdir(osp.join(args.out_dir, data))

        print(k)
        print(data)
        traj, raw_old, _ = open_data(osp.join(args.data_dir, data + '_gt.txt'),
                                     osp.join(args.data_dir, data + '_modified.txt'))

        if args.scale_action == 'fix_scale':
            raise ValueError("we cannot fix scale for synthetic data!")
        elif args.scale_action == 'raw_scale':
            raw = np.copy(raw_old)
        else:
            raise ValueError('you forgot to determine scale action')

        l = len(traj)

        if k == 1:
            file_list = open(osp.join(args.out_dir, "train_list.txt"), "w+")
        if k == 9 * total_number_trajectories // 10:
            file_list.close()
            file_list = open(osp.join(args.out_dir, "valid_list.txt"), "w+")

        stop_generating = float('infinity')

        for j in range(1, patch_per_traj + 1):

            t0 = np.random.randint(0, int(0.85 * l))
            if k < stop_generating:

                traj_patch, raw_patch, s, e, start_v, start_h = get_patch_data_pad(args.has_intersections, t0, traj,
                                                                                   raw,
                                                                                   map_config[map_name]["padded"].shape,
                                                                                   map_config[map_name]["ver_pad"],
                                                                                   map_config[map_name]["hor_pad"],
                                                                                   out_patch_size)

                raw_patch_shifted = shift_traj(traj_patch, raw_patch, np.copy(start_v) / _dpi, np.copy(start_h) / _dpi,
                                               out_patch_size // _dpi)
                if len(raw_patch_shifted) == 1:
                    print('one data empty and ignored')
                    continue

                traj_patch_image = save_plot_image(start_v, start_h, out_patch_size,
                                                   map_config[map_name]["padded"].shape, traj_patch,
                                                   osp.join(args.out_dir, data) + "_(%d)" % j + "_(%d,%d)" % (
                                                   start_v, start_h) + "_opt.png", show_fig=False, save_fig=True,
                                                   give_array=True, dpi=_dpi, res=200)
                raw_patch_image = save_plot_image(start_v, start_h, out_patch_size,
                                                  map_config[map_name]["padded"].shape, raw_patch_shifted,
                                                  osp.join(args.out_dir, data) + "_(%d)" % j + "_(%d,%d)" % (
                                                  start_v, start_h) + "_raw.png", show_fig=False, save_fig=True,
                                                  give_array=True, dpi=_dpi, res=200)

                flow_patch = save_mask_image(start_v, start_h, out_patch_size, map_config[map_name]["padded"].shape,
                                             raw_patch_shifted, traj_patch,
                                             osp.join(args.out_dir, data) + "_raw_flow.npy", give_array=True,
                                             save_array=False)
                np.save(
                    osp.join(args.out_dir, "flows", data + "_(%d)" % j + "_(%d,%d)" % (start_v, start_h) + "_flow.npy"),
                    flow_patch)

                #####write text file
                file_list.write(data + "_(%d)" % j + "_(%d,%d)" % (start_v, start_h) + "\t" + map_name + "\t" + str(
                    start_v) + "\t" + str(start_h) + "\n")

                if args.save_all_figs:
                    save_all_figs(raw_patch_image, traj_patch_image, flow_patch, map_config[map_name]["padded"],
                                  args.out_dir, data, j)

        k = k + 1
    file_list.close()

    # merging of two files
    data = data2 = ""

    # Reading data from file1
    with open(osp.join(args.out_dir, "train_list.txt")) as fp:
        data = fp.read()

    # Reading data from file2
    with open(osp.join(args.out_dir, "valid_list.txt")) as fp:
        data2 = fp.read()

    # Merging 2 files To add the data of file2 from next line
    data += data2

    with open(osp.join(args.out_dir, "all_list.txt"), 'w') as fp:
        fp.write(data)
