import os.path as osp
import sys

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
from optim.optimizer import *

font = {'family': 'DejaVu Sans',
        'size': 4}

matplotlib.rc('font', **font)

if __name__ == '__main__':
    with open(osp.join('../data_paths.json'), 'r') as f:
        default_config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="[Optional] file to load configuration from")
    parser.add_argument('--data_path', type=str, default=None, help="Path to folder containing hdf5 and json files")
    parser.add_argument('--data_list', type=str, default=None)
    parser.add_argument('--data_dir', type=str, help='Path to parent of folders containing hdf5 files')
    parser.add_argument('--result_dir', type=str, help='Path to folder containing trajectory results from cnn')

    parser.add_argument('--traj_freq', type=int, default=50, help='Frequency of trajectory data')
    parser.add_argument('--result_freq', type=int, default=50, help='Frequency of cnn trajectory results')

    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--prefix', type=str)

    # specify a single map_file with data_path or specify map_dir
    parser.add_argument('--map_path', type=str, default=None, help="Path to map image")
    parser.add_argument('--map_dpi', type=int)
    parser.add_argument('--map_dir', type=str, default=default_config.get('floorplan_dir', None))

    parser.add_argument('--starting_variables', type=str,
                        help='Path to file containing optimization varibles (ensure same number of parameter)')
    # position priors from cnn_result
    parser.add_argument('--position_radius', type=float, default=2, help="Accuracy radius for position in meters")
    parser.add_argument('--position_sample', type=int, default=4, help="Select sparse position points every x seconds")

    # optimization params
    parser.add_argument('--scale', type=str, choices=Variable.get_var_states(False), help='Status of scale param')
    parser.add_argument('--scale_interval', type=int,
                        help='Variable interval for piecewise linear function of scale (0 '
                             'for constant)')
    parser.add_argument('--bias', type=str, choices=Variable.get_var_states(False), help='Status of bias param')
    parser.add_argument('--bias_interval', type=int, help='Variable interval for piecewise linear function of bias')

    # optimization_functions
    parser.add_argument('--pos_prior_weight', type=float, help="Weight for manual position prior (-1 for no prior)")
    parser.add_argument('--pos_prior_loss', type=float, help="Norm loss function for pos prior (-1 for default)")
    parser.add_argument('--interpolate_kind', type=str, default="quadratic", choices=["linear", "quadratic", "cubic"],
                        help="Interpolation type")

    parser.add_argument('--scale_reg_weight', type=float, help="Weight for scale regularization (-1 for no prior)")
    parser.add_argument('--bias_reg_weight', type=float, help="Weight for bias regularization (-1 for no prior)")

    # optimization parameters
    parser.add_argument('--n_iterations', type=int, default=50)
    parser.add_argument('--loop', action='store_true', help='When set, iterate until the energy is stable (with cmd)')
    parser.add_argument('--verbose', action='store_true', help='When set progress of optimization is printed')
    parser.add_argument('--no_gui', action='store_true', help='If true, run optimization without gui')

    args = parser.parse_args()
    args, _ = load_config('default_config.json', args)
    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})

    # data_list (data name, map name, csv name)
    data_list, map_dir, root_dir, single = [], None, None, False
    if args.data_path is not None:
        single = True
        if args.data_path[-1] == '/':
            args.data_path = args.data_path[:-1]
        data_list.append([osp.split(args.data_path)[1], osp.split(args.map_path)[1]])

        root_dir = osp.split(args.data_path)[0]
        if len(root_dir) == 0: root_dir = args.data_dir
        map_dir = osp.split(args.map_path)[0]
        if len(map_dir) == 0: map_dir = args.map_dir
    elif args.data_list is not None:
        # data list should have data path, map name
        # cannot work with starting params and position_priors from file
        root_dir = args.data_dir
        with open(args.data_list) as f:
            for s in f.readlines():
                if len(s) > 0 and s[0] != '#':
                    s = s.strip().split()
                    data_list.append([s[0].strip(" \'\""), s[1].strip(" \'\"")])
        map_dir = args.map_dir
    else:
        print('Error: data_list or data_path must be specified')
        exit(1)

    dpi = args.map_dpi
    if not osp.exists(args.out_dir):
        os.makedirs(args.out_dir)
    out_dir_main = args.out_dir

    for data_info in data_list:
        print('-' * 10, '\nProcessing ', data_info)
        map_image = plt.imread(osp.join(map_dir, data_info[1]))
        if len(map_image == 3):
            map_image = map_image[:, :, :3]
        im_shape = np.asarray(map_image.shape[:2])
        print('Image size: {}, dpi: {} pixels per meter '.format(im_shape, dpi))

        # get gt trajectory
        ts, m_vel, a_vel, traj = data_loader.load_trajectory_as_polar_velocity(osp.join(root_dir, data_info[0]),
                                                                               type='ronin',
                                                                               interp_freq=args.traj_freq,
                                                                               visualize=False)

        # prepare output directory
        dir_name = '{}_{}'.format(data_info[0], args.prefix) if args.prefix is not None else data_info[0]
        args.out_dir = osp.join(out_dir_main, dir_name)
        if not osp.exists(args.out_dir):
            os.makedirs(args.out_dir)
        write_config(args)

        # load position priors (or fla priors) and bias parameters from file
        map_aligned_points, starting_params = None, {}
        gt_points = None

        with h5py.File(osp.join(root_dir, data_info[0], 'data.hdf5'), 'r') as f:
            if 'filtered/locations' in f.keys():
                gt_points = np.copy(f['filtered/locations'])
        if gt_points is not None and len(gt_points) > 0:
            gt_points = match_timestamps(ts, gt_points[:, 0], gt_points[:, 1:3] / dpi, allow_err=5)
        else:
            gt_points = None

        # Load position priors from cnn result
        cnn_traj = np.loadtxt(osp.join(args.result_dir, '{}.txt'.format(data_info[0])))  # format: position x, y
        cnn_timestamp, cnn_traj = cnn_traj[:, 0], cnn_traj[:, 1:]
        if args.position_sample > 1:
            map_aligned_points = find_sparse(args.position_sample, cnn_timestamp, cnn_traj)
        else:
            map_aligned_points = np.concatenate([cnn_timestamp[:, None], cnn_traj], axis=1)
        map_aligned_points = np.concatenate(
            [map_aligned_points, np.full([len(map_aligned_points), 1], args.position_radius)],
            axis=1)  # add error radius
        map_aligned_points = match_timestamps(ts, map_aligned_points[:, 0], map_aligned_points[:, 1:], allow_err=1)
        print('Loaded position priors from cnn result. {} selected'.format(len(map_aligned_points)))
        if gt_points is not None:
            create_plot(map_image, dpi, cnn_traj, save_files=True,
                        filename=osp.join(args.out_dir, 'cnn_trajectory_with_gt.png'),
                        pos_constraints=gt_points, calc_errors=True)

        if single and args.starting_variables is not None:
            with open(osp.join(args.starting_variables), 'r') as f:
                starting_params = json.load(f)
            if 'angle' in starting_params:
                # in polar_velocity model, bias is the correction to velocity direction
                starting_params['bias'] = starting_params['angle'].copy()

        plt.close('all')
        if args.no_gui:
            optimizer = Cmd(map_image, dpi, traj, args, ts, m_vel, a_vel, map_aligned_points, starting_params,
                            loop=args.loop)
        else:
            optimizer = Gui(map_image, dpi, traj, args, ts, m_vel, a_vel, map_aligned_points, starting_params)
            optimizer.run()

        if optimizer.status == Status.OPTIMIZING:
            print('saving results')
            np.savetxt(osp.join(args.out_dir, 'final_trajectory.txt'),  # format: timestamp, pos_x, pos_y
                       np.concatenate([ts[:, None], optimizer.refined_traj], axis=1))
            np.savetxt(osp.join(args.out_dir, 'final_constraints.txt'),
                       # format: point_idx, pos_x, pos_y, radius, timestamp
                       np.concatenate(
                           [optimizer.map_aligned_points, ts[optimizer.map_aligned_points[:, 0].astype(int)][:, None]],
                           axis=1))
            create_plot(map_image, dpi, optimizer.refined_traj, save_files=True,
                        filename=osp.join(args.out_dir, 'final_trajectory.png'))
            if gt_points is not None:
                create_plot(map_image, dpi, optimizer.refined_traj, save_files=True,
                            filename=osp.join(args.out_dir, 'final_trajectory_with_gt.png'),
                            pos_constraints=gt_points, calc_errors=True)
            with open(osp.join(args.out_dir, 'final_optimization_variable.txt'), 'w') as f:
                var = create_variable_dict(optimizer.compound_functor.get_variables(optimizer.solution),
                                           correct_angle=True)
                json.dump(var, f)
