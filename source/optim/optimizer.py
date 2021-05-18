import argparse
import copy
import os
import os.path as osp
import sys
import time
import tkinter as tk
from collections import OrderedDict
from enum import Enum

import h5py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import least_squares

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
import util.data_loader as data_loader
from gui.draggable_scatter import ScatterInteractor
from optim.residual_functions import *
from optim.optimization_config import *
from util.map_util import match_timestamps, find_sparse, process_flp_data
from util.other import *
from util.math_utils import *
from util.evaluation import create_plot

font = {'family': 'DejaVu Sans',
        'size': 4}

matplotlib.rc('font', **font)

# In our final model we estimate corrections to velocity magnitude (scale) and velocity direction (bias), along with
# starting position (pos).
# The model can also be modified to use correct velocity magnitude (scale) and angular rate (bias), with starting
# position (pos) and starting velocity
# direction (yaw).
# Noise is a sparse variable that can model changes in orientation due to sudden motion, but not used in our final
# model.
_my_variables = ['pos', 'yaw', 'scale', 'bias', 'noise']


def write_config(args, out_dir=None):
    if out_dir is None:
        out_dir = args.out_dir
    with open(osp.join(out_dir, 'config.json'), 'w') as f:
        values = vars(args)
        values['file'] = "gui_polar"
        json.dump(values, f, sort_keys=True)


def get_angle_between_vectors(V_s, V_t):
    angle = np.arctan2(V_t[1], V_t[0]) - np.arctan2(V_s[1], V_s[0])
    return angle


def get_angle_to_vector(V_s, V_t):
    angle = np.arctan2(V_t[1], V_t[0]) - V_s
    angle = np.arctan2(np.sin(angle), np.cos(angle))
    print(V_s, V_t, angle)
    return angle


def create_variable_dict(values, correct_angle=True):
    # for polar_velocity correction store pos, angle and scale
    # for velocity magnitude, angular rate model store pos, yaw, bias, scale
    result = {}
    for k, v in zip(_my_variables, values):
        if v is not None:
            result[k] = list(v)

    if correct_angle:
        result['angle'] = result['bias'].copy()
        result['bias'] = None
    else:
        result['angle'] = None
    return result


class Status(Enum):
    INIT = 0
    OPTIMIZING = 1
    START = 2


class Gui():

    def __init__(self, map_image, dpi, origin_traj, args, ts, vel_norm, vel_angle, map_aligned_points=None,
                 starting_params={}):
        self.map_image = map_image
        self.dpi = dpi
        self.args = args
        self.origin_args = copy.deepcopy(args)
        self.fig = None
        self.inverted_map_left, self.inverted_map_right = False, False

        self.origin_traj = origin_traj
        self.refined_traj = origin_traj
        self.vel_norm = vel_norm
        self.vel_angle = vel_angle
        self.ts = ts

        self.status = Status.INIT
        self.origin_map_aligned_points = map_aligned_points
        self.origin_params = starting_params
        self.image_count = np.zeros(6)

        self._init_view()
        self._reset()

    def _init_view(self):
        # create gui
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.bind_all('<KeyPress>', self.key_press)

        # gui parameters
        a = Counter()
        tk.Label(self.root, text="PARAMETERS", font=10).grid(row=a.get_next(), column=1)
        tk.Label(self.root, text="state", font=5).grid(row=a.val, column=1)
        tk.Label(self.root, text="variable", font=5).grid(row=a.val, column=0)
        tk.Label(self.root, text="reg_weight", font=5).grid(row=a.get_next(), column=2)

        self.start_pos_on_var = tk.StringVar()
        tk.OptionMenu(self.root, self.start_pos_on_var, *Variable.get_var_states(True)).grid(row=a.val, column=1,
                                                                                             sticky=tk.W)
        tk.Label(self.root, text="start_pos", font=5).grid(row=a.get_next(), column=0)

        self.scale_on_var = tk.StringVar()
        tk.OptionMenu(self.root, self.scale_on_var, *Variable.get_var_states(False)).grid(row=a.val, column=1,
                                                                                          sticky=tk.W)
        tk.Label(self.root, text="scale", font=5).grid(row=a.val, column=0)
        self.scale_reg_var = tk.StringVar()
        tk.Entry(self.root, textvariable=self.scale_reg_var, width=6).grid(row=a.get_next(), column=2)

        self.bias_on_var = tk.StringVar()
        tk.OptionMenu(self.root, self.bias_on_var, *Variable.get_var_states(False)).grid(row=a.val, column=1,
                                                                                         sticky=tk.W)
        tk.Label(self.root, text="angle", font=5).grid(row=a.val, column=0)
        self.bias_reg_var = tk.StringVar()
        tk.Entry(self.root, textvariable=self.bias_reg_var, width=6).grid(row=a.get_next(), column=2)

        # gui priors
        tk.Label(self.root, text="PRIORS", font=10).grid(row=a.get_next(), column=1)

        self.pos_on_var = tk.IntVar()
        tk.Checkbutton(self.root, text="pos", variable=self.pos_on_var).grid(row=a.val, column=0, sticky=tk.W)
        tk.Label(self.root, text="pos prior weight", font=5).grid(row=a.val, column=1)
        self.pos_weight_var = tk.StringVar()
        tk.Entry(self.root, textvariable=self.pos_weight_var, width=6).grid(row=a.get_next(2), column=2)

        # control bottons
        tk.Label(self.root, text="OPTIONS", font=10).grid(row=a.get_next(), column=1)
        optimize_b = tk.Button(self.root,
                               text='set solution',
                               width=15, height=2,
                               command=self._get_solution_from_gui)
        optimize_b.grid(row=a.val, column=1)

        tk.Label(self.root, text="iterations", font=5).grid(row=a.get_next(), column=2)
        self.n_iteration_var = tk.StringVar()
        tk.Entry(self.root, textvariable=self.n_iteration_var, width=6).grid(row=a.val, column=2)

        optimize_b = tk.Button(self.root,
                               text='optimize',
                               width=15, height=2,
                               command=self._optimize)
        optimize_b.grid(row=a.get_next(), column=1)

        update_b = tk.Button(self.root,
                             text='reset',
                             width=15, height=2,
                             command=self._reset)
        update_b.grid(row=a.get_next(), column=1)

        quit_b = tk.Button(self.root,
                           text='quit',
                           width=15, height=2,
                           command=self._quit)
        quit_b.grid(row=a.get_next(), column=1)

        optimize_b = tk.Button(self.root,
                               text='save',
                               width=15, height=2,
                               command=self._save_values)
        optimize_b.grid(row=a.val, column=1)
        self.save_choices = {0: '<select option>', 1: '[Img] Traj. on map', 2: '[Img] Constraints',
                             3: '[Img] Traj. only', 4: '[txt] Variables',
                             5: '[txt] Constraints', 6: '[Img] Variables'}
        self.n_save_option = tk.StringVar()
        self.n_save_option.set(self.save_choices[0])
        tk.OptionMenu(self.root, self.n_save_option, *self.save_choices.values()).grid(row=a.get_next(), column=2,
                                                                                       sticky='ew')

        # map params
        self.map_choices = {0: 'left', 1: 'right', 2: 'both'}
        self.n_map_option = tk.StringVar()
        self.n_map_option.set(self.map_choices[0])
        tk.OptionMenu(self.root, self.n_map_option, *self.map_choices.values()).grid(row=a.val, column=2, sticky='ew')
        tk.Button(self.root,
                  text='switch map',
                  width=15, height=2,
                  command=self._switch_map).grid(row=a.get_next(2), column=1)

        # map params
        tk.Label(self.root, text="Plot points", font=10).grid(row=a.val, column=0)
        refine_b = tk.Button(self.root,
                             text='refine',
                             width=15, height=1,
                             command=self._refine_plot)
        refine_b.grid(row=a.val, column=1)
        refine_b = tk.Button(self.root,
                             text='del. active',
                             width=15, height=1,
                             command=self._remove_points)
        refine_b.grid(row=a.get_next(), column=2)

        self.map_start_var = tk.StringVar()
        tk.Entry(self.root, textvariable=self.map_start_var, width=8).grid(row=a.val, column=0)
        self.map_end_var = tk.StringVar()
        tk.Entry(self.root, textvariable=self.map_end_var, width=8).grid(row=a.val, column=1)
        self.map_step_var = tk.StringVar()
        tk.Entry(self.root, textvariable=self.map_step_var, width=8).grid(row=a.get_next(), column=2)

        tk.Label(self.root, text="start", font=1).grid(row=a.val, column=0)
        tk.Label(self.root, text="end", font=1).grid(row=a.val, column=1)
        tk.Label(self.root, text="step", font=1).grid(row=a.get_next(2), column=2)

        # gui results
        tk.Label(self.root, text="RESULTS", font=10).grid(row=a.get_next(), column=1)

        tk.Label(self.root, text="init pos", font=5).grid(row=a.val, column=0)
        self.pos_result_var = tk.StringVar()
        tk.Entry(self.root, textvariable=self.pos_result_var).grid(row=a.get_next(), column=1, columnspan=2,
                                                                   sticky='ew')

        tk.Label(self.root, text="scale", font=5).grid(row=a.val, column=0)
        self.scale_result_var = tk.StringVar()
        self.scale_entry = tk.Entry(self.root, textvariable=self.scale_result_var)
        self.scale_entry.grid(row=a.get_next(), column=1, columnspan=2, sticky='ew')
        self.scaleEntryScroll = tk.Scrollbar(self.root, orient=tk.HORIZONTAL,
                                             command=self._scale_scroll_handler)
        self.scaleEntryScroll.grid(row=a.get_next(), column=1, columnspan=2, sticky=tk.E + tk.W)
        self.scale_entry['xscrollcommand'] = self.scaleEntryScroll.set

        tk.Label(self.root, text="angle", font=5).grid(row=a.val, column=0)
        self.bias_result_var = tk.StringVar()
        self.bias_entry = tk.Entry(self.root, textvariable=self.bias_result_var)
        self.bias_entry.grid(row=a.get_next(), column=1, columnspan=2, sticky='ew')
        self.biasEntryScroll = tk.Scrollbar(self.root, orient=tk.HORIZONTAL,
                                            command=self._bias_scroll_handler)
        self.biasEntryScroll.grid(row=a.get_next(4), column=1, columnspan=2, sticky=tk.E + tk.W)
        self.bias_entry['xscrollcommand'] = self.biasEntryScroll.set

        # energy results
        tk.Label(self.root, text="ENERGY", font=10).grid(row=a.get_next(), column=1)
        tk.Label(self.root, text="previous", font=5).grid(row=a.val, column=1)
        tk.Label(self.root, text="current", font=5).grid(row=a.get_next(), column=2)
        self.prev_pos_energy = tk.StringVar()
        tk.Label(self.root, text="pos. constrains", font=5).grid(row=a.val, column=0)
        tk.Entry(self.root, textvariable=self.prev_pos_energy).grid(row=a.val, column=1, sticky='ew')
        self.current_pos_energy = tk.StringVar()
        tk.Entry(self.root, textvariable=self.current_pos_energy).grid(row=a.get_next(), column=2, sticky='ew')

    def key_press(self, event):
        kp = event.char
        if type(event.widget).__name__ == 'Canvas':
            self.scatter_interact.key_press_callback_from_tk(kp)

    def _save_values(self):
        n_plot = self.n_save_option.get()
        if n_plot == self.save_choices[1]:
            extent = self.ax[0, 0].get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            self.fig.savefig(osp.join(self.args.out_dir, 'trajectory_plot_{}.png'.format(self.image_count[0])),
                             bbox_inches=extent.expanded(1.1, 1.2))
            self.image_count[0] += 1

        elif n_plot == self.save_choices[2]:
            extent = self.ax[0, 1].get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            self.fig.savefig(osp.join(self.args.out_dir, 'constraints_{}.png'.format(self.image_count[1])),
                             bbox_inches=extent.expanded(
                                 1.1, 1.2))
            self.image_count[1] += 1

        elif n_plot == self.save_choices[3]:
            fig = plt.figure(figsize=(8, 12))
            plt.scatter(self.scatter_interact.points[
                        self.scatter_interact.start:self.scatter_interact.end:self.scatter_interact.step, 0],
                        self.scatter_interact.points[
                        self.scatter_interact.start:self.scatter_interact.end:self.scatter_interact.step, 1],
                        color=self.scatter_interact.colors[
                              self.scatter_interact.start:self.scatter_interact.end:self.scatter_interact.step],
                        picker=5, s=70)
            plt.plot(self.scatter_interact.points[:, 0], self.scatter_interact.points[:, 1], color='y')
            plt.ylim([self.map_image.shape[0] / self.dpi - .5, -.5])
            plt.axis('equal')
            fig.savefig(osp.join(self.args.out_dir, 'plot_only_{}.png'.format(self.image_count[2])), transparent=True)
            self.image_count[2] += 1

        elif n_plot == self.save_choices[4] and self.status == Status.OPTIMIZING:
            with open(osp.join(self.args.out_dir, 'optimization_variables_{}.txt'.format(self.image_count[3])),
                      'w') as f:
                json.dump(create_variable_dict(self.compound_functor.get_variables(self.solution),
                                               correct_angle=True), f)
            self.image_count[3] += 1

        elif n_plot == self.save_choices[5]:
            # format: point_idx, pos_x, pos_y, radius, timestamp
            np.savetxt(osp.join(self.args.out_dir, 'constraints_{}.txt'.format(self.image_count[4])),
                       np.concatenate(
                           [self.map_aligned_points, self.ts[self.map_aligned_points[:, 0].astype(int)][:, None]],
                           axis=1))
            self.image_count[4] += 1

        elif n_plot == self.save_choices[6]:
            fig, ax = plt.subplots(2)
            ax[0].plot(self.variables['scale'].last_value, color='b')
            ax[0].set_title('scale_correction')
            ax[1].plot(self.variables['bias'].last_value, color='r')
            ax[1].set_title('angle_correction')
            fig.savefig(osp.join(self.args.out_dir, 'scale_and_bias_{}.png'.format(self.image_count[5])))
            self.image_count[5] += 1

    def _switch_map(self):
        self._update_pos_constraints()
        n_map = self.n_map_option.get()
        if n_map == self.map_choices[0]:
            self.inverted_map_left = not self.inverted_map_left
        elif n_map == self.map_choices[1]:
            self.inverted_map_right = not self.inverted_map_right
        else:
            self.inverted_map_right = not self.inverted_map_right
            self.inverted_map_left = not self.inverted_map_left
        self._update_plot(self.refined_traj)

    def _bias_scroll_handler(self, *L):
        op, howMany = L[0], L[1]
        if op == 'scroll':
            units = L[2]
            self.bias_entry.xview_scroll(howMany, units)
        elif op == 'moveto':
            self.bias_entry.xview_moveto(howMany)

    def _scale_scroll_handler(self, *L):
        op, howMany = L[0], L[1]
        if op == 'scroll':
            units = L[2]
            self.scale_entry.xview_scroll(howMany, units)
        elif op == 'moveto':
            self.scale_entry.xview_moveto(howMany)

    def _get_solution_from_gui(self):
        self._get_args_from_gui()  # load changed variable options first
        # get solution result in gui
        v_pos = self._string_to_np_array(self.pos_result_var.get())
        v_scale = self._string_to_np_array(self.scale_result_var.get())
        v_bias = self._string_to_np_array(self.bias_result_var.get())

        change_if = ['active', 'last']
        if v_pos is not None and self.variables['pos'].get_current_state() in change_if:
            self.variables['pos'].last_value = v_pos
        if v_scale is not None and self.variables['scale'].get_current_state() in change_if:
            self.variables['scale'].last_value = v_scale
        if v_bias is not None and self.variables['bias'].get_current_state() in change_if:
            v_bias = adjust_angle_arr(v_bias)  # to prevent the differences getting larger
            self.variables['bias'].last_value = v_bias

        self.solution = get_active_params(self.variables)

        v_pos, v_yaw, v_scale, v_bias, v_noise = self.compound_functor.get_variables(self.solution)
        print('\tinit: ', v_pos)
        print('\tscale ', v_scale, '\n')
        print('\tbias ', v_bias, '\n')
        self.refined_traj = self.functions['pos_prior'].func.get_modified_trajectory(v_pos, v_yaw, v_bias, v_noise,
                                                                                     v_scale)
        self._update_traj_plot_only(self.refined_traj)
        self._set_solution_in_gui()

    def _get_args_from_gui(self):
        # get configuration from gui
        self.variables['pos'].set_state(self.start_pos_on_var.get())
        self.variables['bias'].set_state(self.bias_on_var.get())
        self.variables['scale'].set_state(self.scale_on_var.get())

        self.functions['bias_regularize'].weight = float(self.bias_reg_var.get())
        self.functions['scale_regularize'].weight = float(self.scale_reg_var.get())

        self.functions['pos_prior'].weight = -1 if not self.pos_on_var.get() else float(self.pos_weight_var.get())

        self.args.n_iterations = int(self.n_iteration_var.get())

    @staticmethod
    def _np_array_to_string(a):
        if a is None:
            return ""
        else:
            return " ".join([str(i) for i in a])

    @staticmethod
    def _string_to_np_array(a):
        if a == "":
            return None
        else:
            return np.asarray([float(i) for i in a.split(" ")])

    def _set_args_in_gui(self):
        # store variable and function config in gui
        self.start_pos_on_var.set(self.variables['pos'].get_current_state())
        self.bias_on_var.set(self.variables['bias'].get_current_state())
        self.scale_on_var.set(self.variables['scale'].get_current_state())

        self.bias_reg_var.set(str(self.functions['bias_regularize'].weight))
        self.scale_reg_var.set(str(self.functions['scale_regularize'].weight))

        # set priors
        self.pos_on_var.set(int(self.functions['pos_prior'].is_active()))

        self.pos_weight_var.set(str(self.functions['pos_prior'].weight))

        self.n_iteration_var.set(str(self.args.n_iterations))
        self.map_start_var.set(str(self.start))
        self.map_end_var.set(str(self.end))
        self.map_step_var.set(str(self.step))

    def _set_solution_in_gui(self):
        if self.solution is not None:
            v_pos, v_yaw, v_scale, v_bias, v_noise = self.compound_functor.get_variables(self.solution)
        else:
            v_pos, v_scale = self.variables['pos'].last_value, self.variables['scale'].last_value
            v_bias, v_noise = self.variables['bias'].last_value, self.variables['noise'].last_value
        self.pos_result_var.set(self._np_array_to_string(v_pos))
        self.scale_result_var.set(self._np_array_to_string(v_scale))
        self.bias_result_var.set(self._np_array_to_string(v_bias))

        self.prev_pos_energy.set(self.current_pos_energy.get())
        if self.solution is not None:
            self.current_pos_energy.set(str(self.compound_functor.get_energy(self.solution, 'pos_prior')))
        self.canvas.get_tk_widget().focus_set()

    def _reset(self):
        self.variables, self.functions = OrderedDict(), OrderedDict()
        self.compound_functor = CompoundFunctor(self.variables, self.functions)
        self.map_aligned_points, self.starting_params = self.origin_map_aligned_points, self.origin_params
        self.inverted_map_left, self.inverted_map_right = False, False

        self.lsq = None
        self.scatter_interact = None
        self.solution = None

        self.args = copy.deepcopy(self.origin_args)
        self.status = Status.START
        self.start, self.end, self.step = 0, len(self.origin_traj), 500

        self._init_params()
        self._set_args_in_gui()
        self._update_plot(self.origin_traj)
        self._set_solution_in_gui()

    def _init_params(self):
        # create all variables and functions
        args = self.args
        self.variables['pos'] = Variable(name='pos', length=2, active=True, required=True,
                                         default_value=0, last_value=self.starting_params.get('pos', None))
        self.variables['yaw'] = Variable(name='yaw', length=0, active=False, required=False, default_value=0)  # unused

        scale_params, scale_index = 1, None
        if args.scale_interval > 0:
            scale_params = math.ceil(len(self.vel_norm) / args.scale_interval) + 1
            scale_index = np.arange(0, scale_params) * args.scale_interval
        print('Adding scale per every {} frames, {} params'.format(args.scale_interval / args.traj_freq, scale_params))
        if self.starting_params.get('scale'):
            if len(self.starting_params.get('scale')) < scale_params:
                self.starting_params.get('scale').append(1)
        self.variables['scale'] = Variable(name='scale', length=scale_params,
                                           required=False, default_value=1,
                                           last_value=self.starting_params.get('scale', None))
        self.variables['scale'].set_state(args.scale)
        self.functions['scale_regularize'] = Function(name='scale_regularize', weight=args.scale_reg_weight,
                                                      select_param=True, param='scale')
        self.functions['scale_regularize'].func = RatioPriorFunctor()

        bias_params = math.ceil(len(self.vel_angle) / args.bias_interval) + 1
        bias_index = np.arange(0, bias_params) * args.bias_interval
        print('Adding bias per every {} frames, {} params'.format(args.bias_interval / args.traj_freq, bias_params))
        self.variables['bias'] = Variable(name='bias', length=bias_params, required=False,
                                          default_value=0, bias_index=bias_index,
                                          last_value=self.starting_params.get('bias', None))
        self.variables['bias'].set_state(args.bias)
        self.functions['bias_regularize'] = Function(name='bias_regularize', weight=args.bias_reg_weight,
                                                     select_param=True,
                                                     param='bias')
        self.functions['bias_regularize'].func = DriftPriorFunctor(weight=0.6)

        self.variables['noise'] = Variable(name='noise', length=0, required=False, active=False,
                                           default_value=0)  # unused

        self.functions['pos_prior'] = Function(name='pos_prior', weight=args.pos_prior_weight)
        self.functions['pos_prior'].func = PositionPriorFunctor(self.vel_norm, self.vel_angle, self.ts,
                                                                bias_index=bias_index, scale_index=scale_index,
                                                                noise_index=None,
                                                                correct_angle=True, kind=args.interpolate_kind)
        self.current_pos_energy.set('-')
        self.prev_pos_energy.set('-')

    def _optimize(self):
        self._update_pos_constraints()
        print('Selected constraints: ', len(self.map_aligned_points))
        self.functions['pos_prior'].func.set_pos_constraints(self.map_aligned_points[:, 0],
                                                             self.map_aligned_points[:, 1:3],
                                                             self.map_aligned_points[:, 3])
        self._get_solution_from_gui()

        print('Starting optimization')
        start_t = time.time()
        self.lsq = least_squares(self.compound_functor, self.solution, loss='linear',
                                 max_nfev=self.args.n_iterations, gtol=1e-15, xtol=1e-15, ftol=1e-15,
                                 verbose=2 if self.args.verbose else 1)
        self.solution = self.lsq.x
        end_t = time.time()
        print('Optimization done. [Time: {} s]'.format(end_t - start_t))
        self.canvas.get_tk_widget().focus_set()

        store_as_last_params(self.variables, self.solution)
        v_pos, v_yaw, v_scale, v_bias, v_noise = self.compound_functor.get_variables(self.solution)
        print('\tinit: ', v_pos)
        print('\tscale ', v_scale, '\n')
        print('\tbias ', v_bias, '\n')
        self.refined_traj = self.functions['pos_prior'].func.get_modified_trajectory(v_pos, v_yaw, v_bias, v_noise,
                                                                                     v_scale)
        self.status = Status.OPTIMIZING
        self._set_solution_in_gui()
        self._update_traj_plot_only(self.refined_traj)

    def _update_pos_constraints(self):
        # update position constraints
        self.scatter_interact.update_selected_points()
        if not self.scatter_interact.selected_points:
            print('No points selected')
        else:
            n = len(self.scatter_interact.selected_points)
            result = np.empty([n, 4])
            keys = np.sort(np.asarray(list(self.scatter_interact.selected_points.keys()))).astype(int)
            for i, key in enumerate(keys):
                result[i, 0] = key
                dr = self.scatter_interact.selected_points[key]
                result[i, 1], result[i, 2] = dr.point.center
                result[i, 3] = dr.point.get_radius()
            print("Selected {} points".format(n))
            self.map_aligned_points = result
            self.functions['pos_prior'].func.set_pos_constraints(self.map_aligned_points[:, 0],
                                                                 self.map_aligned_points[:, 1:3],
                                                                 self.map_aligned_points[:, 3])

        # initialize variables using rigid transformation before first iteration
        if self.status == Status.START and not self.starting_params:
            self.variables['pos'].last_value = self.map_aligned_points[0, 1:3]
            self.pos_result_var.set(self._np_array_to_string(self.variables['pos'].last_value))

            # rigid transformation
            print('Starting rigid transformation')
            start_t = time.time()
            self.variables['scale'].set_state('none')
            self.functions['bias_regularize'].weight = 1e08  # large value to enfore same orientation
            solution = get_active_params(self.variables)
            lsq = least_squares(self.compound_functor, solution, loss='linear',
                                max_nfev=20, gtol=1e-15, xtol=1e-15, ftol=1e-15,
                                verbose=2 if self.args.verbose else 1)
            print(solution)
            solution = lsq.x
            end_t = time.time()
            print('Optimization done. [Time: {} s]'.format(end_t - start_t))

            store_as_last_params(self.variables, solution)
            v_pos, v_yaw, v_scale, v_bias, v_noise = self.compound_functor.get_variables(solution)
            print('\tinit: ', v_pos)
            print('\tbias ', v_bias, '\n')
            self.refined_traj = self.functions['pos_prior'].func.get_modified_trajectory(v_pos, v_yaw, v_bias, v_noise,
                                                                                         v_scale)
            # back to normal
            self.variables['scale'].set_state(self.scale_on_var.get())
            self.functions['bias_regularize'].weight = float(self.bias_reg_var.get())

            self.status = Status.INIT
            self._update_traj_plot_only(self.refined_traj)

    def _update_plot(self, traj, step_size=1):
        ax_dim = 3 if self.map_image.shape[0] > self.map_image.shape[1] else 0
        if self.fig is not None:
            self.fig.clf()
            # self.fig.set_size_inches(10, 10)
            self.fig.add_gridspec(3, 2)
            if ax_dim == 3:
                self.ax = self.fig.subplots(3, 2, gridspec_kw={'height_ratios': [25, 1, 1]})
            else:
                self.ax = self.fig.subplots(6, 1, gridspec_kw={'height_ratios': [25, 1, 1, 25, 1, 1]}, squeeze=False)
                self.ax = self.ax.reshape([2, 3]).transpose()
        else:
            if ax_dim == 3:
                self.fig, self.ax = plt.subplots(3, 2, figsize=(15, 12), gridspec_kw={'height_ratios': [25, 1, 1]})
            else:
                self.fig, self.ax = plt.subplots(6, 1, figsize=(15, 12),
                                                 gridspec_kw={'height_ratios': [25, 1, 1, 25, 1, 1]},
                                                 squeeze=False)
                self.ax = self.ax.reshape([2, 3]).transpose()
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)  # a tk.DrawingArea.
            self.canvas.get_tk_widget().grid(row=0, column=15, rowspan=70)

        self.ax[0, 0].imshow(1 - self.map_image if self.inverted_map_left else self.map_image,
                             extent=[0, self.map_image.shape[1] / self.dpi, self.map_image.shape[0] / self.dpi, 0])
        self.ax[0, 1].imshow(1 - self.map_image if self.inverted_map_right else self.map_image,
                             extent=[0, self.map_image.shape[1] / self.dpi, self.map_image.shape[0] / self.dpi, 0])
        self.scatter_interact = ScatterInteractor(self.ax, traj, step_size, self.map_aligned_points, self.canvas)
        self.ax[0, 1].set_title('Modified key points')

        self.ax[0, 1].set_xlabel('Drag and drop to move circles. Press e /t to delete/ reset selected circle')
        self.ax[2, 0].set_title('Radius of selected point (click button below to update)')
        self.ax[2, 1].set_title('Radius of selected circle (click button below to update)')

        self.ax[0, 1].set_xlim(self.ax[0, 0].get_xlim())
        self.ax[0, 1].set_ylim(self.ax[0, 0].get_ylim())
        self.canvas.draw()
        self._refine_plot()

    def _update_traj_plot_only(self, traj, step_size=1):
        # coord = np.flip(coord, axis=1)
        ax_dim = 3 if self.map_image.shape[0] > self.map_image.shape[1] else 0

        self.ax[0, 0].clear()
        self.ax[0, 0].imshow(1 - self.map_image if self.inverted_map_left else self.map_image,
                             extent=[-.5, self.map_image.shape[1] / self.dpi - .5,
                                     self.map_image.shape[0] / self.dpi - .5, -.5],
                             cmap='Greys_r')
        self.scatter_interact.points = traj
        self.scatter_interact.draw_plot_only()
        self.ax[0, 1].set_xlim(self.ax[0, 0].get_xlim())
        self.ax[0, 1].set_ylim(self.ax[0, 0].get_ylim())
        self.canvas.draw()
        self._refine_plot()

    def _refine_plot(self):
        self._update_pos_constraints()
        self.start = int(self.map_start_var.get())
        self.end = int(self.map_end_var.get())
        self.step = int(self.map_step_var.get())
        self.scatter_interact.refine_plot(self.start, self.end, self.step)
        self.canvas.get_tk_widget().focus_set()

    def _remove_points(self):
        if self.scatter_interact:
            self.scatter_interact.remove_all_active_circles()
            self.canvas.get_tk_widget().focus_set()
        self._update_pos_constraints()

    def _quit(self):
        plt.close('all')
        self.root.destroy()

    def run(self):
        self.root.mainloop()


class Cmd():
    # command line optimization when all the hyper parameters are known (faster)
    def __init__(self, map_image, dpi, origin_traj, args, ts, vel_norm, vel_angle, map_aligned_points=None,
                 starting_params={}, loop=False):

        self.map = map_image
        self.dpi = dpi

        self.origin_traj = origin_traj
        self.refined_traj = origin_traj

        self.args = args
        self.vel_norm = vel_norm
        self.vel_angle = vel_angle
        self.ts = ts

        self.map_aligned_points = map_aligned_points
        self.origin_params = starting_params
        self.loop = loop

        self.prev_pos_energy, self.current_pos_energy = None, None

        start_t = time.time()
        self._init()

        self._optimize()
        self.status = Status.OPTIMIZING
        i = 1
        if loop:
            # loop while stopping conditions are not met and position prior deceases
            while i < 10 and self.lsq.status < 1 and self.prev_pos_energy / self.current_pos_energy - 1 > 1 * 1e-03:
                print('Looping : ', i, self.prev_pos_energy / self.current_pos_energy - 1)
                self._optimize()
                i += 1
        end_t = time.time()
        print('All optimization done. iterations: {}, time {:.2f} min'.format(args.n_iterations * i,
                                                                              (end_t - start_t) / 60))

    def _init(self):
        self.variables, self.functions = OrderedDict(), OrderedDict()
        self.compound_functor = CompoundFunctor(self.variables, self.functions)
        self.starting_params = self.origin_params
        self.solution = None
        self.lsq = None

        self._init_params()
        self._init_constraints()

    def _init_params(self):
        # create all variables and functions
        args = self.args
        self.variables['pos'] = Variable(name='pos', length=2, active=True, required=True,
                                         default_value=0, last_value=self.starting_params.get('pos', None))
        self.variables['yaw'] = Variable(name='yaw', length=0, active=False, required=False, default_value=0)  # unused

        scale_params, scale_index = 1, None
        if args.scale_interval > 0:
            scale_params = math.ceil(len(self.vel_norm) / args.scale_interval) + 1
            scale_index = np.arange(0, scale_params) * args.scale_interval
        print('Adding scale per every {} frames, {} params'.format(args.scale_interval / args.traj_freq, scale_params))
        if self.starting_params.get('scale'):
            if len(self.starting_params.get('scale')) < scale_params:
                self.starting_params.get('scale').append(1)
        self.variables['scale'] = Variable(name='scale', length=scale_params,
                                           required=False, default_value=1,
                                           last_value=self.starting_params.get('scale', None))
        self.variables['scale'].set_state(args.scale)
        self.functions['scale_regularize'] = Function(name='scale_regularize', weight=args.scale_reg_weight,
                                                      select_param=True, param='scale')
        self.functions['scale_regularize'].func = RatioPriorFunctor()

        bias_params = math.ceil(len(self.vel_angle) / args.bias_interval) + 1
        bias_index = np.arange(0, bias_params) * args.bias_interval
        print('Adding bias per every {} frames, {} params'.format(args.bias_interval / args.traj_freq, bias_params))
        self.variables['bias'] = Variable(name='bias', length=bias_params, required=False,
                                          default_value=0, bias_index=bias_index,
                                          last_value=self.starting_params.get('bias', None))
        self.variables['bias'].set_state(args.bias)
        self.functions['bias_regularize'] = Function(name='bias_regularize', weight=args.bias_reg_weight,
                                                     select_param=True,
                                                     param='bias')
        self.functions['bias_regularize'].func = DriftPriorFunctor(weight=0.6)

        self.variables['noise'] = Variable(name='noise', length=0, active=False, required=False,
                                           default_value=0)  # unused

        f_loss = NormPriorFunctor() if args.pos_prior_loss > 0 else None
        self.functions['pos_prior'] = Function(name='pos_prior', weight=args.pos_prior_weight)
        self.functions['pos_prior'].func = PositionPriorFunctor(self.vel_norm, self.vel_angle, self.ts,
                                                                bias_index=bias_index, scale_index=scale_index,
                                                                noise_index=None,
                                                                loss_func=f_loss, correct_angle=True,
                                                                kind=self.args.interpolate_kind)

    def _init_constraints(self):
        print('Selected constraints: ', len(self.map_aligned_points))
        if len(self.map_aligned_points) > 0 and not self.starting_params:
            self.variables['pos'].last_value = self.map_aligned_points[0, 1:3]

        self.functions['pos_prior'].func.set_pos_constraints(self.map_aligned_points[:, 0],
                                                             self.map_aligned_points[:, 1:3],
                                                             self.map_aligned_points[:, 3])
        # rigid transformation
        print('Starting rigid transformation')
        start_t = time.time()
        self.variables['scale'].set_state('none')
        self.functions['bias_regularize'].weight = 1e08
        solution = get_active_params(self.variables)
        lsq = least_squares(self.compound_functor, solution, loss='linear',
                            max_nfev=20, gtol=1e-15, xtol=1e-15, ftol=1e-15,
                            verbose=2 if self.args.verbose else 1)
        end_t = time.time()
        print('Optimization done. [Time: {} s]'.format(end_t - start_t))

        solution = lsq.x
        store_as_last_params(self.variables, solution)
        v_pos, v_yaw, v_scale, v_bias, v_noise = self.compound_functor.get_variables(solution)
        print('\tinit: ', v_pos)
        print('\tbias ', v_bias, '\n')
        self.refined_traj = self.functions['pos_prior'].func.get_modified_trajectory(v_pos, v_yaw, v_bias, v_noise,
                                                                                     v_scale)

        # back to normal
        self.variables['scale'].set_state(self.args.scale)
        self.functions['bias_regularize'].weight = self.args.bias_reg_weight
        self.status = Status.INIT

    def _update_with_solution(self, verbose=True):
        self.solution = get_active_params(self.variables)

        v_pos, v_yaw, v_scale, v_bias, v_noise = self.compound_functor.get_variables(self.solution)
        if v_bias is not None and self.variables['bias'].get_current_state() in ['active', 'last']:
            v_bias = adjust_angle_arr(v_bias)
            self.variables['bias'].last_value = v_bias

        if verbose:
            print('\tinit: ', v_pos)
            print('\tscale ', v_scale, '\n')
            print('\tbias ', v_bias, '\n')
        self.refined_traj = self.functions['pos_prior'].func.get_modified_trajectory(v_pos, v_yaw, v_bias, v_noise,
                                                                                     v_scale)
        self._update_energy()

    def _update_energy(self):
        # energy functions
        self.prev_pos_energy = self.current_pos_energy
        self.current_pos_energy = self.compound_functor.get_energy(self.solution, 'pos_prior')
        print('ENERGY:\t position: {} -> {}'.format(self.prev_pos_energy, self.current_pos_energy))

    def _optimize(self):
        self._update_with_solution(verbose=False)

        print('Starting optimization')
        start_t = time.time()
        self.lsq = least_squares(self.compound_functor, self.solution, loss='linear',
                                 max_nfev=self.args.n_iterations, gtol=1e-15, xtol=1e-15, ftol=1e-15,
                                 verbose=2 if self.args.verbose else 1)
        self.solution = self.lsq.x
        end_t = time.time()
        print('Optimization done. [Time: {} s]'.format(end_t - start_t))

        store_as_last_params(self.variables, self.solution)
        self._update_with_solution()


if __name__ == '__main__':
    with open(osp.join('../data_paths.json'), 'r') as f:
        default_config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="[Optional] file to load configuration from")
    parser.add_argument('--data_path', type=str, default=None, help="Path to folder containing hdf5 and json files")
    parser.add_argument('--data_list', type=str, default=None)
    parser.add_argument('--data_dir', type=str)

    parser.add_argument('--type', type=str, choices=['gt', 'ronin', 'raw'])
    parser.add_argument('--traj_freq', type=int, help='Frequency of trajectory data')

    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--prefix', type=str)

    # specify a single map_file with data_path or specify map_dir and map_csv_dir (required only if flp priors are used)
    parser.add_argument('--map_path', type=str, default=None, help="Path to map image")
    parser.add_argument('--map_dpi', type=int)
    parser.add_argument('--map_latlong_path', type=str, default=None,
                        help="Constraints to map floorplan with FLP data (default: <map_path>.csv)")
    parser.add_argument('--map_dir', type=str, default=default_config.get('floorplan_dir', None))
    parser.add_argument('--map_csv_dir', type=str, default=None)

    parser.add_argument('--starting_variables', type=str,
                        help='Path to file containing optimization varibles (ensure same number of parameter)')
    parser.add_argument('--pos_prior_type', type=str, default='flp', choices=['flp', 'manual', 'from_file'],
                        help='Position prior type [flp or manual positions are in hdf5 file, or specify input file')
    parser.add_argument('--flp_adjust_radius', type=float, default=2, help="Factor to adjust FLP radius")
    parser.add_argument('--flp_sample', type=int, default=60, help="Sparse FLP points, sample every x seconds")
    parser.add_argument('--pos_prior_file', type=str, default=None,
                        help='Path to file containing positions priors, if pos_prior_type is from file')

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
    data_list, map_dir, map_csv_dir, root_dir, single = [], None, None, None, False
    if args.data_path is not None:
        single = True
        if args.data_path[-1] == '/':
            args.data_path = args.data_path[:-1]
        if args.map_latlong_path is None:
            args.map_latlong_path = args.map_path.rsplit('.', 1)[0] + '.csv'
        data_list.append(
            [osp.split(args.data_path)[1], osp.split(args.map_path)[1], osp.split(args.map_latlong_path)[1]])

        root_dir = osp.split(args.data_path)[0]
        if len(root_dir) == 0: root_dir = args.data_dir
        map_dir = osp.split(args.map_path)[0]
        if len(map_dir) == 0: map_dir = args.map_dir
        map_csv_dir = osp.split(args.map_latlong_path)[0]
        map_csv_dir = map_csv_dir if len(map_csv_dir) > 0 else \
            args.map_csv_dir if args.map_csv_dir is not None else map_dir
    elif args.data_list is not None:
        # data list should have data path, map name, csv file name
        # cannot work with starting params and position_priors from file
        root_dir = args.data_dir
        with open(args.data_list) as f:
            for s in f.readlines():
                if len(s) > 0 and s[0] != '#':
                    s = s.strip().split()
                    data_list.append([s[0].strip(" \'\""), s[1].strip(" \'\""), s[2].strip(" \'\"")])
        map_dir = args.map_dir
        map_csv_dir = args.map_csv_dir if args.map_csv_dir is not None else map_dir
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
                                                                               type=args.type,
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
        gt_points, flp_points = None, None

        with h5py.File(osp.join(root_dir, data_info[0], 'data.hdf5'), 'r') as f:
            if 'filtered/locations' in f.keys():
                gt_points = np.copy(f['filtered/locations'])
        if gt_points is not None and len(gt_points) > 0:
            gt_points = match_timestamps(ts, gt_points[:, 0], gt_points[:, 1:3] / dpi, allow_err=5)
        else:
            gt_points = None

        flp_data = process_flp_data(map_image, dpi, osp.join(map_csv_dir, data_info[2]),
                                    osp.join(root_dir, data_info[0]), visualize=False)
        if flp_data is not None:
            flp_points = match_timestamps(ts, flp_data[:, 0], flp_data[:, 1:], allow_err=1)

        if args.pos_prior_type == 'manual':
            map_aligned_points = np.concatenate([gt_points[:, :3], np.ones([len(gt_points), 1]), gt_points[:, 3:]],
                                                axis=1)
            print('Loaded manual position priors. {} selected'.format(len(map_aligned_points)))
        elif single and args.pos_prior_type == 'from_file' and args.spos_prior_file is not None:
            # format: point_idx, pos_x, pos_y, radius, timestamp (same as final_constraints.txt saved from this program)
            map_aligned_points = np.loadtxt(args.pos_prior_file)
            map_aligned_points = match_timestamps(ts, map_aligned_points[:, -1], map_aligned_points[:, 1:-1],
                                                  allow_err=5)
            print('Loaded position priors from file. {} selected'.format(len(map_aligned_points)))
        elif args.pos_prior_type == 'flp':
            if flp_data is not None:
                flp_data[:, -1] /= args.flp_adjust_radius
                if args.flp_sample > 1:
                    flp_data = find_sparse(args.flp_sample, flp_data[:, 0], flp_data[:, 1:])
                    flp_points = np.concatenate(
                        [flp_points, np.in1d(flp_points[:, -1], flp_data[:, 0], assume_unique=True)[:, None]], axis=1)
                else:
                    flp_points = np.concatenate([flp_points, np.ones(len(flp_points), 1)], axis=1)
                map_aligned_points = match_timestamps(ts, flp_data[:, 0], flp_data[:, 1:], allow_err=1)
                print('Loaded flp as position priors. {} selected'.format(len(map_aligned_points)))
        if args.no_gui and map_aligned_points is None:
            print('Skipping: no position priors selected')
            continue

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
                        filename=osp.join(args.out_dir, 'final_trajectory_with_constraints.png'),
                        pos_constraints=map_aligned_points, calc_errors=True)
            if gt_points is not None:
                create_plot(map_image, dpi, optimizer.refined_traj, save_files=True,
                            filename=osp.join(args.out_dir, 'final_trajectory_with_gt.png'),
                            pos_constraints=gt_points, calc_errors=True)
            with open(osp.join(args.out_dir, 'final_optimization_variable.txt'), 'w') as f:
                var = create_variable_dict(optimizer.compound_functor.get_variables(optimizer.solution),
                                           correct_angle=True)
                json.dump(var, f)
