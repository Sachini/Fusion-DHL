import os.path as osp
import sys
from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
from util.math_utils import adjust_angle_arr


class CompoundFunctor:
    """
    Holds residual functions that should be used for least square optimization.
    :param has_scale - True, if the optimization variables include scale parameters in addition to translation.

    When has_scale is true, use:
     - add_functor() for functions that only operates on translation params
     - add_scaled_functor() for functions that takes in translation and scale transformation
    If has_scale is false, all functions are handled the same.
    """

    def __init__(self, variables, functors):
        self.variables = variables
        self.functors = functors

    def add_functor(self, functor):
        self.functors[functor.name] = functor
        print("Adding {} with weight {}".format(functor.name, functor.weight))

    def _fetch_value(self, var, x, a, active_only=False):
        if var.active:
            return x[a:a + var.length], a + var.length
        elif active_only:
            return None, a
        elif var.last:
            return var.last_value, a
        elif var.default:
            return var.default_value, a
        else:
            return None, a

    def get_variables(self, x):
        a = 0
        v_init_pos, a = self._fetch_value(self.variables['pos'], x, a)
        v_init_yaw, a = self._fetch_value(self.variables['yaw'], x, a)
        v_scale, a = self._fetch_value(self.variables['scale'], x, a)
        v_bias, a = self._fetch_value(self.variables['bias'], x, a)
        v_noise, _ = self._fetch_value(self.variables['noise'], x, a)

        if v_bias is not None:
            v_bias = adjust_angle_arr(v_bias)
        return v_init_pos, v_init_yaw, v_scale, v_bias, v_noise

    def get_energy(self, x, func_name):
        """
        x -   [init_x, init_y, init_ori]
              [scale factor (D:1)]
              [bias correction (bx1, D:0)]
              [noise correction (nx1, D:0)]
        """
        v_init_pos, v_init_yaw, v_scale, v_bias, v_noise = self.get_variables(x)
        residual = []

        assert func_name in self.functors
        f = self.functors[func_name]
        if f.select_param:
            assert self.variables[f.param].active
            if f.param == 'bias':
                residual.append(f.func(v_bias) * f.weight)
            elif f.param == 'noise':
                residual.append(f.func(v_noise) * f.weight)
            elif f.param == 'scale':
                residual.append(f.func(v_scale) * f.weight)
            elif f.param == 'yaw':
                residual.append(f.func(v_init_yaw) * f.weight)
        else:
            residual.append(f.func(v_init_pos, v_init_yaw, v_bias, v_noise, v_scale) * f.weight)

        energy = np.concatenate(residual, axis=0)
        return np.sqrt(np.average(np.power(energy, 2)))

    def __call__(self, x, *args, **kwargs):
        """
        x -   [init_x, init_y, init_ori]
              [scale factor (D:1)]
              [bias correction (bx1, D:0)]
              [noise correction (nx1, D:0)]
        """
        v_init_pos, v_init_yaw, v_scale, v_bias, v_noise = self.get_variables(x)
        residual = []

        for key, f in self.functors.items():
            if not f.is_active(): continue
            if f.select_param:
                if not self.variables[f.param].active: continue
                if f.param == 'bias':
                    err = f.func(v_bias) * f.weight
                elif f.param == 'noise':
                    err = f.func(v_noise) * f.weight
                elif f.param == 'scale':
                    err = f.func(v_scale) * f.weight
                elif f.param == 'yaw':
                    err = f.func(v_init_yaw) * f.weight
            else:
                err = f.func(v_init_pos, v_init_yaw, v_bias, v_noise, v_scale) * f.weight

            # normalize error
            residual.append(err / len(err))

        return np.concatenate(residual, axis=0)


class NormPriorFunctor:
    # Calculate norm of each element of array with mean=value

    def __init__(self, norm=None, value=0, aggregate=None, factor=1):
        self.norm = norm
        self.mean = value
        self.factor = factor
        self.aggregate = aggregate

    @property
    def aggregate(self):
        return self._aggregate

    @aggregate.setter
    def aggregate(self, value):
        if value not in ['mean', 'sum', None]:
            print('Valid aggregate functions are mean, sum or None')
            return
        self._aggregate = value

    def __call__(self, x):
        if self.norm is None:
            loss = x - self.mean
        else:
            loss = np.power(np.abs(x - self.mean) * self.factor, self.norm)
        if self._aggregate is None:
            return loss.flatten()
        elif self._aggregate == 'mean':
            return np.mean(loss)
        else:  # self._aggregate == 'sum'
            return np.sum(loss)


class RatioPriorFunctor:
    # scale regularization
    def __call__(self, x):
        l = np.clip(x, a_min=1e-3, a_max=1e03)
        pl = np.maximum(l, 1 / l) - 1
        return pl


class DriftPriorFunctor:
    # linear and constant prior for angle regularization
    def __init__(self, weight):
        self.pl_w = weight

    def __call__(self, x):
        pc = x[1:] - x[:-1]
        pl = x[2:] + x[:-2] - 2 * x[1:-1]
        return np.append(pc * (1 - self.pl_w), pl * self.pl_w)


class ResidualFunctions(ABC):
    """
    Parent class of all function related to navigation priors.
    :param vel_m - nx1 array
    :param angle - (norm-1)x1 array of velocity direction (if correct_angle=False, pass angle_rate instead of angle
    here)
    :param ts - timestamp coorsponding to feat.
    :param bias_index - frames corresponding to optimization bias variables
    :param scale_index - frames corresponding to optimization scale variables
    :param noise_index - frames corresponding to optimization noise variables
    :param local - True if feat are in local CF.
    :param correct_angle - variable correct angle instead of angle_rate
    :param kind - interpolation kind for piecewise_variables

    Note that when correct_angle is True, bias refers to angle correction of velocity direction and yaw parameter is
    None. When correct_angle is False,
    bias refers to correction of angular_rate and yaw is the velocity direction of frame zero.
    """

    def __init__(self, vel_m, angle, ts, bias_index=None, scale_index=None, noise_index=None, loss_func=None,
                 correct_angle=True, kind='linear'):
        super(ResidualFunctions, self).__init__()
        self.vel_m = vel_m
        self.vel_yaw = angle
        self.dt = np.median(ts[1:] - ts[:-1])
        self.bias_index = bias_index
        self.scale_index = scale_index
        self.noise_index = noise_index
        self.feat_index = np.arange(0, self.vel_m.shape[0])

        self.loss_func = loss_func
        self.correct_angle = correct_angle
        self.kind = kind

    def get_processed_bias(self, x):
        # return interpolated bias values from 0 to end_frame
        init_bias = interp1d(self.bias_index, x, axis=0, bounds_error=False, kind=self.kind,
                             fill_value=0)(self.feat_index)
        return init_bias

    def get_processed_scale(self, x):
        if self.scale_index is None:
            return x
        else:
            return interp1d(self.scale_index, x, axis=0, bounds_error=False,
                            fill_value=1)(self.feat_index)[:, None]

    def get_processed_noise(self, x):
        noise = interp1d(self.noise_index, x, kind='previous', bounds_error=False, fill_value='extrapolate')(
            self.feat_index)
        noise[:self.noise_index[0]] = 0
        return noise

    def get_modified_yaw(self, init_yaw, bias=None, noise=None):
        # return corrected yaw from 0 to end_frame
        if self.correct_angle and bias is not None:
            init_bias = self.get_processed_bias(bias)
            mod_yaw = self.vel_yaw + init_bias
        elif not self.correct_angle:
            if bias is not None:
                init_bias = self.get_processed_bias(bias)
                mod_ang_rate = self.vel_yaw + init_bias[1:]  # bias[0] should be near zero
            mod_yaw = np.zeros(self.vel_m.shape)
            mod_yaw[0] = np.arctan2(init_yaw[0], init_yaw[1])
            mod_yaw[1:] = np.cumsum(mod_ang_rate) + mod_yaw[0]
        else:
            mod_yaw = np.copy(self.vel_yaw)

        if noise is not None:
            mod_yaw += self.get_processed_noise(noise)

        return mod_yaw

    def get_modified_trajectory(self, init_p, init_yaw, bias=None, noise=None, scale=None):
        mod_yaw = self.get_modified_yaw(init_yaw, bias, noise)

        mod_vel = np.stack([np.cos(mod_yaw), np.sin(mod_yaw)], axis=1) * self.vel_m[:, None] * self.dt
        if scale is not None:
            mod_vel *= self.get_processed_scale(scale)
        mod_traj = np.zeros([self.vel_m.shape[0] + 1, 2])
        mod_traj[0] = init_p
        mod_traj[1:] = np.cumsum(mod_vel, axis=0) + init_p

        return mod_traj

    @abstractmethod
    def __call__(self, init_p, init_yaw, bias, noise=None, scale=None):
        pass


class PositionPriorFunctor(ResidualFunctions):
    """
    Priors from pre-identified map positions
    :param c_frames, c_pos, c_radius: : pre_identified constraints
    :param loss_func : loss function of type NormPriorFunctor
    """

    def __init__(self, vel_m, angle, ts, bias_index, scale_index=None, noise_index=None, loss_func=None,
                 correct_angle=True, kind='linear'):
        super(PositionPriorFunctor, self).__init__(vel_m, angle, ts, bias_index, scale_index, noise_index, loss_func,
                                                   correct_angle, kind)

        self.c_frames = None
        self.c_pos = None
        self.c_radius = None

        if loss_func is None:
            self.loss_func = NormPriorFunctor()

    def set_pos_constraints(self, c_frames, c_pos, c_radius):
        self.c_frames = c_frames.astype(int)
        self.c_pos = c_pos
        self.c_radius = c_radius

    def pos_loss(self, pos):
        dist = np.linalg.norm(pos - self.c_pos, axis=1)
        return np.maximum(dist - self.c_radius / 2, 0)

    def __call__(self, init_p, init_yaw, bias=None, noise=None, scale=None):
        traj = self.get_modified_trajectory(init_p, init_yaw, bias, noise, scale)
        return self.loss_func(self.pos_loss(traj[self.c_frames]))


class MapPriorFunctor(ResidualFunctions):
    """
    Priors from pre-identified map positions
    :param map_image : grayscale image (0 if walkable space, 1 if wall)
    :param m_loopdpi: pixels per meter in map image
    :param loss_func : loss function of type NormPriorFunctor
    """

    def __init__(self, vel_m, angle, ts, bias_index, map_image, m_dpi, loss_func=None, active=False, scale_index=None,
                 noise_index=None,
                 correct_angle=True, kind='linear'):
        super(MapPriorFunctor, self).__init__(vel_m, angle, ts, bias_index, scale_index, noise_index, loss_func,
                                              correct_angle, kind)

        self.map = map_image
        self.dpi = m_dpi
        self.map_size = np.asarray([map_image.shape[1] - 1, map_image.shape[0] - 1]) / m_dpi
        self.map_func = RegularGridInterpolator((np.arange(0, map_image.shape[1]) / m_dpi,
                                                 np.arange(0, map_image.shape[0]) / m_dpi), map_image.T,
                                                method='linear')
        if loss_func is None:
            self.loss_func = NormPriorFunctor()
        self.active = active
        self._out_of_bounds_loss = 10000

    def map_loss(self, pos):
        p_loss = np.zeros(len(pos))

        # check out of bound
        condition = np.logical_and(np.logical_and(0 <= pos[:, 0], pos[:, 0] <= self.map_size[0]),
                                   np.logical_and(0 <= pos[:, 1], pos[:, 1] <= self.map_size[1]))
        p_loss[condition == False] = self._out_of_bounds_loss
        p_loss[condition] = self.map_func(pos[condition])

        return p_loss

    def __call__(self, init_p, init_yaw, bias=None, noise=None, scale=None):
        traj = self.get_modified_trajectory(init_p, init_yaw, bias, noise, scale)
        return self.loss_func(self.map_loss(traj))
