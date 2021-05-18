import numpy as np
import quaternion
import scipy


def interpolate_vector_linear(input, input_timestamp, output_timestamp):
    """
    This function interpolate n-d vectors (despite the '3d' in the function name) into the output time stamps.
    Args:
        input: Nxd array containing N d-dimensional vectors.
        input_timestamp: N-sized array containing time stamps for each of the input quaternion.
        output_timestamp: M-sized array containing output time stamps.
    Return:
        quat_inter: Mxd array containing M vectors.
    """
    assert input.shape[0] == input_timestamp.shape[0]
    func = scipy.interpolate.interp1d(input_timestamp, input, axis=0)
    interpolated = func(output_timestamp)
    return interpolated


def process_data_source(raw_data, output_time, method):
    input_time = raw_data[:, 0]
    if method == 'vector':
        output_data = interpolate_vector_linear(raw_data[:, 1:], input_time, output_time)
    elif method == 'quaternion':
        assert raw_data.shape[1] == 5
        output_data = interpolate_quaternion_linear(raw_data[:, 1:], input_time, output_time)
    else:
        raise ValueError('Interpolation method must be "vector" or "quaternion"')
    return output_data


def interpolate_quaternion_linear(data, ts_in, ts_out):
    """
    This function interpolate the input quaternion array into another time stemp.
    Args:
        data: Nx4 array containing N quaternions.
        ts_in: input_timestamp- N-sized array containing time stamps for each of the input quaternion.
        ts_out: output_timestamp- M-sized array containing output time stamps.
    Return:
        Mx4 array containing M quaternions.
    """

    assert np.amin(ts_in) <= np.amin(ts_out), 'Input time range must cover output time range'
    assert np.amax(ts_in) >= np.amax(ts_out), 'Input time range must cover output time range'
    pt = np.searchsorted(ts_in, ts_out)
    d_left = quaternion.from_float_array(data[pt - 1])
    d_right = quaternion.from_float_array(data[pt])
    ts_left, ts_right = ts_in[pt - 1], ts_in[pt]
    d_out = quaternion.quaternion_time_series.slerp(d_left, d_right, ts_left, ts_right, ts_out)
    return quaternion.as_float_array(d_out)


def adjust_angle_arr(yaw_angle):
    new_angle = np.copy(yaw_angle)
    yaw_diff = yaw_angle[1:] - yaw_angle[:-1]
    if np.any(np.abs(yaw_diff) > 4 * np.pi):
        print('Possible angle error', yaw_angle)

    yaw_diff_cand = yaw_diff[:, None] - np.array([-np.pi * 4, -np.pi * 2, 0, np.pi * 2, np.pi * 4])
    min_id = np.argmin(np.abs(yaw_diff_cand), axis=1)

    yaw_diff = np.choose(min_id, yaw_diff_cand.T)
    new_angle[1:] = np.cumsum(yaw_diff) + new_angle[0]
    return new_angle
