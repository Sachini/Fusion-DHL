import os.path as osp

import h5py
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import utm
from scipy.interpolate import interp2d, interp1d


def convert_to_utm(input_location):
    """
    Convert gps location coordinates to utm coordinates
    :param input_location: list of gps locations with timestamps (time, latitude, longitude)
    :return: numpy array of utm locations in same timestamp order (easting, northing)
    """
    utm_data = [utm.from_latlon(location[0], location[1])[:2] for location in input_location]
    return np.asarray(utm_data)[:, :2].astype(np.float)


def _find_nearest(target, source):
    # Find indices of nearest values of source in target (e.g match two timestamps) Assume two arrays are sorted.
    source = np.atleast_1d(source)
    assert np.all(np.diff(target) >= 0), 'Target array is not sorted'
    assert np.all(np.diff(source) >= 0), 'Value array is not sorted'

    indices, s = [], 0
    for v in source:
        i = np.abs(target[s:] - v).argmin()
        indices.append(s + i)
        s = i
    indices = np.asarray(indices)
    return indices


def find_sparse(time_interval, flp_time, flp_data):
    # Pick data points from flp data series at a given interval
    timestamp = np.arange(flp_time[0], flp_time[-1], time_interval)
    new_time_indices = _find_nearest(flp_time, timestamp)
    return np.concatenate([flp_time[new_time_indices][:, None], flp_data[new_time_indices]], axis=1)


def match_timestamps(timestamp, flp_time, flp_data, allow_err=0, interpolate=False):
    """
    Match FLP data to the nearest timestamps
    :param timestamp: array (l) timestamp of data points in seconds
    :param flp_time: array (n) timestamp of fla data points in seconds
    :param flp_data: ndarray (nxm), data series
    :param allow_err: use data before and after the data series (in seconds)
    :param interpolate: interpolate data to match the timestamps
    :return: matched [indices of timestamp, data...(m), timestamp] x n
    """
    valid_points = \
    np.where(np.logical_and(flp_time >= timestamp[0] - allow_err, flp_time <= timestamp[-1] + allow_err))[0]
    flp_time, flp_data = flp_time[valid_points], flp_data[valid_points]
    new_time_indices = _find_nearest(timestamp, flp_time)
    if interpolate:
        new_data = interp1d(flp_time, flp_data, axis=0, bounds_error=False, fill_value="extrapolate")(
            timestamp[new_time_indices])
        return np.concatenate([new_time_indices[:, None], new_data, timestamp[new_time_indices][:, None]], axis=1)
    else:
        return np.concatenate([new_time_indices[:, None], flp_data, flp_time[:, None]], axis=1)


def latlong_coordinates_on_map(map, dpi, map_coordinates, flp_data, visualize=True):
    """
    Given map and lat/long of few map positions (map_coordinates) return the location on map of flp_coordinates
    :param map: image
    :param dpi: pixels per meter in image
    :param map_coordinates: ndarray [x, y, latitude, longitude] x n rows
        (x, y) are pixel position of point in image
    :param flp_data: [timestamp, latitude, longitude, horizontal_accuracy (meters),....] x m rows
    :param visualize: Display map if True

    :return: matching fla_coordinates [timestamp, x, y, radius]
        (x, y) are position of point in image in meters
    """

    map_coordinates[:, :2] /= dpi
    utm_coords = convert_to_utm(map_coordinates[:, 2:])

    map_func_x = interp2d(utm_coords[:, 0], utm_coords[:, 1], map_coordinates[:, 0], kind='linear', bounds_error=False)
    map_func_y = interp2d(utm_coords[:, 0], utm_coords[:, 1], map_coordinates[:, 1], kind='linear', bounds_error=False)

    flp_data[:, 1:3] = convert_to_utm(flp_data[:, 1:3])
    converted_data = np.asarray(
        [[map_func_x(flp_data[i, 1], flp_data[i, 2]), map_func_y(flp_data[i, 1], flp_data[i, 2])] for i in
         range(len(flp_data))])
    flp_data[:, 1:3] = converted_data.squeeze()

    if visualize:
        fig, ax = plt.subplots(figsize=(25, 25))
        ax.imshow(map, extent=[-0.5, map.shape[1] / dpi, map.shape[0] / dpi, -0.5], cmap='Greys_r')
        cmap = matplotlib.cm.rainbow(np.linspace(0, 1, len(flp_data)))
        for i in range(len(flp_data)):
            circle = patches.Circle((flp_data[i, 1], flp_data[i, 2]), radius=flp_data[i, 3], fc=cmap[i], alpha=0.2)
            ax.add_patch(circle)
        ax.plot(flp_data[:, 1], flp_data[:, 2])
        plt.show()

    return flp_data[:, :4]


def process_flp_data(map, dpi, map_path, data_path, visualize=True):
    """
    Given map and lat/long of few map positions (map_coordinates) return the location on map of fla_coordinates
    :param map: image
    :param dpi: pixels per meter in image
    :param map_path: Path to file
    :param visualize: Display map if True

    :return: matching fla_coordinates [timestamp, x, y, radius]
        (x, y) are position of point in image in meters
    """
    map_data = np.genfromtxt(map_path, delimiter=',')

    with h5py.File(osp.join(data_path, 'data.hdf5'), 'r') as f:
        if 'filtered/flp' in f:
            fla_data = np.copy(f['filtered/flp'])
        else:
            return None
    return latlong_coordinates_on_map(map, dpi, map_data, fla_data, visualize)
