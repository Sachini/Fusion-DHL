import json
import math

import numpy as np


class Counter:
    def __init__(self, val=0):
        self.val = val

    def get_next(self, s=1):
        self.val += s
        return self.val - s

    def next_get(self, s=1):
        self.val += s
        return self.val


def ang_rate_to_angles(yaw, bias, bias_interval):
    if len(yaw) > 1:
        yaw = math.atan2(yaw[0], yaw[1])
    else:
        yaw = yaw[0]

    angle = [yaw + bias[0]]
    for i in range(1, len(bias)):
        angle.append((bias[i - 1] + bias[i]) * bias_interval / 2 + angle[-1])
    return angle


def angles_to_ang_rate(angles, bias_interval):
    yaw = [math.sin(angles[0]), math.cos(angles[0])]
    angles = np.asarray(angles) - angles[0]
    bias = [0]
    for i in range(1, len(angles)):
        bias.append((angles[i] - angles[i - 1]) / bias_interval * 2 - bias[-1])
    return yaw, bias


def load_config(default_config, args, unknown_args=None):
    # create dictionary from unknown args
    kwargs = {}

    def convert_value(y):
        try:
            return int(y)
        except:
            pass
        try:
            return float(y)
        except:
            pass
        if y == 'True' or y == 'False':
            return y == 'True'
        else:
            return y

    def convert_arrry(x):
        if not x:
            return True
        elif len(x) == 1:
            return x[0]
        return x

    i = 0
    if unknown_args:
        while i < len(unknown_args):
            if unknown_args[i].startswith('--'):
                token = unknown_args[i].lstrip('-')
                options = []
                i += 1
                while i < len(unknown_args) and not unknown_args[i].startswith('--'):
                    options.append(convert_value(unknown_args[i]))
                    i += 1
                kwargs[token] = convert_arrry(options)

    values = vars(args)

    def add_missing_config(dictionary, remove=False):
        for key in values:
            if values[key] is None and key in dictionary:
                values[key] = dictionary[key]
                if remove:
                    del dictionary[key]

    add_missing_config(kwargs, True)  # specified args listed as unknowns
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        add_missing_config(config)  # configuration from file for unspecified variables
    with open(default_config, 'r') as f:
        default_configs = json.load(f)
    add_missing_config(default_configs)

    return args, kwargs
