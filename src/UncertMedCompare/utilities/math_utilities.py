import numpy as np


def nearest_idx(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def nearest_ceiling_idx(array, value):
    array = np.asarray(array)
    diff = array - value
    diff = np.where(diff < 0, diff, np.min(diff))
    return diff.argmax()
