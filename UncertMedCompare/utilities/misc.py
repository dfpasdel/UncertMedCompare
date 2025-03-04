import numpy as np


def weighted_error_mean_and_std(ref_values, new_values, sample_weight=None):
    ref_values = np.asarray(ref_values)
    new_values = np.asarray(new_values)
    diff = new_values - ref_values
    diff_mean = np.average(diff, weights=sample_weight)
    diff_variance = np.average((diff - diff_mean) ** 2, weights=sample_weight)
    diff_std = np.sqrt(diff_variance)
    return diff_mean, diff_std


def rolling_mean(x, n):
    assert(n % 2 == 1)
    c = np.cumsum(np.insert(x, 0, 0))
    m = (c[n:] - c[:-n]) / float(n)
    m = np.pad(m, pad_width=((int((n-1) / 2), int((n-1) / 2))), mode="edge")
    return m