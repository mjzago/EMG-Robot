import numpy as np


def get_samples_per_window(window_length, dt):
    return int(window_length // dt)


def get_window_params(num_samples, samples_per_window, overlap):
    if isinstance(overlap, float):
        num_overlap = int(overlap * samples_per_window)
    else:
        num_overlap = overlap
        
    window_offset = int(samples_per_window - num_overlap)
    # Make sure that the final window can be long enough. Trailing
    # values will be ignored
    num_windows = int(num_samples // window_offset -
                      samples_per_window / window_offset + 1)
    return num_windows, window_offset


def get_window_intervals(num_samples, samples_per_window, overlap):
    if isinstance(overlap, int):
        r_overlap = overlap / samples_per_window
    else:
        r_overlap = overlap
    num_windows, _ = get_window_params(num_samples, samples_per_window, overlap)
    idx = np.arange(num_windows) * (1. - r_overlap)
    idx = np.tile(idx, (2, 1)).T
    idx[:, 1] += 1
    idx *= samples_per_window
    return idx.astype(np.int32)


def get_window_indices(num_windows, target_len):
    return np.repeat(np.arange(num_windows), target_len).astype(np.int32)