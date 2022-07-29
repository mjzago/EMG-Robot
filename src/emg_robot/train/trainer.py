from soupsieve import match
import torch
from torch.nn import RNN
import numpy as np
import pandas as pd
import re
import os

from emg_robot.prepare.features import all_features


EMG_CHANNELS = 5
NUM_FEATURES = len(all_features)
BATCH_SIZE = 1
WINDOW_LENGTH = 150
NUM_INPUT_FEATURES = EMG_CHANNELS * NUM_FEATURES
LOOKBACK_STEPS = 5
PREDICTION_STEPS = 1

emg_dir = ''
gt_dir = ''


# See https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
rnn = RNN(input_size=NUM_INPUT_FEATURES, hidden_size=PREDICTION_STEPS, num_layers=LOOKBACK_STEPS, bidirectional=True)
hidden_state = torch.rand(2 * LOOKBACK_STEPS, BATCH_SIZE, PREDICTION_STEPS)


def load_data(dir):
    regex = re.compile(r'(.*)_(c[AD][12])_features\.csv')
    sets = {}

    # Group files by common prefix
    for f in os.listdir(dir):
        match = regex.search(f)
        if match:
            sets.setdefault(match[1], []).append(f)

    # Load files and concatenate those that belong together
    keys = []
    datasets = []
    for group in sets:
        group = sorted(group)
        frames = []
        for f in group:
            w_type = regex.search(f)[2]
            data = pd.read_csv(f)
            data.rename(columns=lambda l: f'l_{w_type}')
            frames.append(data)
        keys.append(w_type[1])
        # Stack horizontally so all features for each frame are next to each other
        datasets.append(pd.concat(frames, axis=1))

    # Stack vertically to create one big dataset
    df = pd.concat(datasets, axis=0, keys=keys)
    return torch.from_numpy(df.values)


def train():
    pass # TODO


def save_model():
    pass # TODO
