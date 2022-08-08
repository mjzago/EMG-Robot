import os
import re
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..preprocessing.features import all_features

BATCH_SIZE = 4  # Number of data batches when training (should be a power of 2)
SEQ_LENGTH = 5  # The number of EMG windows to consider (i.e. how far back the RNN will remember)
NUM_LAYERS = 1  # How many stacks the RNN should have
HIDDEN_SIZE = 128  # TODO can probably be smaller (should be a power of ?2)
OUTPUT_SIZE = 2  # pitch & roll of the forearm

NUM_FEATURES = len(all_features)
EMG_CHANNELS = 5  # Number of EMG channels
WT_DECOMPOSITIONS = 3  # level 2 wavelet
NUM_INPUT_FEATURES = EMG_CHANNELS * NUM_FEATURES * WT_DECOMPOSITIONS


class RNNModel(torch.nn.Module): 
    def __init__(self) -> None:
        super().__init__()

        # See https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        # TODO input size must take ignored features into account
        self.rnn = nn.RNN(input_size=NUM_INPUT_FEATURES, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, input):
        batch_size = input.size(0)

        # Bidirectional introduces a factor of 2 in some places
        # Input shape is (BATCH_SIZE, SEQ_LENGTH, NUM_INPUT_FEATURES)
        # Output shape is (BATCH_SIZE, SEQU_LENGTH, 2 * HIDDEN_SIZE)
        # Hidden shape is (2 * NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE)
        hidden = torch.zeros(2 * NUM_LAYERS, batch_size, HIDDEN_SIZE)
        out, hidden = self.rnn(input, hidden)
        estimates = self.fc(out)

        return estimates, hidden


def load_data(dir, files=None, ignored_features=None):
    # Group files by common prefix
    regex = re.compile(r'(.*)_(c[AD][12])_features\.csv')
    sets = OrderedDict()
    keys = set()
    files = sorted(os.listdir(dir))
    for f in files:
        match = regex.search(f)
        if match:
            keys.add(match[1])
            sets.setdefault(match[1], []).append(f)

    # Load files and concatenate those that belong together
    datasets = []
    for group in sets.values():
        group = sorted(group)
        frames = []
        for f in group:
            label = regex.search(f)
            data = pd.read_csv(os.path.join(dir, f))
            data = data.rename(columns=lambda l: f'{l}_{label[2]}')
            frames.append(data)
        # Stack horizontally so all features for each frame are next to each other
        datasets.append(pd.concat(frames, axis=1))

    # Load the groundtruth for each recording
    groundtruths = []
    for k in sorted(keys):
        try:
            gt = pd.read_csv(os.path.join(dir, k + '_orientation_kalman_windowed.csv'))
        except FileNotFoundError:
            gt = pd.read_csv(os.path.join(dir, k + '_orientation_simple_windowed.csv'))
        groundtruths.append(gt)

    # Stack vertically to create one big dataset
    all_input = pd.concat(datasets, axis=0, keys=keys)
    all_groundtruth = pd.concat(groundtruths, axis=0, keys=keys).drop(columns='window')

    if ignored_features:
        ignored_features = ['f_' + f if not f.startswith('f_') else f for f in ignored_features]
        channels = [f'emg_{i}' for i in range(EMG_CHANNELS)]
        ignored_cols = [f'{c}_{f}' for f in ignored_features for c in channels]
        all_input[ignored_cols] = 0.0

    training_data = torch.from_numpy(all_input.values.astype(np.float32))
    groundtruth_data = torch.from_numpy(all_groundtruth.values.astype(np.float32))

    return TensorDataset(training_data, groundtruth_data)


def train(data):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Training RNN on CUDA device')
    else:
        device = torch.device('cpu')
        print('Training RNN on CPU')

    model = RNNModel()
    model.to(device)

    epochs = 100
    learning_rate = 0.01

    end_condition = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataloader = DataLoader(data, batch_size=BATCH_SIZE * SEQ_LENGTH, shuffle=True, drop_last=True)
    num_samples = len(dataloader.dataset)

    #data.to(device)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        for batch_id, (samples, groundtruth) in enumerate(dataloader):
            # Clear existing gradients from previous epoch
            optimizer.zero_grad()

            # Get the model's outputs
            output, _ = model(samples.view(BATCH_SIZE, SEQ_LENGTH, -1))
            loss = end_condition(output, groundtruth.view(BATCH_SIZE, SEQ_LENGTH, -1))
            
            # Actual training step
            loss.backward()
            optimizer.step()
        
            if batch_id % 100 == 0:
                loss, current = loss.item(), (batch_id + 1) * len(samples)
                print(f"loss: {loss:>7f}  [{current:>5d}/{num_samples:>5d}]")
    
    return model


def save_model(model, dir):
    # Load like this:
    # 
    # model = RNNModel(...)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()
    model_name = f'model_{time.now()}.torch'
    out_file = os.path.join(dir, model_name)
    torch.save(model.state_dict(), out_file)
    return out_file
