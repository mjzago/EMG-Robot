import os
import re
import time
from collections import OrderedDict

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..preprocessing.features import all_features

BATCH_SIZE = 4  # Number of data batches when training
NUM_LAYERS = 1  # How many stacks the RNN should have
LOOKBACK_STEPS = 5  # Number of windows to consider
HIDDEN_SIZE = 100  # TODO can probably be much smaller
OUTPUT_SIZE = 2  # pitch & roll of the forearm

NUM_FEATURES = len(all_features)
EMG_CHANNELS = 5  # Number of EMG channels
WT_DECOMPOSITIONS = 3  # level 2 wavelet
NUM_INPUT_FEATURES = EMG_CHANNELS * NUM_FEATURES * WT_DECOMPOSITIONS


class RNNModel(torch.nn.Module): 
    def __init__(self) -> None:
        super().__init__()

        # See https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.rnn = nn.RNN(input_size=NUM_INPUT_FEATURES, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = torch.zeros(2 * NUM_LAYERS, batch_size, HIDDEN_SIZE)

        out, hidden = self.rnn(x, hidden)

        out = out.contiguous().view(-1, HIDDEN_SIZE)
        out = self.fc(out)

        return out, hidden


def load_data(dir, files=None):
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
            data = pd.read_csv(os.path.join(dir, f), sep=';')
            data = data.rename(columns=lambda l: f'{l}_{label[2]}')
            frames.append(data)
        # Stack horizontally so all features for each frame are next to each other
        datasets.append(pd.concat(frames, axis=1))

    # Load the groundtruth for each recording
    groundtruths = []
    for k in sorted(keys):
        try:
            gt = pd.read_csv(os.path.join(dir, k + '_orientation_kalman_windowed.csv'), sep=';')
        except IOError:
            gt = pd.read_csv(os.path.join(dir, k + '_orientation_windowed.csv'), sep=';')
        groundtruths.append(gt)

    # Stack vertically to create one big dataset
    training_data = torch.from_numpy(pd.concat(datasets, axis=0, keys=keys))
    groundtruth_data = torch.from_numpy(pd.concat(groundtruths, axis=0, keys=keys))

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

    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)
    num_samples = len(dataloader.dataset)

    data.to(device)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        for batch_id, (samples, groundtruth) in enumerate(dataloader):
            # Clear existing gradients from previous epoch
            optimizer.zero_grad()

            # Get the model's outputs
            output, _ = model(samples)
            loss = end_condition(output, groundtruth.view(-1).long())
            
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
