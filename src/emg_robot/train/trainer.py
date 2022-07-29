from soupsieve import match
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import OrderedDict
import re
import time
import os

from emg_robot.prepare.features import all_features


EMG_CHANNELS = 5
NUM_FEATURES = len(all_features)
BATCH_SIZE = 1
WINDOW_LENGTH = 150
NUM_INPUT_FEATURES = EMG_CHANNELS * NUM_FEATURES
LOOKBACK_STEPS = 5
HIDDEN_SIZE = 100
OUTPUT_SIZE = 2  # pitch & roll of the forearm


class RNNModel(torch.nn.Module): 
    def __init__(self) -> None:
        super().__init__()

        # See https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.rnn = nn.RNN(input_size=NUM_INPUT_FEATURES, hidden_size=HIDDEN_SIZE, num_layers=LOOKBACK_STEPS, bidirectional=True)
        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = torch.zeros(2 * LOOKBACK_STEPS, batch_size, HIDDEN_SIZE)

        out, hidden = self.rnn(x, hidden)

        out = out.contiguous().view(-1, HIDDEN_SIZE)
        out = self.fc(out)

        return out, hidden


def load_data(dir):
    # Group files by common prefix
    regex = re.compile(r'(.*)_(c[AD][12])_features\.csv')
    sets = OrderedDict()
    keys = []
    files = sorted(os.listdir(dir))
    for f in files:
        match = regex.search(f)
        if match:
            keys.append(match[1])
            sets.setdefault(match[1], []).append(f)

    # Load files and concatenate those that belong together
    datasets = []
    for group in sets:
        group = sorted(group)
        frames = []
        for f in group:
            label = regex.search(f)
            data = pd.read_csv(f)
            data.rename(columns=lambda l: f'{l}_{label[2]}')
            frames.append(data)
        # Stack horizontally so all features for each frame are next to each other
        datasets.append(pd.concat(frames, axis=1))

    # Load the groundtruth for each recording
    groundtruths = []
    for k in keys:
        try:
            gt = pd.read_csv(os.path.join(dir, k + '_orientation_kalman.csv'))
        except IOError:
            gt = pd.read_csv(os.path.join(dir, k + '_orientation.csv'))
        groundtruths.append(gt)

    # Stack vertically to create one big dataset
    training_data = pd.concat(datasets, axis=0, keys=keys)
    groundtruth_data = pd.concat(groundtruths, axis=0, keys=keys)

    return torch.from_numpy(training_data.values), \
           torch.from_numpy(groundtruth_data.values)


def train(data, groundtruth):
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

    data.to(device)
    for epoch in range(1, epochs + 1):
        # Clear existing gradients from previous epoch
        optimizer.zero_grad()
        output, _ = model(data)
        loss = end_condition(output, groundtruth.view(-1).long())
        
        # Actual training step
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}/{epochs}.............', end=' ')
            print(f'Loss: {loss.item():.4f}')
    
    return model



if __name__ == '__main__':
    data_dir = ''
    
    data, gt = load_data(data_dir)
    model = train(data, gt)

    # Load like this:
    # 
    # model = RNNModel(...)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()
    model_name = f'model_{time.now()}.torch'
    torch.save(model.state_dict(), os.path.join(dir, model_name))
