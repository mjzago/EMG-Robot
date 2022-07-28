from unicodedata import bidirectional
from torch.nn import RNN


WINDOW_LENGTH = 150
NUM_FEATURES = 57
NUM_LAYERS = 5

emg_dir = ''
gt_dir = ''


# Input shape will be [NUM_WINDOWS, WINDOW_LENGTH, NUM_FEATURES]
# Output shape will be [NUM_WINDOWS, WINDOW_LENGTH, NUM_PREDICTIONS]
# Hidden shape will be [NUM_LAYERS * 2, NUM_WINOWS, NUM_PREDICTIONS] (if bidirectional)
rnn = RNN(input_size=NUM_FEATURES, hidden_size=2, num_layers=NUM_LAYERS, bidirectional=True)


def load_data():
    pass # TODO


def train_ai():
    pass # TODO


