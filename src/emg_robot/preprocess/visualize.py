import plotly.express as px
import pandas as pd

import os
import re


def show_features(data):
    if isinstance(data, str):
        data = pd.read_csv(str)

    fig = px.imshow(data.to_numpy())
    fig.show()


if __name__ == '__main__':
    regex = re.compile(r'(.*)_(c[AD][12])_features\.csv')
    files = sorted(os.listdir("d:/Code/workspace/EMG Robot/datasets/test/preprocessed/"))

    for f in files:
        match = regex.search(f)
        if match:
            show_features(f)
