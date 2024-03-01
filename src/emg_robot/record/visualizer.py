import pandas as pd
import plotly.express as px
from augmentor import calc_emg_features, normalize


win_len = 30


def show(df):
    fig = px.line(df)
    fig.show()

if __name__ == '__main__':
    df = pd.read_csv('/home/robokind/EMG-Robot-Dataset/recordings/data-testebruno.csv')
    df = normalize(df[:])
    show(df)

    emg_cols = [c for c in df.columns if c.startswith('emg_')]
    calc_emg_features(df, win_len, emg_cols)
    for col in emg_cols:
        related = [c for c in df.columns if c.startswith(col)]
        show(df[related])
