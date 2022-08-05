import os
import time

import numpy as np
import pandas as pd
import pywt
from imusensor.filters import kalman
from scipy.signal import butter, lfilter

from .features import all_features


EMG_RES = 2**12  # 12bit ADC
# TODO adjust
WINDOW_LENGTH_MS = 100  # How long each window should roughly be in ms
WINDOW_OVERLAP = 0.5
IMU_METHOD = 'kalman'  # simple | kalman

COL_DT = 'dt'
COL_EMG_CHANNELS = ['emg_0', 'emg_1', 'emg_2', 'emg_3', 'emg_4']
COL_IMU_ACCELS = ['acc_x', 'acc_y', 'acc_z']
COL_IMU_GYROS = ['gyro_x', 'gyro_y', 'gyro_z']


def get_col_label(channel, feature):
    return f"{channel}_{feature.split('_', 1)[1]}"


def get_dt(df):
    if isinstance(COL_DT, float):
        COL_DT
    return df[COL_DT].mean()


def calc_window_params(source_len, samples_per_window):
    window_offset = int(samples_per_window -
                        int(samples_per_window * WINDOW_OVERLAP))
    # Make sure that the final window can be long enough. Trailing values will be ignored
    return int(source_len // window_offset - samples_per_window / window_offset + 1), window_offset


def get_window_indices(num_windows, target_len):
    return np.repeat(np.arange(num_windows), target_len)


def get_window_length(df):
    return int(WINDOW_LENGTH_MS // (1000 * get_dt(df)))


def normalize(df):
    min = df.min(axis=0)
    return (df - min) / (df.max(axis=0) - min)


def filter_butterworth(data, lowcut, highcut, fs, order=5):
    # See https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if high > 1.0:
        print(
            f' ! Sampling frequency {fs}Hz is below butterworth highcut of {highcut}Hz')
        high = 0.99

    b, a = butter(order, [low, high], btype='band')

    print(
        f' >  Butterworth filter ({lowcut}, {highcut})... ', end='', flush=True)
    now = time.time()
    y = lfilter(b, a, data, axis=1)
    print(f'{time.time() - now : .3f}s')

    return y


def calc_features(df):
    index = [get_col_label(c, f.__name__)
             for c in df.columns[1:] for f in all_features]
    ret = pd.DataFrame(np.empty([1, len(index)]), columns=index)
    for f in all_features:
        for c in df.columns[1:]:
            ret[get_col_label(c, f.__name__)][0] = f(df[c])
    return ret


def calc_emg_wavelet_features(coeffs, label):
    print('  > Wavelet features (' + label + ')... ', end='', flush=True)
    now = time.time()
    f = coeffs.groupby('window').apply(calc_features)
    print(f'{time.time() - now : .3f}s')
    return f


def calc_emg_wavelets(df):
    print(' >  Discrete wavelet decompositions... ', end='', flush=True)
    now = time.time()

    cols = COL_EMG_CHANNELS
    samples_per_window = get_window_length(df)
    num_windows, window_offset = calc_window_params(
        df.shape[0], samples_per_window)

    wavelet = pywt.Wavelet('db1')
    cf_len1 = int(np.floor((samples_per_window + wavelet.dec_len - 1) / 2))
    cf_len2 = int(np.ceil(cf_len1 / 2))
    # lvl 2 approximation (LP), lvl 2 details (HP), lvl 1 details (HP)
    cA2 = np.empty([cf_len2 * num_windows, len(cols) + 1])
    cD2 = np.empty([cf_len2 * num_windows, len(cols) + 1])
    cD1 = np.empty([cf_len1 * num_windows, len(cols) + 1])

    # Window index for easier grouping
    cA2[:, 0] = get_window_indices(num_windows, cf_len2).astype(np.int16)
    cD2[:, 0] = get_window_indices(num_windows, cf_len2).astype(np.int16)
    cD1[:, 0] = get_window_indices(num_windows, cf_len1).astype(np.int16)

    for i in range(num_windows):
        for c_idx, c in enumerate(cols):
            coeff = pywt.wavedec(
                df[c][i * window_offset:i * window_offset + samples_per_window], wavelet, level=2)
            cA2[i * cf_len2:(i + 1) * cf_len2, c_idx + 1] = coeff[0]
            cD2[i * cf_len2:(i + 1) * cf_len2, c_idx + 1] = coeff[1]
            cD1[i * cf_len1:(i + 1) * cf_len1, c_idx + 1] = coeff[2]

    print(f'{time.time() - now : .3f}s')
    return [pd.DataFrame(arr, columns=['window'] + [c for c in cols]) for arr in (cA2, cD2, cD1)]


def calc_emg_features(df, outdir, basename):
    emg = df[COL_EMG_CHANNELS] / EMG_RES
    emg = pd.DataFrame(filter_butterworth(
        emg, 10, 500, 1 / get_dt(df)), columns=emg.columns)
    emg.to_csv(os.path.join(outdir, basename +
                            '_butterworth.csv'), index=False)
    
    coeffs = calc_emg_wavelets(df)

    for x, label in zip(coeffs, ('cA2', 'cD2', 'cD1')):
        x.to_csv(os.path.join(outdir, basename +
                              '_' + label + '.csv'), index=False)
        f = calc_emg_wavelet_features(x, label)
        f.to_csv(os.path.join(outdir, basename + '_' +
                              label + '_features.csv'), index=False)


def calc_imu_orientation_simple(df):
    ax, ay, az = (df[c] for c in COL_IMU_ACCELS)
    ret = pd.DataFrame(columns=['pitch', 'roll'])
    ret['roll'] = np.arctan2(ay, az)
    ret['pitch'] = np.arctan2(-ax, np.sqrt(np.square(ay) + np.square(az)))
    return ret


def calc_imu_orientation_kalman(df, kf=kalman.Kalman()):
    imu = df[COL_IMU_ACCELS + COL_IMU_GYROS]
    acc_idx = [imu.columns.get_loc(c) for c in COL_IMU_ACCELS]
    gyro_idx = [imu.columns.get_loc(c) for c in COL_IMU_GYROS]
    dt = get_dt(df)

    def do_kalman(row):
        kf.computeAndUpdateRollPitch(*row[acc_idx], *row[gyro_idx[:2]], dt)
        return kf.pitch, kf.roll

    ret = pd.DataFrame([do_kalman(row)
                        for row in imu.to_numpy()], columns=['pitch', 'roll'])
    ret = np.radians(ret)
    ret.columns.name = 'kalman'
    return ret


def calc_imu_orientation(df):
    # TODO the MPU-6050 has a motion processing unit on board, but using it takes quite some effort
    # See https://github.com/jrowberg/i2cdevlib/blob/master/Arduino/MPU6050/MPU6050_6Axis_MotionApps612.cpp
    print('  > Estimating IMU angles (' + IMU_METHOD + ')... ', end='', flush=True)
    now = time.time()

    if IMU_METHOD == 'kalman':
        ret = calc_imu_orientation_kalman(df)
    elif IMU_METHOD == 'simple':
        ret = calc_imu_orientation_simple(df)
    else:
        raise ValueError('Unknown method ' + IMU_METHOD)

    print(f'{time.time() - now : .3f}s')
    return ret


def calc_imu_orientation_windowed(df):
    # Calculate the IMU mean for every EMG window so that we have one target value per window
    samples_per_window = get_window_length(df)
    num_windows, window_offset = calc_window_params(
        df.shape[0], samples_per_window)
    owin = np.empty([num_windows, 2])

    print('  > Windowing IMU angles... ', end='', flush=True)
    now = time.time()

    for i in range(num_windows):
        owin[i] = np.mean(
            df[i * window_offset:i * window_offset + samples_per_window])

    print(f'{time.time() - now : .3f}s')
    owin = pd.DataFrame(owin, columns=['pitch_avg', 'roll_avg'])
    owin.index.name = 'window'
    return owin


def calc_imu_features(df, outdir, basename):
    o = calc_imu_orientation(df)
    o.to_csv(os.path.join(outdir, basename + '_orientation_' +
                          IMU_METHOD + '.csv'), index=False)

    owin = calc_imu_orientation_windowed(o)
    owin.to_csv(os.path.join(outdir, basename + '_orientation_' +
                             IMU_METHOD + '_windowed.csv'), index=True)


def process_recordings(recordings_dir, out_dir=None, do_emg=True, do_imu=True):
    if not out_dir:
        out_dir = os.path.join(recordings_dir, 'preprocessed')

    os.makedirs(out_dir, exist_ok=True)
    for emg_f in os.listdir(recordings_dir):
        if not emg_f.endswith('.csv'):
            continue

        print(emg_f)
        basename = os.path.splitext(emg_f)[0]
        df = pd.read_csv(os.path.join(recordings_dir, emg_f))

        now = time.time()
        if do_emg:
            calc_emg_features(df, out_dir, basename)
        if do_imu:
            calc_imu_features(df, out_dir, basename)
        print(f'done ({time.time() - now : .3f}s)\n')
