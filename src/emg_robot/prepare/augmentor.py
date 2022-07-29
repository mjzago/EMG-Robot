import os
import time
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from imusensor.filters import kalman
import pywt
from emg_robot.prepare.features import all_features


EMG_RES = 2**12  # 12bit ADC
# TODO adjust
WINDOW_LENGTH = 150  # 75 = 50ms at 1500Hz sampling rate
WINDOW_OVERLAP = WINDOW_LENGTH / 2


def normalize(df):
    min = df.min(axis=0)
    return (df - min) / (df.max(axis=0) - min)


def filter_butterworth(data, lowcut, highcut, fs, order=5):
    # See https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    print(f' >  Butterworth filter ({lowcut}, {highcut})... ', end='', flush=True)
    now = time.time()
    y = lfilter(b, a, data, axis=1)
    print(f'{time.time() - now : .3f}s')
    
    return y


def calc_features(df):
    index = [c + '_' + f.__name__ for c in df.columns[1:] for f in all_features]
    ret = pd.DataFrame(np.empty([1, len(index)]), columns=index)
    for f in all_features:
        for c in df.columns[1:]:
            ret[c + '_' + f.__name__.split('_', 1)[1]][0] = f(df[c])
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

    cols = df.columns
    window_offset = int(WINDOW_LENGTH - WINDOW_OVERLAP)
    # Make sure that the final window can be long enough. Trailing values will be ignored
    num_windows = int(df.shape[0] // window_offset - WINDOW_LENGTH / window_offset + 1)

    wavelet = pywt.Wavelet('db1')
    cf_len1 = int(np.floor((WINDOW_LENGTH + wavelet.dec_len - 1) / 2))
    cf_len2 = int(np.ceil(cf_len1 / 2))
    # lvl 2 approximation (LP), lvl 2 details (HP), lvl 1 details (HP)
    cA2 = np.empty([cf_len2 * num_windows, len(cols) + 1])
    cD2 = np.empty([cf_len2 * num_windows, len(cols) + 1])
    cD1 = np.empty([cf_len1 * num_windows, len(cols) + 1])

    # Window index for easier grouping
    cA2[:,0] = np.repeat(np.arange(num_windows), cf_len2).astype(np.int16)
    cD2[:,0] = np.repeat(np.arange(num_windows), cf_len2).astype(np.int16)
    cD1[:,0] = np.repeat(np.arange(num_windows), cf_len1).astype(np.int16)

    for i in range(num_windows):
        for c_idx,c in enumerate(cols):
            coeff = pywt.wavedec(df[c][i * window_offset:i * window_offset + WINDOW_LENGTH], wavelet, level=2)
            cA2[i * cf_len2:(i + 1) * cf_len2, c_idx + 1] = coeff[0]
            cD2[i * cf_len2:(i + 1) * cf_len2, c_idx + 1] = coeff[1]
            cD1[i * cf_len1:(i + 1) * cf_len1, c_idx + 1] = coeff[2]

    print(f'{time.time() - now : .3f}s')
    return [pd.DataFrame(arr, columns=['window'] + [c for c in cols]) for arr in (cA2, cD2, cD1)]


# TODO dt
def calc_emg_features(df, outdir, basename, cols=['emg_0', 'emg_1', 'emg_2', 'emg_3', 'emg_4'], dt=1496.0):
    if not isinstance(dt, float):
        dt = df[dt].mean()

    emg = df[cols] / EMG_RES
    emg = pd.DataFrame(filter_butterworth(emg, 10, 500, dt), columns=emg.columns)
    emg.to_csv(os.path.join(outdir, basename + '_butterworth.csv'), index=False, sep=';')
    coeffs = calc_emg_wavelets(emg)

    for x, label in zip(coeffs, ('cA2', 'cD2', 'cD1')):
        x.to_csv(os.path.join(outdir, basename + '_' + label + '.csv'), index=False, sep=';')
        f = calc_emg_wavelet_features(x, label)
        f.to_csv(os.path.join(outdir, basename + '_' + label + '_features.csv'), index=False, sep=';')




def calc_imu_orientation_simple(df, acc_c):
    ax, ay, az = (df[c] for c in acc_c)
    ret = pd.DataFrame(columns=['pitch', 'roll'])
    ret['roll'] = np.arctan2(ay, az)
    ret['pitch'] = np.arctan2(-ax, np.sqrt(np.square(ay) + np.square(az)))
    return ret


def calc_imu_orientation_kalman(df, acc_c, gyro_c, dt, kf = kalman.Kalman()):
    imu = df[acc_c + gyro_c]
    acc_idx = [imu.columns.get_loc(c) for c in acc_c]
    gyro_idx = [imu.columns.get_loc(c) for c in gyro_c]
    if not isinstance(dt, float):
        dt = df[dt].mean()

    def do_kalman(row):
        kf.computeAndUpdateRollPitch(*row[acc_idx], *row[gyro_idx[:2]], dt)
        return kf.pitch, kf.roll

    ret = pd.DataFrame([do_kalman(row) for row in imu.to_numpy()], columns=['pitch', 'roll'])
    ret = np.radians(ret)
    ret.columns.name = 'kalman'
    return ret


def calc_imu_orientation(df, method='kalman', acc_c=['acc_x', 'acc_y', 'acc_z'], gyro_c=['gyro_x', 'gyro_y', 'gyro_z'], dt='dt'):
    # TODO the MPU-6050 has a motion processing unit on board, but using it takes quite some effort
    # See https://github.com/jrowberg/i2cdevlib/blob/master/Arduino/MPU6050/MPU6050_6Axis_MotionApps612.cpp
    print('  > Estimating IMU angles (' + method + ')... ', end='', flush=True)
    now = time.time()

    if method == 'kalman':
        ret = calc_imu_orientation_kalman(df, acc_c, gyro_c, dt)
    elif method == 'simple':
        ret = calc_imu_orientation_simple(df, acc_c)
    else:
        raise ValueError('Unknown method ' + method)

    print(f'{time.time() - now : .3f}s')
    return ret


def calc_imu_orientation_windowed(df):
    # Calculate the IMU mean for every EMG window so that we have one target value per window
    window_offset = int(WINDOW_LENGTH - WINDOW_OVERLAP)
    num_windows = int(df.shape[0] // window_offset - WINDOW_LENGTH / window_offset + 1)
    owin = np.empty([num_windows, 2])

    print('  > Windowing IMU angles... ', end='', flush=True)
    now = time.time()

    for i in range(num_windows):
        owin[i] = np.mean(df[i * window_offset:i * window_offset + WINDOW_LENGTH])

    print(f'{time.time() - now : .3f}s')
    owin = pd.DataFrame(owin, columns=['win_pitch', 'win_roll'])
    owin.index.name = 'window'
    return owin


# TODO dt
def calc_imu_features(df, outdir, basename, method='simple', dt=1/1496):
    o = calc_imu_orientation(df, method, dt=dt)
    o.to_csv(os.path.join(outdir, basename + '_orientation_' + method + '.csv'), index=False, sep=';')

    owin = calc_imu_orientation_windowed(o)
    owin.to_csv(os.path.join(outdir, basename + '_orientation_' + method + '_windowed.csv'), index=True, sep=';')


def process_recordings(recordings_dir, out_dir=None):
    if not out_dir:
        out_dir = os.path.join(recordings_dir, 'features')
        os.makedirs(out_dir, exist_ok=True)

    for emg_f in os.listdir(recordings_dir):
        if not emg_f.endswith('.csv'):
            continue

        print(emg_f)
        basename = os.path.splitext(emg_f)[0]
        df = pd.read_csv(os.path.join(recordings_dir, emg_f))

        now = time.time()
        calc_emg_features(df, out_dir, basename)
        calc_imu_features(df, out_dir, basename)
        print(f'done ({time.time() - now : .3f}s)\n')


if __name__ == '__main__':
    process_recordings('recordings/test/')