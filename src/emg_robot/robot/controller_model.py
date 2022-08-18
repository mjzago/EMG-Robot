import sys
import numpy as np
import pywt

from emg_robot.preprocessing import filter_butterworth, all_features
from emg_robot.ai import load_model
from .emg_reader import EMGReader
from .robot import RobotController


A0 = 0x48
A1 = 0x49
A2 = 0x4A
A3 = 0x4B
A4 = 0x4C
A5 = 0x4D
A6 = 0x4E
A7 = 0x4F


# TODO adjust as required
I2C_ADDRESSES = [A4, A5, A2, A6, A0]   # order matters (somewhat)
WINDOW_LENGTH = 0.1  # in seconds
WINDOW_OVERLAP = 0.5  # ratio
BUFFER_SIZE = 2000  # should comfortably fit number of samples per window
AI_MODEL_PATH = "path/to/model"  # TODO
ROBOT_IP = "192.168.2.12"
MAX_JOINT_CHANGE_RAD = 0.1
ROBOT_SPEED_F = 0.1


def calc_features(vals):
    # Windowing is only necessary for preprocessing
    nf = len(all_features)
    ret = np.zeros([nf, vals.shape[1]])
    for idx, f in enumerate(all_features):
        ret[idx, :] = f(vals)
    return ret


def main(i2c_addresses, ai_model_path, robot_ip):
    emg = EMGReader(BUFFER_SIZE, i2c_addresses)
    model = load_model(ai_model_path)
    robot = RobotController(robot_ip, ROBOT_SPEED_F, MAX_JOINT_CHANGE_RAD)
    wavelet = pywt.Wavelet('db1')

    while(True):
        emg.read()
        if not emg.has_full_window():
            continue

        values = emg.get_samples()

        # Preprocessing
        sampling_rate = emg.sampling_rate()
        if sampling_rate > 500:
            values = filter_butterworth(values, 0, 500, sampling_rate)

        cA2, cD2, cD1 = pywt.wavedec(values, wavelet, level=2, axis=0)
        fA2 = calc_features(cA2)
        fD2 = calc_features(cD2)
        fD1 = calc_features(cD1)

        ### Transform features into arm angles
        # Feature vector should be of the shape [num_features * num_channels * 3]
        # NOTE: order must be the same as when training the model!
        features = np.stack([fA2, fD1, fD2], axis=1)
        pitch, roll = model(features)

        ### Move robot
        robot.move(pitch, roll, relative=False)

        # Keep part of this window and continue collecting
        emg.clear(WINDOW_OVERLAP)


if __name__ == '__main__':
    try:
        # TODO parse cmd line arguments?
        main(I2C_ADDRESSES, AI_MODEL_PATH, ROBOT_IP)
    except KeyboardInterrupt:
        print('Terminated by user')
    except Exception as e:
        sys.exit(e)
