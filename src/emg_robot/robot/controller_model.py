import sys
import numpy as np
import pywt

from emg_robot.defaults import I2C_ADDRESSES, ROBOT_IP
from emg_robot.preprocessing import filter_butterworth, all_features
from emg_robot.ai import load_model
from .emg_reader import EMGReader
from .robot import RobotInterface


# TODO 
AI_MODEL_PATH = "path/to/model"  # TODO


def calc_features(vals):
    # Windowing is only necessary for preprocessing
    nf = len(all_features)
    ret = np.zeros([nf, vals.shape[1]])
    for idx, f in enumerate(all_features):
        ret[idx, :] = f(vals)
    return ret


class ModelController():
    def __init__(self, 
                 i2c_addresses, 
                 ai_model_path, 
                 robot_ip,
                 emg_buffer_size = 2000,
                 emg_window_length_s = 0.1,
                 emg_window_overlap = 0.5,
                 robot_velocity_f = 0.05,
                 max_joint_change_rad = 0.1):
        self.i2c_addresses = i2c_addresses
        self.ai_model_path = ai_model_path
        self.robot_ip = robot_ip
        self.emg_window_length_s = emg_window_length_s
        self.emg_window_overlap = emg_window_overlap

        self.emg = EMGReader(emg_buffer_size, i2c_addresses)
        self.model = load_model(ai_model_path)
        self.robot = RobotInterface(robot_ip, robot_velocity_f, max_joint_change_rad)
        self.wavelet = pywt.Wavelet('db1')

        self.running = False

    def run_once(self):
        self.emg.read()

        if not self.emg.has_full_window(self.emg_window_length_s):
            return

        values = self.emg.get_samples()

        # Preprocessing
        sampling_rate = self.emg.sampling_rate()
        if sampling_rate > 500:
            values = filter_butterworth(values, 0, 500, sampling_rate)

        cA2, cD2, cD1 = pywt.wavedec(values, self.wavelet, level=2, axis=0)
        fA2 = calc_features(cA2)
        fD2 = calc_features(cD2)
        fD1 = calc_features(cD1)

        ### Transform features into arm angles
        # Feature vector should be of the shape [num_features * num_channels * 3]
        # NOTE: order must be the same as when training the model!
        features = np.stack([fA2, fD1, fD2], axis=1)
        pitch, roll = self.model(features)

        ### Move robot
        self.robot.move(pitch, roll, relative=False)

        # Keep part of this window and continue collecting
        self.emg.clear(self.emg_window_overlap)

    def run(self):
        self.running = True
        while (self.running):
            self.run_once()

    def stop(self):
        self.running = False


if __name__ == '__main__':
    try:
        # TODO parse cmd line arguments?
        ctrl = ModelController(I2C_ADDRESSES, AI_MODEL_PATH, ROBOT_IP)
        ctrl.run()
    except KeyboardInterrupt:
        print('Terminated by user')
    except Exception as e:
        sys.exit(e)
