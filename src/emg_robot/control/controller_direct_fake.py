import time
import numpy as np

from emg_robot.defaults import I2C_ADDRESSES, ROBOT_IP
from emg_robot.preprocess import features
from .controller_direct import DirectController


# TODO create a base class and let this and the real direct controller inherit from it
class DirectControllerFake(DirectController):
    '''
    Just for testing as it avoids some imports that may fail (e.g. smbus)
    '''
    def __init__(self,
                 i2c_addresses,
                 robot_ip,
                 emg_buffer_size = 2000,
                 emg_window_length_s = 0.1,
                 emg_window_overlap = 0.5,
                 pitch_weights = (1., -1., 0., 0., 0.),
                 pitch_thresholds = (0.1, 0.1, 0., 0., 0.),
                 pitch_f = 1.,
                 roll_weights = (0., 0., 1., -0.5, -0.5),
                 roll_thresholds = (0., 0., 0.1, 0.1, 0.1),
                 roll_f = 1.,
                 channel_aggregation_func = features.f_rms,
                 activation_func = lambda x: x,
                 robot_velocity_f = 0.05,
                 max_joint_change_rad = 0.1):
        self.i2c_addresses = i2c_addresses
        self.robot_ip = robot_ip
        self.emg_window_length_s = emg_window_length_s
        self.emg_window_overlap = emg_window_overlap
        self.pitch_weights = np.array(pitch_weights)
        self.pitch_thresholds = np.array(pitch_thresholds)
        self.pitch_f = pitch_f
        self.roll_weights = np.array(roll_weights)
        self.roll_thresholds = np.array(roll_thresholds)
        self.roll_f = roll_f
        self.channel_aggregation_func = channel_aggregation_func

        self.running = False

    def calc_features(self, values):
        # Transform features into arm angles
        features = self.channel_aggregation_func(values)

        # It's basically a very simple perceptron without hidden neurons
        pitch = features * (features > self.pitch_thresholds) * self.pitch_weights
        roll = features * (features > self.roll_thresholds) * self.roll_weights

        pitch = np.sum(pitch) * self.pitch_f
        roll = np.sum(roll) * self.roll_f

        return pitch, roll

    def run_once(self):
        values = np.ones((10, len(self.i2c_addresses))) * 0.5
        pitch, roll = self.calc_features(values)
        print(f" -> pitch={pitch}")
        print(f" -> roll={roll}")
        time.sleep(0.1)

    def run(self):
        self.running = True
        while (self.running):
            self.run_once()

    def emg_activity(self):
        return np.random.random([len(self.pitch_weights)])

    def stop(self):
        self.running = False
