import sys
import numpy as np

from emg_robot.defaults import I2C_ADDRESSES, ROBOT_IP
from emg_robot.preprocess import filter_butterworth, features
from .emg_reader import EMGReader
from .robot import RobotInterface



def act_linear(values):
    return values


def act_sigmoid(values):
    return 2. / (1. + np.exp(-values)) - 1.


class DirectController():
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
                 activation_func = act_linear,
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
        self.activation_func = activation_func

        self.emg = EMGReader(emg_buffer_size, i2c_addresses)
        self.robot = RobotInterface(
            robot_ip, robot_velocity_f, max_joint_change_rad)

        self.running = False

    def calc_features(self, values):
        # Preprocessing
        sampling_rate = self.emg.sampling_rate()
        if sampling_rate > 500:
            values = filter_butterworth(values, 0, 500, sampling_rate)

        # Transform features into arm angles
        features = self.channel_aggregation_func(values)
        # TODO create a GUI to adjust the weights live
        # It's basically a very simple perceptron without hidden neurons
        pitch = features * (features > self.pitch_thresholds) * self.pitch_weights
        roll = features * (features > self.roll_thresholds) * self.roll_weights

        pitch = self.activation_func(np.sum(pitch) * self.pitch_f)
        roll = self.activation_func(np.sum(roll) * self.roll_f)

        return pitch, roll

    def run_once(self):
        '''
        Once enough values have been collected, the values of the window are 
        reduced to a single value (using channel_aggregation_func) and then
        multiplied by the weights. Channels that are below their threshold 
        (before applying the weights) are set to zero. The resulting features
        are summed and multiplied by a scaling factor to finally calculate 
        the pitch and roll differentials for the robot.
        '''
        self.emg.read()

        if not self.emg.has_full_window(self.emg_window_length_s):
            return

        values = self.emg.get_samples()
        pitch, roll = self.calc_features(values)

        print(f" -> pitch={pitch}")
        print(f" -> roll={roll}")

        # Move robot
        self.robot.move(pitch, roll, relative=True)

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
        ctrl = DirectController(I2C_ADDRESSES, ROBOT_IP)
        ctrl.run()
    except KeyboardInterrupt:
        print('Terminated by user')
    except Exception as e:
        sys.exit(e)
