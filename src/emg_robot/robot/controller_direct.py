import sys
import numpy as np

from emg_robot.preprocessing import filter_butterworth, features
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
PITCH_WEIGHTS = np.array([1., -1., 0., 0., 0.])  # biceps, triceps
ROLL_WEIGHTS = np.array([0., 0., 1., -0.5, -0.5])  # pronator teres, brachioradialis, supinator
CHANNEL_AGGREGATION_FUNCTION = features.f_rms
PITCH_THRESHOLD = 0.1
ROLL_THRESHOLD = 0.1
ROBOT_IP = "192.168.2.12"
MAX_JOINT_CHANGE_RAD = 0.1
ROBOT_SPEED_F = 0.1


def main(i2c_addresses, robot_ip, channel_aggregation_func):
    emg = EMGReader(BUFFER_SIZE, i2c_addresses)
    robot = RobotController(robot_ip, ROBOT_SPEED_F, MAX_JOINT_CHANGE_RAD)

    while(True):
        emg.read()
        if not emg.has_full_window():
            continue

        values = emg.get_samples()

        # Preprocessing
        sampling_rate = emg.sampling_rate()
        if sampling_rate > 500:
            values = filter_butterworth(values, 0, 500, sampling_rate)

        ### Transform features into arm angles
        features = channel_aggregation_func(values)
        # TODO create a GUI to adjust the weights live
        # It's basically a very simple perceptron without hidden neurons
        pitch = np.sum(features * PITCH_WEIGHTS)
        roll = np.sum(features * ROLL_WEIGHTS)

        if abs(pitch) < PITCH_THRESHOLD:
            pitch = 0.
        if abs(roll) < ROLL_THRESHOLD:
            roll = 0.

        ### Move robot
        robot.move(pitch, roll, relative=True)

        # Keep part of this window and continue collecting
        emg.clear(WINDOW_OVERLAP)


if __name__ == '__main__':
    try:
        # TODO parse cmd line arguments?
        main(I2C_ADDRESSES, ROBOT_IP, CHANNEL_AGGREGATION_FUNCTION)
    except KeyboardInterrupt:
        print('Terminated by user')
    except Exception as e:
        sys.exit(e)
