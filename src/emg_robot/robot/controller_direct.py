import sys
import time
import numpy as np
import pywt

import smbus
from frankx import Robot, JointMotion

from emg_robot.preprocessing import filter_butterworth, all_features


A0 = 0x48
A1 = 0x49
A2 = 0x4A
A3 = 0x4B
A4 = 0x4C
A5 = 0x4D
A6 = 0x4E
A7 = 0x4F


# TODO adjust as required
I2C_ADDRESSES = [A0, A1, A2, A4, A6]   # order matters (somewhat)
WINDOW_LENGTH = 0.1  # in seconds
WINDOW_OVERLAP = 0.5  # ratio
BUFFER_SIZE = 2000  # should comfortably fit number of samples per window
ROBOT_IP = "192.168.2.12"
MAX_JOINT_CHANGE_RAD = 0.1


class EMGBuffer():
    def __init__(self, shape) -> None:
        self.buffer = np.zeros(shape)
        self.pos = 0
        self.dt_avg = 0.
        self.m2 = 0.
        self.n = 0

    def append(self, values, dt):
        assert(values.shape == self.buffer.shape[1:])
        # Note that this won't keep a reference to values
        self.buffer[self.pos] = values
        self.pos += 1
        self.n += 1

        # See https://stackoverflow.com/a/3907612/2061551
        delta = dt - self.dt_avg
        self.dt_avg += delta / self.n
        self.m2 += np.sqrt(delta * (dt - self.dt_avg))  # use the updated value

    def values(self):
        return self.buffer[:self.pos]

    def dt_avg(self):
        return self.dt_avg

    def dt_var(self):
        return self.m2 / self.n

    def sampling_rate(self):
        return self.pos / self.dt_avg

    def is_window_full(self, window_length):
        '''
        Check if there are enough values in this buffer to fill a window
        of the given length in seconds.
        '''
        return self.pos >= window_length // self.dt_avg

    def clear(self, keep_ratio=0.0):
        b = self.buffer
        num_keep = min(b.shape[0], np.ceil(self.pos * keep_ratio))
        b[:num_keep] = b[self.pos - num_keep:]
        self.pos = num_keep
        # keep dt, m2 and n

    def reset(self):
        self.clear()
        self.dt = 0.
        self.m2 = 0.
        self.n = 0

    def __sizeof__(self) -> int:
        return self.pos


class EMGReader():
    def __init__(self, buffer_size, channels) -> None:
        self.channels = channels
        self.bus = smbus.SMBus(1)
        self.buffer = EMGBuffer([buffer_size, len(channels)])
        self.buffer_row = np.zeros([len(channels)])
        self.last_read = time.time()

    def read(self):
        r = self.buffer_row
        bus = self.bus

        for idx, a in enumerate(self.channels):
            data = bus.read_i2c_block_data(a, 0, 2)
            r[idx] = (data[0] << 8) | data[1]

        now = time.time()
        self.buffer.append(r, self.last_read - now)
        self.last_read = now

    def sampling_rate(self):
        return self.buffer.sampling_rate()

    def clear(self, keep_ratio=0.0):
        self.buffer.clear(keep_ratio)

    def has_full_window(self, window_length):
        return self.buffer.is_window_full(window_length)

    def get_samples(self):
        return self.buffer.values()


class RobotController():
    def __init__(self, ip, dynamic_limit_rel=0.1, joint_change_limit_rad=0.05) -> None:
        self.ip = ip
        self.robot = Robot(ip)
        # Percentage of robot's maximum velocity, acceleration and jerk
        self.robot.set_dynamic_rel(dynamic_limit_rel)
        self.joint_change_limit_rad = joint_change_limit_rad

    def limit_joint_motion(self, curr, new):
        if new < curr - self.joint_change_limit_rad:
            return curr - self.joint_change_limit_rad
        if new > curr + self.joint_change_limit_rad:
            return curr + self.joint_change_limit_rad
        return new

    def move(self, pitch, roll):
        state = self.robot.get_state()
        if any(dq > 0.1 for dq in state.dq):
            print('Warning: robot is currently moving!')

        j = state.q
        j[3] = self.limit_joint_motion(j[3], pitch)  # elbow
        j[4] = self.limit_joint_motion(j[4], roll)  # forearm

        try:
            self.robot.recover_from_errors()
            self.robot.move(JointMotion(j))
        except Exception as e:
            print(str(e))


def calc_features(vals):
    # Windowing is only necessary for preprocessing
    nf = len(all_features)
    ret = np.zeros([nf, vals.shape[1]])
    for idx, f in enumerate(all_features):
        ret[idx, :] = f(vals)
    return ret


def main(robot_ip, i2c_addresses):
    emg = EMGReader(BUFFER_SIZE, i2c_addresses)
    robot = RobotController(robot_ip)
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
        pitch, roll = apply_model(features)

        ### Move robot
        robot.move(pitch, roll)

        # Keep part of this window and continue collecting
        emg.clear(WINDOW_OVERLAP)


if __name__ == '__main__':
    try:
        main(ROBOT_IP, I2C_ADDRESSES)
    except KeyboardInterrupt:
        print('Terminated by user')
    except Exception as e:
        sys.exit(e)
