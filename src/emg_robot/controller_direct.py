from cmath import tau
import sys
import time
import numpy as np
from torch import addr
import smbus 
from frankx import Robot, JointMotion
from utils import CircularBuffer
import features


emg_buffer_size = 50
max_change_rad = 0.1


A0 = 0x48
A1 = 0x49
A2 = 0x4A 
A3 = 0x4B
A4 = 0x4C 
A5 = 0x4D 
A6 = 0x4E
A7 = 0x4F 

# get the address from programm arguments 
robot_ip = sys.argv[1]
addresses = [locals()[arg] for arg in sys.argv[2:]]

buffer = CircularBuffer(emg_buffer_size, [len(addresses)])
bus = smbus.SMBus(1)
robot = Robot(robot_ip)
robot.set_dynamic_rel(0.1)


def read_emg():
    row = buffer.row()
    for idx,a in enumerate(addresses):
        data = bus.read_i2c_block_data(a, 0, 2)
        row[idx] = (data[0] << 8) | data[1]
    buffer.step()


def calc_emg_features(df):
    for f in features.all_features:
        df[f.__name__] = df.apply(f, axis=0)
    return df


def is_emg_active(df):
    pass


def update_joint(curr, new):
    if curr - new > max_change_rad:
        return curr - max_change_rad
    if new - curr > max_change_rad:
        return curr + max_change_rad
    return new 


def robot_move(phi, tau):
    state = robot.get_state()
    if any(dq > 0.1 for dq in state.dq):
        print('Warning: robot is currently moving!')

    j = state.q
    j[3] = update_joint(j[3], phi)  # elbow
    j[4] = update_joint(j[4], tau)  # forearm

    try:
        robot.recover_from_errors()
        robot.move(JointMotion(j))
    except Exception as e:
        print(str(e))


while True:



def emg_filter(data, emg_config):
    # TODO apply band filter, noise reduction, whitening, normalization, smoothing (moving average)...
    pass


def emg_classify(data, emg_config):
    joint_estimates = {
        'joints': None,            # array of joint estimates for EITHER the actual human's arm OR the resulting robot's arm
        'timestamp': time.time(),  # needed to calculate joint velocities
    }

    # TODO get joint estimates from classifier (e.g. neural network)
    return joint_estimates

def limit_joint_velocities(prev_joint_estimates, joint_estimates):
    # TODO avoid sudden accelerations due to noise or wrong classifications
    pass

def send_to_robot(joint_estimates, robot_config):
    # TODO send new joint space pose to robot (ROS)
    pass



if __name__ == '__main__':
    # Configuration and metadata of the EMG source
    emg_config = {
        'source': 'dataset',
        'derivatives': 16,
        'samples_per_second': 100,
        'moving_average_window': 50,
    }
    # Everything we need to connect to the robot and control it
    robot_config = {
        'ip': '192.168.2.1',
        'port': 3000,
    }

    init_emg(emg_config)
    init_robot(robot_config)

    data = np.zeros((emg_config['moving_average_window'], emg_config['derivatives']))
    prev_joint_estimates = None

    while True:
        try:
            # MOHAMMED, dein Part :)
            read_data(data, emg_config)
            emg_filter(data, emg_config)
            
            # Tendenziell mein Part!
            joint_estimates = emg_classify(data, emg_config)
            limit_joint_velocities(prev_joint_estimates, joint_estimates)
            send_to_robot(joint_estimates, robot_config)
        except KeyboardInterrupt:
            print('Terminated by user')
            break
        except Exception as e:
            sys.exit(e)

    sys.exit(0)
