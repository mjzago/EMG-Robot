import sys
import time
import numpy as np


def init_emg(emg_config):
    # TODO load and initialize neural network or whatever classifier we'll be using
    if (emg_config.source == 'emg'):
        # TODO initialize EMG sensors. This is probably also where we would run a calibration or even 
        #      additional training for our classifier
        pass
    elif (emg_config.source == 'dataset'):
        # TODO load dataset
        pass


def read_data(data, emg_config):
    # TODO read next sample(s) into data and remove old ones
    if (emg_config.source == 'emg'):
        # TODO return samples collected since last call (might have to collect in a separate thread?)
        pass
    elif (emg_config.source == 'dataset'):
        # TODO 
        pass


def emg_filter(data, emg_config):
    # TODO apply band filter, noise reduction, whitening, normalization, smoothing (moving average)...
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
