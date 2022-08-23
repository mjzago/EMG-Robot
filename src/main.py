import os
from argparse import ArgumentParser


def do_recording(args):
    # from emg_robot.recording import
    # TODO
    raise NotImplementedError()


def do_preprocessing(args):
    from emg_robot.preprocess import process_recordings

    process_recordings(os.path.join(args.data_dir, 'recordings/'),
                       os.path.join(args.data_dir, 'preprocessed/'))


def do_training(args):
    from emg_robot.learn import load_data, save_model, train

    data = load_data(os.path.join(args.data_dir, 'preprocessed/'))
    model = train(data)
    model_file = save_model(model, args.data_dir)

    print(f'Saved model to {model_file}')


def do_simulate(args):
    # TODO
    raise NotImplementedError()


def do_fake_control(args):
    from emg_robot.control import start_gui
    from emg_robot.control.controller_direct_fake import DirectControllerFake
    from emg_robot.defaults import I2C_ADDRESSES, ROBOT_IP, EMG_CHANNEL_NAMES
    
    ctrl = DirectControllerFake(I2C_ADDRESSES, ROBOT_IP)
    start_gui(ctrl, EMG_CHANNEL_NAMES)


def do_control_direct(args):
    from emg_robot.control import start_gui
    from emg_robot.control.controller_direct import DirectController
    from emg_robot.defaults import I2C_ADDRESSES, ROBOT_IP, EMG_CHANNEL_NAMES
    
    ctrl = DirectController(I2C_ADDRESSES, ROBOT_IP)
    start_gui(ctrl, EMG_CHANNEL_NAMES)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-r", "--record", dest="do_recording",
                        help="Start recording samples from the EMG sensors until XXX")
    parser.add_argument("-p", "--preprocess", dest="do_preprocessing", action='store_true',
                        help="Preprocess a folder of previously recorded EMG data "
                             "(load data from data_dir/recordings/ and save data to data_dir/preprocessed/)")
    parser.add_argument("-t", "--train", dest="do_training", action='store_true',
                        help="Train an AI model on the preprocessed data "
                             "(load data from data_dir/preprocessed and save model to data_dir/)")
    parser.add_argument("-s", "--simulate", dest="do_simulate", action='store_true',
                        help="Use a trained AI model to classify live EMG data without a robot attached "
                             "(load model from data_dir/)")
    parser.add_argument("-f", "--fake", dest="do_fake_control", action='store_true',
                        help="For testing: use a fake controller (no EMG readings, direct control GUI)")
    parser.add_argument("-c", "--control", dest="do_control", action='store_true',
                        help="Use a trained AI model to classify live EMG data and control a robot "
                             "(load model from data_dir/)")
    # parser.add_argument("-i", "--ignore-features", dest="ignore_features", action="extend", nargs="+",
    #                     help="Do not use these features")
    parser.add_argument("data_dir",
                        help="The base folder to load from and store data in")

    args = parser.parse_args()
    # if args.ignore_features:
    #     args.ignore_features = ['f_' + f if not f.startswith('f_') else f for f in args.ignore_features]
    # else:
    #     args.ignore_features = []

    if args.do_recording:
        do_recording(args)

    elif args.do_preprocessing:
        do_preprocessing(args)

    elif args.do_training:
        do_training(args)

    elif args.do_simulate:
        do_simulate(args)

    elif args.do_fake_control:
        do_fake_control(args)

    elif args.do_control:
        do_control_direct(args)
