from emg_robot.ai import load_data, train, save_model


if __name__ == '__main__':
    data_dir = '/home/dfki.uni-bremen.de/ndahn/devel/EMG-Robot/samples/'
    
    data, gt = load_data(data_dir)
    model = train(data, gt)
    model_file = save_model(model, data_dir)

    print(f'Saved model to {model_file}')
