# EMG-Robot
This code was implemented for the thesis of Mohammed Fakih and used for recording, preprocessing and live-processing data from several Myoware EMG-sensors chained together through I2C. The idea was to train a neural network to estimate arm positions from EMG signals and use these estimates to control a robot arm. Due to time limitations, this part was implemented but not really tested or finetuned. Instead, a simpler approach where weights of a tiny neural network (input->output, no hidden layers) were set by hand was implemented. 

The approach was largely inspired by Azhiri et Al. 2021 - Real-Time EMG Signal Classification via Recurrent Neural Networks [https://doi.org/10.1109/BIBM52615.2021.9669872]. Accordingly, the preprocessing calculates a db1 2-level wavelet transformation for each EMG channel and then calculates various features on windows sliding over the resulting coefficients. Due to the problems mentioned above, whether this approach is reasonable could not be answered. 

If you are interested in this work, you can find a dataset recorded for this thesis at (EMG-Robot-Dataset)[https://github.com/ndahn/EMG-Robot-Dataset].
