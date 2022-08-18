import time
import numpy as np


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

    def is_window_full(self, window_length_s):
        '''
        Check if there are enough values in this buffer to fill a window
        of the given length in seconds.
        '''
        return self.pos >= window_length_s // self.dt_avg

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
        import smbus
        
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

    def has_full_window(self, window_length_s):
        return self.buffer.is_window_full(window_length_s)

    def get_samples(self):
        return self.buffer.values()