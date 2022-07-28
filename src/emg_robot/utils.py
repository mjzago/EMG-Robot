import numpy as np


class CircularBuffer(np.ndarray):
    def __init__(self, maxlength, dim=(1,)):
        super().__init__(np.zeros([maxlength] + list(dim)))
        self._pos = 0

    def append(self, x):
        self[self._pos] = x
        self.step()
        
    def row(self):
        return self[self._pos]

    def step(self):
        self._pos = (self._pos + 1) % len(self)

    def reset(self):
        self[:] = 0
        self._pos = 0

    def __sizeof__(self):
        return self.shape[0]
        
    def __repr__(self) -> str:
        return f'CircularBuffer(mean={self.mean()}, var={self.var()}, len={len(self)})'
