import numpy as np


class QRegister:
    def __init__(self, n_qbits, init):
        self._n = n_qbits
        assert len(init) == self._n

        self._data = np.zeros((2 ** self._n), dtype=np.complex64)
        self._data[int('0b' + init, 2)] = 1
