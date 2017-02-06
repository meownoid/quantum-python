import numpy as np


class QRegister:
    def __init__(self, n_qbits, init):
        self._n = n_qbits
        assert len(init) == self._n

        self._data = np.zeros((2 ** self._n), dtype=np.complex64)
        self._data[int('0b' + init, 2)] = 1

    def measure(self):
        probs = np.real(self._data) ** 2 + np.imag(self._data) ** 2
        states = np.arange(2 ** self._n)
        mstate = np.random.choice(states, size=1, p=probs)[0]
        return f'{mstate:>0{self._n}b}'
