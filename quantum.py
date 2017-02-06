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

    def apply(self, gate):
        assert isinstance(gate, QGate)
        assert self._n == gate._n
        self._data = gate._data @ self._data


class QGate:
    def __init__(self, matrix):
        self._data = np.array(matrix, dtype=np.complex64)

        assert len(self._data.shape) == 2
        assert self._data.shape[0] == self._data.shape[1]

        self._n = np.log2(self._data.shape[0])

        assert self._n.is_integer()

        self._n = int(self._n)

    def __matmul__(self, other):
        return QGate(np.kron(self._data, other._data))

    def __pow__(self, n, modulo=None):
        x = self._data.copy()

        for _ in range(n - 1):
            x = np.kron(x, self._data)

        return QGate(x)
