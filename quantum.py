from typing import Optional

import numpy as np
from itertools import product


class QGate:
    def __init__(self, matrix):
        """
        Initializes quantum gate from corresponding unitary complex matrix.

        :param matrix: unitary complex matrix in any form that can
        be used in the numpy array constructor
        """
        self._data = np.array(matrix, dtype=np.complex64)

        if len(self._data.shape) != 2:
            raise ValueError('input value must be 2-dimensional')

        if self._data.shape[0] != self._data.shape[1]:
            raise ValueError('input matrix must be square')

        size: float = np.log2(self._data.shape[0])

        if not size.is_integer():
            raise ValueError('input matrix size must be a power of two')

        self._size = int(size)

    @property
    def size(self) -> int:
        """
        Returns size of the quantum gate.

        :return: size of the quantum gate
        """

        return self._size

    @property
    def data(self) -> np.ndarray:
        """
        Returns complex matrix corresponding to the quantum gate.

        :return: complex matrix corresponding to the quantum gate
        """
        return self._data.copy()

    def __matmul__(self, other: 'QGate') -> 'QGate':
        """
        Composes two gates.

        :param other: other gate
        :return: composition of this gate and the provided gate as a new gate
        """
        return QGate(np.kron(self._data, other._data))

    def __pow__(self, n: int, modulo=None) -> 'QGate':
        """
        Composes gate with itself n times.

        :param n: number of compositions
        :param modulo: not used
        :return: this gate composed with self n times as a new gate
        """
        x = self._data.copy()

        for _ in range(n - 1):
            x = np.kron(x, self._data)

        return QGate(x)


class QRegister:
    def __init__(self, size: int, state: Optional[str] = None):
        """
        Creates quantum register. Size must be a positive integer and state
        must be a string of '0' and '1'.

        >>> QRegister(1, '0')

        >>> QRegister(5, '01011')

        If state is None, all zeroes is used.

        :param size: size of the quantum register
        :param state: state of the quantum register (default: None)
        """
        if not size > 0:
            raise ValueError('size must be positive')

        if state is None:
            state = '0' * size

        if size != len(state):
            raise ValueError('state length must equal to the register size')

        self._size = size
        self._data = np.zeros((2 ** self._size), dtype=np.complex64)
        self._data[int('0b' + state, 2)] = 1

    @property
    def size(self) -> int:
        """
        Returns size of the quantum register.

        :return: size of the quantum register
        """
        return self._size

    def measure(self) -> str:
        """
        Measures state of the quantum register.

        :return: state of the quantum register as a string of '0' and '1'
        """
        probabilities = np.real(self._data) ** 2 + np.imag(self._data) ** 2
        states = np.arange(2 ** self._size)
        measured = np.random.choice(states, size=1, p=probabilities)[0]

        return f'{measured:>0{self._size}b}'

    def apply(self, gate: QGate) -> None:
        """
        Applies quantum gate to the register in-place.

        :param gate: :class:`~quantum.QGate` object
        :return: None
        """
        if self.size != gate.size:
            raise ValueError('gate size must be equal to the register size')

        self._data = gate.data @ self._data


I = QGate([[1, 0], [0, 1]])
H = QGate(np.array([[1, 1], [1, -1]]) / np.sqrt(2))
X = QGate([[0, 1], [1, 0]])
Y = QGate([[0, -1j], [1j, 0]])
Z = QGate([[1, 0], [0, -1]])


def U(f, n) -> QGate:
    """
    Creates quantum gate from the binary function f with n arguments.

    :param f: binary function
    :param n: number of arguments
    :return: quantum gate corresponding to the function f
    """
    m = n + 1

    result = np.zeros((2 ** m, 2 ** m), dtype=np.complex64)

    def bin2int(bs):
        r = 0
        for i, b in enumerate(reversed(bs)):
            r += b * 2 ** i
        return r

    for xs in product({0, 1}, repeat=m):
        x = xs[:~0]
        y = xs[~0]

        z = y ^ f(*x)

        in_state = bin2int(xs)
        out_state = bin2int(list(x) + [z])
        result[in_state, out_state] = 1

    return QGate(result)
