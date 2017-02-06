from quantum import QRegister, H


def quantum_randbit():
    a = QRegister(1, '0')
    a.apply(H)

    return a.measure()


for i in range(32):
    print(quantum_randbit(), end='')
print()
