from quantum import QRegister, H, I, U


def is_constant(f, n):
    q = QRegister(n + 1, '0' * n + '1')
    q.apply(H ** (n + 1))
    q.apply(U(f, n))
    q.apply(H ** n @ I)

    return q.measure()[:~0] == '0' * n


def f1(x):
    return x


def f2(x):
    return 1


def f3(x, y):
    return x ^ y


def f4(x, y, z):
    return 0


print('f(x) = x is {}'.format('constant' if is_constant(f1, 1) else 'balansed'))
print('f(x) = 1 is {}'.format('constant' if is_constant(f2, 1) else 'balansed'))
print('f(x, y) = x ^ y is {}'.format('constant' if is_constant(f3, 2) else 'balansed'))
print('f(x, y, z) = 0 is {}'.format('constant' if is_constant(f4, 3) else 'balansed'))
