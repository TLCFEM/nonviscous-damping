import numpy as np


def map(m, s):
    """
    Convert damping coefficients from ``(m, s)`` to an equivalent transformed pair for UDD and UDA.
    """
    m = np.asarray(m, dtype=np.complex128)
    s = np.asarray(s, dtype=np.complex128)
    ms = m / s

    scalar = 1 - np.sum(ms)

    poles, ev = np.linalg.eig(np.diag(s) * scalar + np.outer(ms, s))

    ev = ev[:, idx := np.argsort(poles)]
    poles = poles[idx]

    ms = s @ ev * np.linalg.solve(ev, ms) / (-scalar * poles)
    s = poles / scalar

    return ms * s, s


if __name__ == "__main__":
    print(map([-2], [10]))
