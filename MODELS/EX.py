#!/usr/bin/env python3

import numpy as np


def map(m, s):
    """
    Transform damping coefficients between UDD and UDA.
    """
    m, s = np.asarray(m, np.complex128), np.asarray(s, np.complex128)

    scalar = 1 - np.sum(ms := m / s)
    poles, ev = np.linalg.eig(np.diag(s) * scalar + np.outer(ms, s))

    ev = ev[:, idx := np.argsort(poles)]
    poles = poles[idx]

    ms = s @ ev * np.linalg.solve(ev, ms) / (-scalar * poles)
    s = poles / scalar

    return ms * s, s


if __name__ == "__main__":
    print(map([-2], [10]))
