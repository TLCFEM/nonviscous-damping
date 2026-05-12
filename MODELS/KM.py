import numpy as np


def inverse_poles_residues(s, m, tol=1e-12):
    s = np.asarray(s, dtype=np.complex128)
    m = np.asarray(m, dtype=np.complex128)

    n = len(s)

    ones = np.ones(n, dtype=np.complex128)
    b = np.zeros(n, dtype=np.complex128)

    A = np.diag(s) + np.outer(m, ones)

    a, R = np.linalg.eig(A)

    a_left, L = np.linalg.eig(A.T)

    used = np.zeros(n, dtype=bool)

    for k in range(n):
        idx = None

        for j in range(n):
            if not used[j] and abs(a_left[j] - a[k]) < tol:
                idx = j
                used[j] = True
                break

        if idx is None:
            raise ValueError("Could not match left/right eigenvalues")

        b[k] = -(ones @ R[:, k]) * (L[:, idx] @ m) / (L[:, idx] @ R[:, k])

    return a, b


def random_sm(
    n,
    s_range=(0.1, 10.0),
    m_range=(-1.0, 1.0),
    complex_valued=False,
    seed=None,
):
    rng = np.random.default_rng(seed)

    s_real = rng.uniform(s_range[0], s_range[1], size=n)

    s_real.sort()

    for k in range(1, n):
        if abs(s_real[k] - s_real[k - 1]) < 1e-3:
            s_real[k] += 1e-2

    m_real = rng.uniform(m_range[0], m_range[1], size=n)

    if complex_valued:
        s_imag = rng.uniform(*s_range, size=n)
        m_imag = rng.uniform(*m_range, size=n)

        s = s_real + 1j * s_imag
        m = m_real + 1j * m_imag

    else:
        s = s_real.astype(np.complex128)
        m = m_real.astype(np.complex128)

    return s, m


if __name__ == "__main__":
    s, m = random_sm(5, complex_valued=False, seed=42)
    a, b = inverse_poles_residues(s, m)
    print("s:", s)
    print("m:", m)
    print("a:", a)
    print("b:", b)
    s, m = inverse_poles_residues(a, b)
    print("Recovered s:", s)
    print("Recovered m:", m)
