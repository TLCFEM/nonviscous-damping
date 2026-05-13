import numpy as np


def inverse(s, m):
    m = np.asarray(m, dtype=np.complex128)
    ones = np.ones_like(m)

    A = np.diag(np.asarray(s, dtype=np.complex128)) + np.outer(m, ones)

    ar, R = np.linalg.eig(A)
    al, L = np.linalg.eig(A.T)

    a = ar[idx := np.argsort(ar)]
    R = R[:, idx]
    L = L[:, np.argsort(al)]

    b = np.zeros_like(a)
    for k in range(b.size):
        b[k] = -(ones @ R[:, k]) * (L[:, k] @ m) / (L[:, k] @ R[:, k])

    return a, b


def random_sm(
    n,
    s_range=(10, 1000),
    m_range=(-1, 0),
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
    s, m = random_sm(500, complex_valued=True)
    rs, rm = inverse(*inverse(s, m))
    print("|s-rs|:", np.linalg.norm(s - rs))
    print("|m-rm|:", np.linalg.norm(m - rm))

    ra, rb = inverse([20, 10], [-3, -4])
    print("ra", ra)
    print("rb", rb)
