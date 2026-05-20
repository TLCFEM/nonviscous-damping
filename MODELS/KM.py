import numpy as np
import matplotlib.pyplot as plt


def inverse(m, s):
    m = np.asarray(m, dtype=np.complex128)
    ones = np.ones_like(m)

    ar, R = np.linalg.eig(
        np.diag(np.asarray(s, dtype=np.complex128)) + np.outer(m, ones)
    )

    R = R[:, idx := np.argsort(ar)]

    return -(ones @ R) * np.linalg.solve(R, m), ar[idx]


def inverse_highpass(m, s):
    m = np.asarray(m, dtype=np.complex128)
    s = np.asarray(s, dtype=np.complex128)
    ms = m * s
    ones = np.ones_like(m)

    D = 1.0 + np.sum(m)

    ar, R = np.linalg.eig(np.diag(s) - np.outer(ones, ms) / D)

    R = R[:, idx := np.argsort(ar)]
    rs = ar[idx]

    return -(ms @ R) * np.linalg.solve(R, ones) / (D**2 * rs), rs


def plot_response(m, s, ax, omega=None, w_min=None, w_max=None, reciprocal=False):
    s_scale = np.maximum(np.abs(s), 1e-12)
    if w_min is None:
        w_min = max(1e-3, 0.1 * np.min(s_scale))
    if w_max is None:
        w_max = 10.0 * np.max(s_scale)

    omega = np.logspace(np.log10(w_min), np.log10(w_max), 200)

    iw = 1j * omega[None, :]
    H = 1.0 + np.sum((m[:, None] * iw) / (s[:, None] + iw), axis=0)
    if reciprocal:
        H = 1.0 / H

    ax.semilogx(omega, np.abs(H))
    ax.set_ylabel(r"$|H(i\omega)|$")
    ax.grid(True, which="both", ls=":")

    return omega, H


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

    return m, s


if __name__ == "__main__":
    m, s = random_sm(500, complex_valued=True, seed=21621431)
    rm, rs = inverse_highpass(m, s)
    ax = plt.subplot(1, 1, 1)
    plot_response(m, s, ax)
    plot_response(rm, rs, ax, reciprocal=True)
    plt.tight_layout()
    plt.show()