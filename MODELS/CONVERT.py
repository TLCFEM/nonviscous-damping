import numpy as np
import matplotlib.pyplot as plt


class Damping:
    def __init__(self, m, s, reciprocal=False):
        self.m = np.asarray(m, dtype=np.complex128)
        self.s = np.asarray(s, dtype=np.complex128)
        self.reciprocal = reciprocal

    def convert(self):
        scalar = 1 + np.sum(self.m)

        poles, ev = np.linalg.eig(np.diag(self.s) * scalar - np.outer(self.m, self.s))

        ev = ev[:, idx := np.argsort(poles)]
        poles = poles[idx]

        return Damping(
            self.s @ ev * np.linalg.solve(ev, self.m) / (-scalar * poles),
            poles / scalar,
            not self.reciprocal,
        )

    def amplification(self, omega):
        iw = 1j * omega[None, :]
        amplification = 1 + np.sum(
            (self.m[:, None] * iw) / (self.s[:, None] + iw), axis=0
        )
        return 1 / amplification if self.reciprocal else amplification

    def plot(self, omega):
        plt.semilogx(
            omega,
            np.abs(amplification := self.amplification(omega)),
        )
        return amplification


def randomize(n, m_range, s_range, complex_valued=False, seed=None):
    rng = np.random.default_rng(seed)

    m_real = rng.uniform(*m_range, size=n)
    s_real = 10 ** rng.uniform(*s_range, size=n)
    s_real.sort()

    if complex_valued:
        s = s_real + 0.1j * 10 ** rng.uniform(*s_range, size=n)
        m = m_real + 0.1j * rng.uniform(*m_range, size=n)
    else:
        s = s_real.astype(np.complex128)
        m = m_real.astype(np.complex128)

    return m, s


if __name__ == "__main__":
    m_range = (0, 0.1)
    s_range = (1, 3)
    system = Damping(*randomize(12, m_range, s_range, False, 2162431))

    system.plot(omega := np.logspace(s_range[0] - 2, s_range[1] + 2, 400))
    system.convert().plot(omega)

    plt.ylabel(r"$|H(i\omega)|$")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.show()
