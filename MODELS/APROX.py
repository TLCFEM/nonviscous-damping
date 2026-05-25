import os
from pathlib import Path

from matplotlib import pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter
import numpy as np


matplotlib.rcParams.update({"font.size": 6})


class DampingBase:
    def __init__(self, m, s, reciprocal=False):
        self.m = m
        self.s = s
        self.reciprocal = reciprocal

    def kernel(self, t):
        kernel = np.zeros_like(t, dtype=np.complex128)
        for mj, sj in zip(self.m, self.s):
            kernel += mj * np.exp(-sj * t)
        return kernel

    def pair(self):
        return np.real_if_close(self.m), np.real_if_close(self.s)

    def sample(self):
        log_s = np.log10(self.s.real)
        log_s = np.where(log_s >= 0, np.ceil(log_s), np.floor(log_s)).astype(int)

        return np.logspace(min(log_s), max(log_s), 200)


class Damping(DampingBase):
    def __init__(self, m, s, reciprocal=False):
        m_tmp = np.asarray(m, dtype=np.complex128)
        s_tmp = np.asarray(s, dtype=np.complex128)

        super().__init__(m_tmp[idx := np.argsort(s_tmp)], s_tmp[idx], reciprocal)

        self._ones = np.ones_like(self.m)

    def convert(self):
        poles, ev = np.linalg.eig(np.diag(self.s) + np.outer(self.m, self._ones))

        ev = ev[:, idx := np.argsort(poles)]

        return Damping(
            -self._ones @ ev * np.linalg.solve(ev, self.m),
            poles[idx],
            not self.reciprocal,
        )

    def amplification(self, omega):
        amplification = 1 + np.sum(self.m / (self.s + 1j * omega[:, None]), axis=1)
        return 1 / amplification if self.reciprocal else amplification

    @staticmethod
    def preprocess(input_str: str):
        input_list = [float(x) for x in input_str.split() if x != "-type0"]
        zeta = np.array(input_list[0::2])
        omega = np.array(input_list[1::2])

        return Damping(-2 * zeta * omega, omega)


class DampingC(DampingBase):
    def __init__(self, m, s, reciprocal=False):
        m_tmp = np.asarray(m, dtype=np.complex128)
        s_tmp = np.asarray(s, dtype=np.complex128)

        super().__init__(m_tmp[idx := np.argsort(s_tmp)], s_tmp[idx], reciprocal)

    def convert(self):
        scalar = 1 + np.sum(self.m)

        poles, ev = np.linalg.eig(np.diag(self.s) * scalar - np.outer(self.m, self.s))

        ev = ev[:, idx := np.argsort(poles)]
        poles = poles[idx]

        return DampingC(
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

    @staticmethod
    def preprocess(input_str: str):
        input_list = [float(x) for x in input_str.split() if x != "-type0"]
        zeta = np.array(input_list[0::2])
        omega = np.array(input_list[1::2])

        return DampingC(2 * zeta, omega)


def to_latex_table(system, system_inv, digits=4):
    m, s = system.pair()
    rm, rs = system_inv.pair()

    def fmt(v):
        if np.iscomplexobj(v) and abs(v.imag) > 1e-14:
            return f"{v.real:.{digits}e}{v.imag:+.{digits}e}i"
        return f"{float(np.real(v)):.{digits}e}"

    lines = []
    lines.append(r"\begin{tabular}{cccc}")
    lines.append(r"\toprule")
    lines.append(r"$m_j$ & $s_j$ & $m'_j$ & $s'_j$ \\\midrule")
    for mj, sj, rmj, rsj in zip(m, s, rm, rs):
        lines.append(f"{fmt(mj)} & {fmt(sj)} & {fmt(rmj)} & {fmt(rsj)} \\\\")
    lines.append(r"\midrule")
    lines.append(f"$1+\\sum{{}}m_j$ & {fmt(1 + sum(m))} & & \\\\\\bottomrule")
    lines.append(r"\end{tabular}")

    print("\n".join(lines))


def process(input_str: str, fn: str, *, t_end: float = 0.1, with_kernel: bool = True):
    system = DampingC.preprocess(input_str)
    system_inv = system.convert()

    to_latex_table(system, system_inv)

    dynamic = system.amplification(x := system.sample())
    dynamic_inv = system_inv.amplification(x)

    fig = plt.figure(figsize=(6, 4.5 if with_kernel else 3))

    fig.add_subplot(311 if with_kernel else 211)
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    # plt.gca().yaxis.set_minor_formatter(FuncFormatter(lambda y, _: f"{y:g}"))

    plt.plot(x, dynamic.imag * 100, linestyle="dashed", label=r"$\zeta(m_j,s_j)$")
    plt.plot(x, dynamic_inv.imag * 100, linestyle="dotted", label=r"$\zeta(m'_j,s'_j)$")

    plt.xscale("log")
    plt.xlabel(r"frequency $\omega$")
    plt.ylabel(r"$\dfrac{\text{loss stiffness}}{\text{static stiffness}}$ (%)")
    plt.legend()
    plt.grid(which="both", linestyle="--", linewidth=0.2)

    fig.add_subplot(312 if with_kernel else 212)
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}"))
    # plt.gca().yaxis.set_minor_formatter(FuncFormatter(lambda y, _: f"{y:.1f}"))

    plt.plot(x, dynamic.real, linestyle="dashed", label=r"$\zeta(m_j,s_j)$")
    plt.plot(x, dynamic_inv.real, linestyle="dotted", label=r"$\zeta(m'_j,s'_j)$")

    plt.xscale("log")
    plt.xlabel(r"frequency $\omega$")
    plt.ylabel(r"$\dfrac{\text{storage stiffness}}{\text{static stiffness}}$ (1)")
    plt.legend()
    plt.grid(which="both", linestyle="--", linewidth=0.2)

    if with_kernel:
        fig.add_subplot(313)

        plt.plot(
            t := np.linspace(0, t_end, 500),
            k := np.abs(system.kernel(t)),
            ls="dashed",
            label="$(m_j,s_j)$",
        )
        max_k = max(k)
        k = np.abs(system_inv.kernel(t))
        plt.plot(t, k, ls="dotted", label="$(m'_j,s'_j)$")
        max_k = max(max_k, max(k))

        plt.xlabel("time (s)")
        plt.ylabel("abs. kernel value $|g(t)|$")
        plt.xlim(0, t_end)
        # plt.yscale("log")
        # plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}"))
        # plt.gca().yaxis.set_minor_formatter(FuncFormatter(lambda y, _: f"{y:.1f}"))
        plt.legend()
        plt.grid(which="both", linestyle="--", linewidth=0.2)
        plt.tight_layout(pad=0.01)
        plt.text(
            t_end * 0.7,
            max_k * 0.8,
            r"kernel: $g(t)=\sum{}m_je^{-s_jt}$",
            va="center",
            ha="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9),
        )

    fig.savefig(fn)


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)

    process(
        "-type0 1.98700e-02 3.76642e-01 -type0 1.66460e-02 8.52878e+00 -type0 1.53240e-02 3.16297e+00 -type0 8.86700e-03 7.94340e-02 -type0 1.98710e-02 2.65643e+01 -type0 1.66450e-02 1.17300e+00 -type0 3.62070e-02 1.25893e+02 -type0 2.73430e-02 7.94790e-02",
        "../PIC/MASS.EQ.pdf",
    )

    process(
        "-type0 3.82120e-02 4.72235e-01 -type0 3.91790e-02 1.93519e+02 -type0 3.82670e-02 4.26195e+01 -type0 3.91800e-02 5.16800e-03 -type0 3.82150e-02 9.49720e+00 -type0 3.82130e-02 2.11771e+00 -type0 6.03970e-02 1.11375e+03 -type0 6.03980e-02 8.98000e-04 -type0 3.82140e-02 1.05304e-01 -type0 3.82670e-02 2.34660e-02",
        "../PIC/MASS.EQ.MORE.pdf",
        with_kernel=False,
    )

    process(
        "-type0 1.71494e-01 9.98000e-04 -type0 5.10470e-02 2.71442e+05 -type0 3.16270e-02 4.98400e-01 -type0 1.55273e-01 8.00000e-06 -type0 2.15000e-02 4.62431e+02 -type0 4.76030e-02 4.62450e+02 -type0 8.18500e-02 1.00031e+05 -type0 6.15410e-02 8.00000e-06 -type0 1.06310e-01 2.48000e-04 -type0 6.51340e-02 4.11900e-03 -type0 6.66930e-02 2.20460e-02 -type0 1.12819e-01 3.70000e-05 -type0 9.67750e-02 3.93390e+04 -type0 3.88550e-02 6.27600e-03 -type0 8.50900e-03 3.93076e+04 -type0 3.04700e-03 1.25888e+06 -type0 3.15970e-02 2.01656e+01 -type0 7.96230e-02 9.80000e-05 -type0 1.71000e-04 8.00000e-06 -type0 9.85570e-02 2.07026e+03 -type0 2.13940e-01 1.25892e+06 -type0 6.24320e-02 2.71413e+05 -type0 2.85830e-02 3.17157e+00 -type0 1.72577e-01 9.83336e+03",
        "../PIC/MASS.EQ.MUCHMORE.pdf",
        with_kernel=False,
    )