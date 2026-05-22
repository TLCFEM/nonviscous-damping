import os
from pathlib import Path

from matplotlib import pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter
import numpy as np


matplotlib.rcParams.update({"font.size": 6})


def inverse(m, s):
    m = np.asarray(m, dtype=np.complex128)
    ones = np.ones_like(m)

    ar, R = np.linalg.eig(
        np.diag(np.asarray(s, dtype=np.complex128)) + np.outer(m, ones)
    )

    R = R[:, idx := np.argsort(ar)]

    return -(ones @ R) * np.linalg.solve(R, m), ar[idx]


def to_latex_table(m, s, rm, rs, digits=6):
    m = np.real_if_close(m)
    s = np.real_if_close(s)
    rm = np.real_if_close(rm)
    rs = np.real_if_close(rs)

    def fmt(v):
        if np.iscomplexobj(v) and abs(v.imag) > 1e-14:
            return f"{v.real:.{digits}e}{v.imag:+.{digits}e}i"
        return f"{float(np.real(v)):.{digits}e}"

    lines = []
    lines.append(r"\begin{tabular}{cccc}")
    lines.append(r"\toprule")
    lines.append(r"$m_j$ & $s_j$ & $m'_j$ & $s'_j$ \\")
    lines.append(r"\midrule")
    for mj, sj, rmj, rsj in zip(m, s, rm, rs):
        lines.append(f"{fmt(mj)} & {fmt(sj)} & {fmt(rmj)} & {fmt(rsj)} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def process(input_str: str, fn: str):
    input_list = [float(x) for x in input_str.split() if x != "-type0"]
    zeta = np.array(input_list[0::2])
    omega = np.array(input_list[1::2])
    s = omega
    m = -2 * zeta * omega

    s = s[idx := np.argsort(s)]
    m = m[idx]

    rm, rs = inverse(m, s)

    print(to_latex_table(m, s, rm, rs))

    fig = plt.figure(figsize=(6, 5))
    fig.add_subplot(311)

    log_s = np.log10(s)
    log_s = np.where(log_s >= 0, np.ceil(log_s), np.floor(log_s)).astype(int)

    x = np.logspace(log_s[0] - 1, log_s[-1] + 1, 500)
    dynamic = np.ones_like(x, dtype=np.complex128)
    for mj, sj in zip(m, s):
        dynamic += mj / (sj + 1j * x)
    plt.plot(x, dynamic.imag * 100, linestyle="dashed", label=r"$\zeta(m_j,s_j)$")

    dynamic_inv = np.ones_like(x, dtype=np.complex128)
    for mj, sj in zip(rm, rs):
        dynamic_inv += mj / (sj + 1j * x)
    dynamic_inv = 1 / dynamic_inv

    plt.plot(x, dynamic_inv.imag * 100, linestyle="dotted", label=r"$\zeta(m'_j,s'_j)$")

    plt.xscale("log")
    plt.yscale("log")
    # ax1 = plt.gca()
    # ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    # ax1.yaxis.set_minor_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    plt.xlabel(r"frequency $\omega$")
    plt.ylabel(r"$\dfrac{\text{loss stiffness}}{\text{static stiffness}}$ (%)")
    plt.legend()
    plt.grid(which="both", linestyle="--", linewidth=0.2)

    fig.add_subplot(312)

    plt.plot(x, dynamic.real, linestyle="dashed", label=r"$\zeta(m_j,s_j)$")
    plt.plot(x, dynamic_inv.real, linestyle="dotted", label=r"$\zeta(m'_j,s'_j)$")

    plt.xscale("log")
    plt.yscale("log")
    ax1 = plt.gca()
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}"))
    ax1.yaxis.set_minor_formatter(FuncFormatter(lambda y, _: f"{y:.1f}"))
    plt.xlabel(r"frequency $\omega$")
    plt.ylabel(r"$\dfrac{\text{storage stiffness}}{\text{static stiffness}}$ (1)")
    plt.legend()
    plt.grid(which="both", linestyle="--", linewidth=0.2)

    fig.add_subplot(313)

    t = np.linspace(0, 0.1, 500)
    kernel = np.zeros_like(t, dtype=np.complex128)
    for mj, sj in zip(m, s):
        kernel += mj * np.exp(-sj * t)
    plt.plot(t, np.abs(kernel), ls="dashed", label="$(m_j,s_j)$")

    kernel = np.zeros_like(t, dtype=np.complex128)
    for mj, sj in zip(rm, rs):
        kernel += mj * np.exp(-sj * t)
    plt.plot(t, np.abs(kernel), ls="dotted", label="$(m'_j,s'_j)$")

    plt.xlabel("time (s)")
    plt.ylabel("abs. kernel value $|g(t)|$")
    plt.xlim(0, 0.1)
    plt.yscale("log")
    plt.legend()
    plt.grid(which="both", linestyle="--", linewidth=0.2)
    plt.tight_layout(pad=0.01)
    plt.text(
        0.07,
        4,
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
