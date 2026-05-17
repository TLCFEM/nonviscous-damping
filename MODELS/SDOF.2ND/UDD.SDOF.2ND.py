import os
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from numpy import exp
from scipy.stats import linregress
import dataclasses
from itertools import cycle


def get_line_style():
    ls_tuple = [
        ("solid", (0, ())),
        ("loosely dotted", (0, (1, 4))),
        ("dotted", (0, (1, 2))),
        ("densely dotted", (0, (1, 1))),
        ("loosely dashed", (0, (5, 4))),
        ("dashed", (0, (5, 2))),
        ("densely dashed", (0, (5, 1))),
        ("loosely dashdotted", (0, (3, 4, 1, 4))),
        ("dashdotted", (0, (3, 2, 1, 2))),
        ("densely dashdotted", (0, (3, 1, 1, 1))),
        ("loosely dashdotdotted", (0, (3, 4, 1, 4, 1, 4))),
        ("dashdotdotted", (0, (3, 2, 1, 2, 1, 2))),
        ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
    ]

    for v in cycle(ls_tuple):
        yield v[1]


LS = get_line_style()


@dataclasses.dataclass
class Response:
    time: np.ndarray
    displacement: np.ndarray
    error: np.ndarray = None


u0 = 1
v0 = 0

matplotlib.rcParams.update({"font.size": 6})


def system(omega, s, m):
    roots = np.roots([1, 0, omega**2 - s**2, m * omega**2, -((s * omega) ** 2)])
    print("roots", roots)

    r1, r2, r3, r4 = roots

    coef = np.ones([4, 4], dtype=complex)
    coef[1, 0] = -r2 - r3 - r4
    coef[2, 0] = r2 * r3 + r2 * r4 + r3 * r4
    coef[3, 0] = -r2 * r3 * r4

    coef[1, 1] = -r1 - r3 - r4
    coef[2, 1] = r1 * r3 + r1 * r4 + r3 * r4
    coef[3, 1] = -r1 * r3 * r4

    coef[1, 2] = -r1 - r2 - r4
    coef[2, 2] = r1 * r2 + r1 * r4 + r2 * r4
    coef[3, 2] = -r1 * r2 * r4

    coef[1, 3] = -r1 - r2 - r3
    coef[2, 3] = r1 * r2 + r1 * r3 + r2 * r3
    coef[3, 3] = -r1 * r2 * r3

    w = np.linalg.solve(
        coef,
        np.array([u0, v0, -u0 * (s**2), -v0 * s**2 + m * u0 * omega**2]),
    )

    def _f(_t):
        return np.dot(w, exp(roots * _t)).real

    return _f


def analytical(para):
    vibrator = system(*para)
    t = 20
    dt = 0.01
    x = np.linspace(0, t, int(t / dt) + 1)
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = vibrator(x[i])

    plt.plot(x, y, "r", label="analytical", linewidth=0.5)

    return vibrator


def numerical(vibrator, pick):
    name = "R1-U"
    with h5py.File(f"{name}-{str(pick)}.h5", "r") as f:
        data = f[f"/{name}/{name}2"]
        time = data[:, 0]
        displacement = data[:, 1]

    ref = np.zeros(len(time))
    for i in range(len(time)):
        ref[i] = vibrator(time[i])

    error = ref - displacement

    return Response(time, displacement, error)

def run():
    os.chdir(Path(__file__).parent)

    fig = plt.figure(figsize=(6, 3.5))
    fig.add_subplot(211)

    results = {}

    sdof = analytical([10, 10, -2])

    fig.savefig("../../PIC/UDD.SDOF.2ND.pdf")

    return
    # results["0.0001"] = numerical(sdof, 0.0001)
    # results["0.0002"] = numerical(sdof, 0.0002)
    # results["0.0005"] = numerical(sdof, 0.0005)
    # results["0.001"] = numerical(sdof, 0.001)
    # results["0.002"] = numerical(sdof, 0.002)
    # results["0.005"] = numerical(sdof, 0.005)
    # results["0.01"] = numerical(sdof, 0.01)

    for key, value in results.items():
        plt.plot(
            value.time,
            value.displacement,
            label=f"$\\Delta{{}}t=${float(key):1.0E}",
            linestyle=next(LS),
            linewidth=0.8,
        )

    plt.legend(loc="lower right", ncol=2)
    plt.xlabel("time (s)")
    plt.ylabel("displacement")
    plt.grid(which="both", linestyle="--", linewidth=0.2)
    plt.xlim(0, 5)

    fig.add_subplot(2, 1, 2)

    error_x = []
    error_y = []
    for key, value in results.items():
        error_x.append(float(key))
        error_y.append(np.max(np.abs(value.error)))

    result = linregress(np.log(error_x), np.log(error_y))
    plt.loglog(
        error_x,
        np.exp(result[1]) * np.power(error_x, result[0]),
        "r--",
        label=f"slope {result[0]:.3f} $r^2=${result[2] ** 2:.3f}",
    )
    plt.loglog(error_x, error_y, "o")
    plt.grid(which="both", linestyle="--", linewidth=0.2)
    plt.legend()
    plt.xlabel(r"$\Delta{}t$ (s)")
    plt.ylabel("absolute error $\\epsilon$")

    fig.tight_layout(pad=0.1)
    fig.savefig("../../PIC/UDD.SDOF.pdf")

    fig = plt.figure(figsize=(6, 2))

    for key, value in results.items():
        plt.plot(
            value.time,
            np.abs(value.error),
            label=f"$\\Delta{{}}t=${float(key):1.0E}",
            linestyle=next(LS),
            linewidth=0.8,
        )

    plt.yscale("log")
    plt.legend(loc="lower right", ncol=2)
    plt.xlabel("time (s)")
    plt.ylabel("absolute error $\\epsilon$")
    plt.grid(which="both", linestyle="--", linewidth=0.2)
    plt.xlim(0, 5)

    fig.tight_layout(pad=0.1)
    fig.savefig("../../PIC/UDD.SDOF.ERROR.pdf")



if __name__ == "__main__":
    run()