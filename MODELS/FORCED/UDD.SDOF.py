import os
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
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
    error: np.ndarray = None  # type: ignore


matplotlib.rcParams.update({"font.size": 6})


model = """
node 1 0 0
node 2 1 0

material Elastic1D 1 100

element T2D2 1 1 2 1 1
element Mass 2 2 1 1

fix2 1 1 1
fix2 2 2 1 2

hdf5recorder 1 Node U 2

amplitude Sine 1 1. 1.

cload 1 1 1 1 2

step dynamic 1 10
set ini_step_size {step_time}
set fixed_step_size 1

{damping}

converger AbsIncreDisp 2 1E-14 10 0

analyze

save recorder 1

terminal mv R1-U.h5 R1-U-{scheme}-{step_time}.h5

exit
"""


def execute(step_time: str, damping: str, scheme: str):
    print(f"Executing with step_time={step_time}...")
    target = Path("model.sp")
    target.write_text(model.format(step_time=step_time, damping=damping, scheme=scheme))
    os.system("suanpan -np -f model.sp")
    target.unlink()


def sdof_system(s, m, omega_n, A, B):
    K = (1.0 - m / s) * omega_n**2
    X = s * (omega_n**2 - B**2)
    Y = B * (K - B**2)

    D = X**2 + Y**2

    Uc = (A * m * B * omega_n**2) / D
    Us = (A * (s * X + B * Y)) / D

    r1, r2, r3 = np.roots([1.0, s, K, s * omega_n**2])

    num1 = -Uc * (r2 * r3) + B * Us * (r2 + r3) + (B**2) * Uc
    den1 = (r1 - r2) * (r1 - r3)
    C1 = num1 / den1

    num2 = -Uc * (r1 * r3) + B * Us * (r1 + r3) + (B**2) * Uc
    den2 = (r2 - r1) * (r2 - r3)
    C2 = num2 / den2

    num3 = -Uc * (r1 * r2) + B * Us * (r1 + r2) + (B**2) * Uc
    den3 = (r3 - r1) * (r3 - r2)
    C3 = num3 / den3

    def displacement(t):
        t_arr = np.atleast_1d(t)

        total_u = np.real(
            C1 * np.exp(r1 * t_arr)
            + C2 * np.exp(r2 * t_arr)
            + C3 * np.exp(r3 * t_arr)
            + Uc * np.cos(B * t_arr)
            + Us * np.sin(B * t_arr)
        )

        if np.isscalar(t) or (isinstance(t, np.ndarray) and t.ndim == 0):
            return total_u[0]

        return total_u

    return displacement


def analytical(*para):
    disp = sdof_system(*para)
    dt, t = 0.01, 10
    x = np.linspace(0, t, int(t / dt) + 1)

    plt.plot(x, disp(x), "r", label="analytical", linewidth=0.5)

    return disp


def numerical(disp, pick, scheme):
    name = "R1-U"
    with h5py.File(f"{name}-{scheme}-{str(pick)}.h5", "r") as f:
        data = f[f"/{name}/{name}2"]

        return Response(
            time := data[:, 0],  # type: ignore
            displacement := data[:, 1],  # type: ignore
            disp(time) - displacement,
        )


def generate(damping, steps, refresh=False):
    scheme = "UDD" if "UDD" in damping else "UDA"

    results = {}

    fig = plt.figure(figsize=(6, 3.5))
    fig.add_subplot(211)

    disp = analytical(10, -2, 10, 1, 2 * np.pi)
    for step_time in steps:
        if refresh:
            execute(step_time, damping, scheme)
        results[step_time] = numerical(disp, float(step_time), scheme)

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

    error_x = [float(key) for key in results.keys()]
    error_y = [np.max(np.abs(value.error)) for value in results.values()]

    result = linregress(np.log(error_x), np.log(error_y))
    plt.loglog(
        error_x,
        np.exp(result[1]) * np.power(error_x, result[0]),  # type: ignore
        "r--",
        label=f"slope {result[0]:.3f} $r^2=${result[2] ** 2:.3f}",  # type: ignore
    )
    plt.loglog(error_x, error_y, "o")
    plt.grid(which="both", linestyle="--", linewidth=0.2)
    plt.legend()
    plt.xlabel(r"$\Delta{}t$ (s)")
    plt.ylabel("absolute error $\\epsilon$")

    fig.tight_layout(pad=0.1)
    fig.savefig(f"../../PIC/{scheme}.SDOF.FORCED.pdf")

    fig = plt.figure(figsize=(6, 2))

    v_min, v_max = 1, 0

    for key, value in results.items():
        segment = np.abs(value.error[len(value.error) // 20 :])
        v_min = min(v_min, np.min(segment))
        v_max = max(v_max, np.max(segment))
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
    plt.ylim(v_min, v_max * 2)

    fig.tight_layout(pad=0.1)
    fig.savefig(f"../../PIC/{scheme}.SDOF.ERROR.FORCED.pdf")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)

    generate(
        "integrator UDDNewmark 1 .25 .5 -2 0 10 0",
        ["0.0001", "0.0002", "0.0005", "0.001", "0.002", "0.005", "0.01"],
    )

    generate(
        f"integrator UDANewmark 1 .25 .5 {25.0 / 18.0} 0 {25.0 / 3.0} 0",
        ["0.0001", "0.0002", "0.0005", "0.001", "0.002", "0.005", "0.01"],
    )
