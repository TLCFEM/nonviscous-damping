from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess

from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams.update({"font.size": 6})

model = """file node
file element

material Elastic2D 1 1E5 .2 1e-1

fix2 1 E 1 4 7 8 10 12 14 17 18 20 338 339 340 341 342 343 344 345 346 510 511 512 513 514 515 516 517 518 673 674 675 676 677 678 679 680 681 854 855 856 857 858 859 860 861 862 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 1086 1087 1088 1089 1090 1091 1092 1093 1094 1258 1259 1260 1261 1262 1263 1264 1265 1266 1434 1435 1436 1437 1438 1439 1440 1441 1442 1443 1444 1445 1446 1601 1602 1603 1604 1605 1606 1607 1608 1609 1786 1787 1788 1789 1790 1791 1792 1793 1794 1795 1796 1797 1798

displacement 1 0 -5 3 2 3 6 175 176 177 178 179 180 181 182 183 682 683 684 685 686 687 688 689 690

step {analysis} 1
set ini_step_size 4E-2
set fixed_step_size 1
set symm_mat 1
set sparse_mat 0
set color_model MIS

{damping}

converger AbsIncreDisp 2 1E-8 20 0

analyze

peek stats

exit
"""


@dataclass
class TimingStats:
    initialization: float
    element_state_determination: float
    assembling_global_vector: float
    assembling_global_matrices: float
    assembling_global_effective_matrix: float
    processing_constraints_and_loads: float
    solving_global_system: float


_FLOAT_RE = r"([+-]?\d+\.\d+E[+-]\d+)"


def _extract_time(stdout: str, label: str) -> float:
    m = re.search(
        rf"{re.escape(label)} used:\s*{_FLOAT_RE}\s*s\.", stdout, flags=re.MULTILINE
    )
    if not m:
        raise ValueError(f"Could not find timing for '{label}' in solver output.")
    return float(m.group(1))


def create_comparison_figure(results: dict[str, TimingStats], out_path: str) -> None:
    labels = list(results.keys())
    components = [
        "initialization",
        "element_state_determination",
        "assembling_global_vector",
        "assembling_global_matrices",
        "assembling_global_effective_matrix",
        "processing_constraints_and_loads",
        "solving_global_system",
    ]

    fig, ax = plt.subplots(figsize=(6, 2.5))

    lefts = [0.0] * len(labels)
    for comp in components:
        vals = [getattr(results[k], comp) for k in labels]
        ax.barh(labels, vals, left=lefts, label=comp.replace("_", " "))
        lefts = [left + v for left, v in zip(lefts, vals)]

    ax.set_xlabel("Time (s)")
    ax.invert_yaxis()
    ax.legend()

    fig.tight_layout()

    fig.savefig(out_path)
    plt.close(fig)


def run(line: str):
    target = Path("model.sp")
    target.write_text(
        model.format(
            analysis="static" if line == "Static" else "dynamic",
            damping="" if line == "Static" else line,
        )
    )
    result = subprocess.run(
        ["sp", "-nc", "-f", "model.sp"], capture_output=True, text=True
    )
    target.unlink()
    stdout = result.stdout
    return TimingStats(
        initialization=_extract_time(stdout, "Initialization"),
        element_state_determination=_extract_time(
            stdout, "Updating element trial status"
        ),
        assembling_global_vector=_extract_time(stdout, "Assembling global vector"),
        assembling_global_matrices=_extract_time(stdout, "Assembling global system"),
        assembling_global_effective_matrix=_extract_time(
            stdout, "Assembling global effective system"
        ),
        processing_constraints_and_loads=_extract_time(
            stdout, "Processing constraints"
        ),
        solving_global_system=_extract_time(stdout, "Solving global system"),
    )


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    result = {}
    for key, line in {
        "Static": "Static",
        "Newmark": "",
        "UDA": "integrator UDANewmark 1 .25 .5 5.1234e-03 0.0000e+00 7.4035e-02 0.0000e+00 6.5596e-08 0.0000e+00 7.9445e-02 0.0000e+00 1.2098e-02 0.0000e+00 3.6317e-01 0.0000e+00 2.9694e-02 0.0000e+00 1.1389e+00 0.0000e+00 6.9713e-02 0.0000e+00 3.0807e+00 0.0000e+00 1.9259e-01 0.0000e+00 8.2947e+00 0.0000e+00 6.7065e-01 0.0000e+00 2.5722e+01 0.0000e+00 5.0968e+00 0.0000e+00 1.1907e+02 0.0000e+00",
        "UDD": "integrator UDDNewmark 1 .25 .5 -1.4087e-03 -0.0000e+00 7.9434e-02 0.0000e+00 -4.3464e-03 -0.0000e+00 7.9479e-02 0.0000e+00 -1.4968e-02 -0.0000e+00 3.7664e-01 0.0000e+00 -3.9049e-02 -0.0000e+00 1.1730e+00 0.0000e+00 -9.6939e-02 -0.0000e+00 3.1630e+00 0.0000e+00 -2.8394e-01 -0.0000e+00 8.5288e+00 0.0000e+00 -1.0557e+00 -0.0000e+00 2.6564e+01 0.0000e+00 -9.1164e+00 -0.0000e+00 1.2589e+02 0.0000e+00",
    }.items():
        result[key] = run(line)

    create_comparison_figure(result, "../../PIC/PERFORMANCE.pdf")
