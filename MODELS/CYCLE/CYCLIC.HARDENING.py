import subprocess
import tempfile
from pathlib import Path

import h5py
from matplotlib import pyplot as plt
import numpy as np

model = """node 1 0 0
node 2 1 0

material Elastic1D 1 {freq2}

element T2D2 1 1 2 1 1
element Mass 2 2 1 1

fix2 1 1 1
fix2 2 2 1 2

hdf5recorder 1 Node U 2
hdf5recorder 2 Node RF 2

amplitude Sine 1 {period} {amplitude}

cload 1 1 1 1 2

step dynamic 1 {total_time}
set ini_step_size {time_step}
set fixed_step_size 1

integrator UDDNewmark 1 .25 .5 -20 0 100 0

converger AbsIncreDisp 2 1E-14 10 0

analyze

peek node 2

save recorder 1 2

exit
"""


def _read_first_column_h5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        datasets = []

        def visitor(_, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append(obj)

        f.visititems(visitor)

        if not datasets:
            raise ValueError(f"No dataset found in {path}")

        data = np.asarray(datasets[0][...])

    return data if data.ndim == 1 else data[:, 1]


def run_cyclic_hardening(freq: float, period: float):
    with tempfile.TemporaryDirectory(prefix="suanpan_") as tmp:
        tmp_dir = Path(tmp)
        inp_file = tmp_dir / "model.sp"

        inp_file.write_text(
            model.format(
                freq2=freq**2,
                amplitude=1,
                period=period,
                total_time=2 * period,
                time_step=period / 200,
            ),
            encoding="utf-8",
        )

        subprocess.run(
            ["suanpan", "-np", "-f", str(inp_file)],
            cwd=tmp_dir,
            check=True,
            capture_output=False,
            text=True,
        )

        disp = _read_first_column_h5(tmp_dir / "R1-U.h5")
        force = _read_first_column_h5(tmp_dir / "R2-RF.h5")

        return disp, force


if __name__ == "__main__":
    plt.figure(figsize=(6, 3.5))

    for period in [1e-1]:
        disp, force = run_cyclic_hardening(freq := 1, period)
        plt.plot(disp, force, marker="x", label=f"{period}")
    plt.legend()
    plt.tight_layout()
    plt.show()
