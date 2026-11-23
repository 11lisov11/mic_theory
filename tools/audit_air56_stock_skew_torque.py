"""Resolve the angular/axial quadrature of locked-stock-rotor DC torque."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_air56_terminal_servo import Machine


def audit(level, samples=144):
    m = Machine(level, 0, False, stock=True)
    pitch = 2 * np.pi / 18
    torque = np.array(
        [m.equilibrium(k * pitch / samples, np.array([0.6, 0.0]))[-6] for k in range(samples)]
    )
    extra = m.equilibrium(pitch, np.array([0.6, 0.0]))[-6]
    assert np.isclose(extra, torque[0], atol=1e-8)
    rows = []
    for n in (1, 3, 6, 12, 24, 48, 144):
        # For a whole pitch the midpoint convention only shifts the phase.
        mean = np.mean([np.roll(torque, j * (samples // n)) for j in range(n)], axis=0)
        rows.append(
            dict(
                level=level,
                slices=n,
                maximum_abs_nm=float(abs(mean).max()),
                mean_nm=float(mean.mean()),
                peak_to_peak_nm=float(np.ptp(mean)),
            )
        )
    return dict(rows=rows, raw_field_torque_nm=torque.tolist())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--samples", type=int, choices=(144, 288), default=144)
    a = p.parse_args()
    record = audit(a.level, a.samples)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(record, indent=2), encoding="utf8")
    print(record["rows"])
