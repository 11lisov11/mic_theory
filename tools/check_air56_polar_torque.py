"""Independent coenergy finite differences for the moving air-gap torque."""

from pathlib import Path
import argparse
from run_air56_free_field import BandField, Field, np, MU0
from fem.air56.loaded_section import H_OF_B, STEEL_B_T
from tools.run_air56_skew_audit import write_csv


def coenergy(field, theta, bias):
    field.update(theta)
    a = field.equilibrium(bias)[:-3]
    b = np.hypot(*field.grad(a))
    excess = np.maximum(b - STEEL_B_T[-1], 0.0)
    energy = (
        H_OF_B.antiderivative()(np.minimum(b, STEEL_B_T[-1]))
        + H_OF_B(STEEL_B_T[-1]) * excess
        + excess**2 / (2 * MU0)
    )
    w = field.length * (
        a @ (field.pm + field.W @ bias)
        - np.where(field.steel, energy, b**2 / (2 * MU0)) @ field.b.dx.sum(axis=1)
        - 0.5 * a @ (field.extra @ a)
    )
    return w, field.torque(a)


def check(out):
    controls = []
    old = []
    for level, name in ((4, "L4energy"), (7, "L7bg0")):
        src = np.load(out / f"{name}.npz")
        angles = src["angles"]
        w = src["coenergy_j"]
        coarse = (np.roll(w, -1, axis=0) - np.roll(w, 1, axis=0)) / (4 * np.pi / len(w))
        band = BandField(0.0, level)
        bias = np.array([0.8, 0.0])
        for j in (0, 2, 5, 8):
            a = angles[j]
            f = Field(a, level)
            old.append(
                dict(
                    level=level,
                    angle_deg=np.rad2deg(a),
                    coenergy_nm=coarse[j, 0],
                    stress_nm=f.torque(f.equilibrium(bias)[:-3]),
                )
            )
            torque = coenergy(band, a, bias)[1]
            delta = 1e-4
            numerical = (coenergy(band, a + delta, bias)[0] - coenergy(band, a - delta, bias)[0]) / (
                2 * delta
            )
            controls.append(
                dict(
                    level=level,
                    angle_deg=np.rad2deg(a),
                    virtual_work_nm=torque,
                    finite_difference_nm=numerical,
                    absolute_error_nm=abs(torque - numerical),
                )
            )
    write_csv(out / "band_torque_check.csv", controls)
    write_csv(out / "torque_control.csv", old)
    cogging = []
    for level in (3, 4, 5):
        field = BandField(0.0, level)
        for angle in np.arange(0, 15, 0.5):
            field.update(np.deg2rad(angle))
            cogging.append(dict(level=level, deg=angle, nm=field.torque(field.equilibrium()[:-3])))
    write_csv(out / "cogging.csv", cogging)
    if max(r["absolute_error_nm"] for r in controls) > 5e-5:
        raise ValueError("band torque does not match coenergy derivative")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    check(p.parse_args().out)
