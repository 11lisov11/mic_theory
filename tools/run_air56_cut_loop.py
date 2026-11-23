"""Screen EDM recess codes and a conducting printed loop through the existing phase terminals."""

from pathlib import Path
import argparse
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fem.air56.sliding_section import make_section, solve
from tools.run_air56_skew_audit import article_profile, write_csv


def cut_model(root, level):
    g, design = article_profile()
    # Closed stock rotor, unequal widths and nonuniform spacing. Cut to air.
    marks = [(0, 18, 0.5), (65, 10, 0.25), (215, 26, 0.5)]
    angles = (np.arange(36) + 0.5 * (level > 4)) * 2 * np.pi / 36
    responses = []
    for angle in angles:
        sec = make_section(g, angle, level, design, int(level > 4))
        x, y = sec.mesh.p[:, sec.mesh.t].mean(axis=1)
        local = np.rad2deg(np.arctan2(y, x) - angle)
        r = np.hypot(x, y)
        cut = np.zeros(len(x), bool)
        for center, width, depth in marks:
            cut |= (
                (abs((local - center + 180) % 360 - 180) < width / 2)
                & (r > g.rotor_radius - depth * 0.001)
                & (r < g.rotor_radius)
            )
        mu = np.where(sec.steel & ~cut, g.mu_r, 1.0)
        ans = solve(g, 100.0, sec, 27.5 / 3, element_mu=mu)
        if max(ans[k] for k in ("energy", "residual", "reciprocity")) > 1e-7:
            raise RuntimeError("EDM field consistency")
        responses.append(ans["z"])
    np.savez_compressed(root / f"edm{level}.npz", z=responses, angles=angles)


def loop_screen(stock, out):
    data = np.load(stock)
    rows = []
    # Single axial rectangular turn, uniform radial flux over aperture;
    # effective inductance sweep deliberately includes mounting uncertainty.
    for j, hz in enumerate((100, 500, 2500)):
        w = 2 * np.pi * hz
        field = data["field_t_per_a"][1, j, :, 0, :]
        z = data["z"][1, j]
        for length in (0.004, 0.040):
            width = 0.004
            r = 1.72e-8 * 2 * (length + width) / (0.00030 * 35e-6)
            for lh in (0.1e-6, 1e-6):
                m = field * width * length
                for mode in ("open", "short", "resonant"):
                    z2 = complex(r, 0 if mode == "resonant" else w * lh)
                    # Field is complex from eddy currents. Reciprocal phasor
                    # coupling uses transpose, not Hermitian transpose.
                    dz = (
                        (w * w / z2) * m[:, :, None] * m[:, None, :]
                        if mode != "open"
                        else np.zeros_like(z)
                    )
                    delta = np.linalg.inv(z + dz) - np.linalg.inv(z)
                    centered = delta - delta.mean(axis=0)
                    rows.append(
                        dict(
                            hz=hz,
                            length_mm=length * 1000,
                            inductance_uh=lh * 1e6,
                            mode=mode,
                            r_ohm=r,
                            c_resonance_uf=(1e6 / w**2 / lh if mode == "resonant" else 0),
                            max_contrast_ua_per_v=1e6 * np.linalg.norm(centered, axis=(1, 2)).max(),
                        )
                    )
    write_csv(out / "printed_loop.csv", rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--level", type=int)
    p.add_argument("--stock", type=Path)
    a = p.parse_args()
    if a.level:
        cut_model(a.out, a.level)
    if a.stock:
        loop_screen(a.stock, a.out)
