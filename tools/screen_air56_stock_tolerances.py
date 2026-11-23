"""Estimate the size of electrical sector fingerprints from hypothetical as-built scatter.

This linear, locked-rotor P2 field calculation assumes known material patterns and
known angle within a slot pitch. The two 1 V orthogonal probes are separate experiments.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fem.air56.sliding_section import make_section, solve
from tools.run_air56_skew_audit import article_profile


def screen(level, slices, seed, frequency):
    g, design = article_profile()
    pitch = 2 * np.pi / g.rotor_bars
    rng = np.random.default_rng(seed)
    bars = rng.uniform(-1, 1, g.rotor_bars)
    teeth = rng.uniform(-1, 1, g.stator_slots)
    # Reuse the same physical rotor mesh for the 18 cyclic permutations.
    offsets = 0.5 * (2 * (np.arange(slices) + 0.5) / slices - 1) * pitch
    all_z = np.zeros((2, g.rotor_bars, 2, 2), complex)
    for offset in offsets:
        angle = np.deg2rad(7.32) + offset
        sec = make_section(
            g, angle, level, design, stator_opening_fraction=0.16, stator_opening_depth_m=0.00065
        )
        x, y = sec.mesh.p[:, sec.mesh.t].mean(axis=1)
        phase = np.arctan2(y, x)
        bar = np.floor(((phase - angle) % (2 * np.pi)) / pitch).astype(int) % g.rotor_bars
        tooth = np.floor((phase % (2 * np.pi)) / (2 * np.pi / g.stator_slots)).astype(int) % g.stator_slots
        iron = sec.steel & (np.hypot(x, y) > g.rotor_radius + g.air_gap)
        for variant, spread in enumerate((0.0, 0.10)):
            mu = np.where(sec.steel, g.mu_r, 1.0)
            mu[iron] *= 1 + spread * teeth[tooth[iron]]
            for j in range(g.rotor_bars):
                sig = np.where(sec.cage, g.aluminium_sigma * (1 + 0.05 * bars[(bar - j) % g.rotor_bars]), 0.0)
                all_z[variant, j] += (
                    solve(g, frequency, sec, 0.0, element_mu=mu, element_sigma=sig)["z"] / slices
                )
    rows = []
    for spread, z in zip((0.0, 0.10), all_z):
        amps = np.linalg.inv(z + (27.5 / 3) * np.eye(2))
        diff = amps[:, None] - amps[None, :]
        dist = 1e6 * np.sqrt(np.sum(abs(diff) ** 2, axis=(-1, -2)))
        np.fill_diagonal(dist, np.inf)
        opposite = dist[np.arange(g.rotor_bars), (np.arange(g.rotor_bars) + 9) % g.rotor_bars]
        rows.append(
            dict(
                stator_mu_scatter=spread,
                rotor_sigma_scatter=0.05,
                level=level,
                axial_slices=slices,
                frequency_hz=frequency,
                seed=seed,
                min_pair_ua_per_v=float(dist.min()),
                max_antipodal_ua_per_v=float(opposite.max()),
                max_pair_ua_per_v=float(dist[np.isfinite(dist)].max()),
                half_turn_ua_per_v=opposite.tolist(),
                real_a_per_v=amps.real.tolist(),
                imag_a_per_v=amps.imag.tolist(),
            )
        )
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--slices", type=int, default=1)
    p.add_argument("--seed", type=int, default=96041)
    p.add_argument("--frequency", type=float, default=83.0)
    a = p.parse_args()
    result = screen(a.level, a.slices, a.seed, a.frequency)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(
        [(r["min_pair_ua_per_v"], r["max_antipodal_ua_per_v"], r["max_pair_ua_per_v"]) for r in result],
        flush=True,
    )
