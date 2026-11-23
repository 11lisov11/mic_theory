"""Screen counter-rotating voltage probes on an unmodified closed-slot AIR56 cross-section.

The fundamental slip-frequency approximation includes no mechanical transient or ADC.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_air56_skew_audit import article_profile
from fem.air56.sliding_section import make_section, solve


def screen(level, frequencies=(1., 3., 10., 30.), angular_samples=6):
    geom, stock = article_profile()
    pitch = 2 * np.pi / geom.rotor_bars
    angles = np.arange(angular_samples) * pitch / angular_samples
    slices = [make_section(geom, a, level, stock, stator_opening_fraction=.16,
                           stator_opening_depth_m=.0007) for a in angles]
    resistance = (27.5 / 3) * np.eye(2)
    result = []
    for frequency in frequencies:
        for rpm in (0., .1, -.1):
            currents = [[], []]
            for direction in (1, -1):
                axis = np.array([1, -1j * direction])
                rotor_frequency = frequency - direction * rpm / 60  # one pole pair
                fields = [solve(geom, rotor_frequency, section, 0.)["z"] for section in slices]
                for skew, field in enumerate((fields[0], np.mean(fields, axis=0))):
                    # Field diffusion uses the rotor frequency, stator EMF uses the probe frequency.
                    z = resistance + (frequency / rotor_frequency) * field
                    currents[skew].append(np.vdot(axis, np.linalg.solve(z, axis)) / 2)
            for skew, (forward, backward) in enumerate(currents):
                difference = forward - backward
                result.append(dict(level=level, frequency_hz=frequency, skew_pitch=skew,
                                   true_rpm=rpm, real_difference_ua_v=difference.real*1e6,
                                   imag_difference_ua_v=difference.imag*1e6,
                                   magnitude_ua_v=abs(difference)*1e6))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = screen(3) + screen(5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for r in rows:
        if r["true_rpm"] == .1:
            print(r["level"], r["frequency_hz"], r["skew_pitch"], round(r["magnitude_ua_v"], 3))
