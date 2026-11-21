"""Compare the legacy per-speed transport with the vectorized equivalent."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import timeit

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.active_belief_rotor_observer import ActiveBeliefRotorObserver
from models.rotor_slot_saliency import SlotSaliency


def reference(observer: ActiveBeliefRotorObserver, values: np.ndarray, dt_s: float) -> np.ndarray:
    out = np.empty_like(values)
    bin_width = observer.period_rad / len(observer.angles)
    for j, speed in enumerate(observer.speeds):
        shift = speed * dt_s / bin_width
        whole = math.floor(shift)
        frac = shift - whole
        out[:, j, :] = ((1.0 - frac) * np.roll(values[:, j, :], whole, axis=0)
                        + frac * np.roll(values[:, j, :], whole + 1, axis=0))
    out = observer._neighbor_diffusion(out, 1, observer.speed_diffusion)
    out = observer._neighbor_diffusion(out, 2, observer.gain_diffusion)
    return observer._angle_diffuse(out)


def benchmark(repeats: int, number: int) -> dict:
    speeds = tuple(sorted({-20.0, -5.0, *np.arange(-3.0, 3.01, 0.25), 5.0, 20.0}))
    observer = ActiveBeliefRotorObserver(
        SlotSaliency(strength=0.15), noise_std_a=0.002, speed_hypotheses=speeds,
        angle_diffusion=0.003, speed_diffusion=0.001, gain_diffusion=0.001,
    )
    values = np.random.default_rng(121).random(observer.weights.shape)
    dt_s = 100e-6
    np.testing.assert_allclose(observer._transport(values, dt_s), reference(observer, values, dt_s),
                               rtol=0.0, atol=1e-15)
    old = timeit.repeat(lambda: reference(observer, values, dt_s), repeat=repeats, number=number)
    new = timeit.repeat(lambda: observer._transport(values, dt_s), repeat=repeats, number=number)
    return {"hypotheses": observer.weights.size, "repeats": repeats, "number": number,
            "reference_us_median": float(np.median(old) / number * 1e6),
            "vectorized_us_median": float(np.median(new) / number * 1e6),
            "speedup_ratio": float(np.median(old) / np.median(new)),
            "numerically_equivalent_atol": 1e-15}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--number", type=int, default=500)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.number < 1:
        parser.error("repeats and number must be >=1")
    result = benchmark(args.repeats, args.number)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
