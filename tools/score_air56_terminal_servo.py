"""Score only independent shaft motion. Angles in rad, speeds reported in rpm."""

import argparse, json
from pathlib import Path
import numpy as np


def score(trace, leg=4.2, start=1.0, seconds=None):
    t, actual, estimated, ref = trace.T[:4]
    if len(t) < 2 or (seconds is not None and t[-1] < seconds - 0.51 * (t[1] - t[0])):
        return {"complete": False}
    wrap = lambda x: np.rad2deg(np.angle(np.exp(1j * x)))
    windows = {
        "forward": (start + 0.65, start + leg - 0.02),
        "reverse": (start + leg + 0.65, start + 2 * leg - 0.02),
        "hold": (start + 2 * leg + 0.55, t[-1]),
    }
    result = {}
    for key, (lo, hi) in windows.items():
        mask = (t >= lo) & (t <= hi + 1e-9)
        if not np.any(mask) or t[mask][-1] - t[mask][0] < 0.8:
            return {"complete": False}
        read = lambda v: np.polyfit(t[mask], v[mask], 1)[0] * 30 / np.pi
        result[key + "_rpm"] = float(read(actual))
        result[key + "_read_rpm"] = float(read(estimated))
        result[key + "_duration_s"] = float(np.ptp(t[mask]))
        ends = np.arange(lo + 1.0, hi, 0.1)
        if len(ends):
            for name, position in (("shaft", actual), ("observer", estimated)):
                rpm = (np.interp(ends, t, position) - np.interp(ends - 1.0, t, position)) * 30 / np.pi
                result[key + "_" + name + "_1s_min_rpm"] = float(rpm.min())
                result[key + "_" + name + "_1s_max_rpm"] = float(rpm.max())
                if key != "hold":
                    sign = 1 if key == "forward" else -1
                    result[key + "_" + name + "_1s_sign_fraction"] = float(np.mean(sign * rpm > 0))
    mask = t >= start + 0.65
    result.update(
        max_all_read_deg=float(np.max(abs(wrap(estimated - actual)))),
        p95_read_deg=float(np.percentile(abs(wrap(estimated[mask] - actual[mask])), 95)),
        p95_track_deg=float(np.percentile(abs(wrap(ref[mask] - actual[mask])), 95)),
        hold_range_deg=float(np.rad2deg(np.ptp(actual[t >= windows["hold"][0]]))),
        peak_current_a=float(np.max(np.linalg.norm(trace[:, 13:15], axis=1))),
        max_command_v=float(np.max(np.linalg.norm(trace[:, 11:13], axis=1))),
        complete=True,
    )
    if trace.shape[1] >= 19:
        holding = trace[t >= windows["hold"][0]]
        result["hold_switch_range_deg"] = float(np.rad2deg(holding[:, 18].max() - holding[:, 17].min()))
    result["passes"] = (
        all(
            abs(result[key + suffix] - target) <= tol
            for key, target, tol in [("forward", 0.1, 0.02), ("reverse", -0.1, 0.02), ("hold", 0, 0.02)]
            for suffix in ("_rpm", "_read_rpm")
        )
        and max(result["p95_read_deg"], result["p95_track_deg"], result["hold_range_deg"]) <= 1.0
        and result["max_all_read_deg"] < 90.0
        and result.get("hold_switch_range_deg", result["hold_range_deg"]) <= 1.0
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("traces", type=Path, nargs="+")
    p.add_argument("--leg", type=float, default=4.2)
    p.add_argument("--start", type=float, default=1.0)
    p.add_argument("--seconds", type=float)
    a = p.parse_args()
    for path in a.traces:
        print(
            path.stem,
            json.dumps(score(np.load(path)["trace"], a.leg, a.start, a.seconds), ensure_ascii=False),
        )
