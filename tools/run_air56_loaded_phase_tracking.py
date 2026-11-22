"""Speed readout on phase-resolved frozen-tangent FEM maps of a loaded rotor."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from fem.air56.current_tracking import CurrentMap, real_features
from tools.run_air56_18_slot_openings import file_sha256
from tools.run_air56_loaded_contrast import NOISE, SLIP_HZ, score_case
from tools.run_air56_skew_audit import write_csv
from tools.run_air56_slot_comparison import skewed


PHASE_INTERVAL_S = 1/(12*SLIP_HZ)
SAMPLE_INTERVAL_S = 2*PHASE_INTERVAL_S
WINDOW_S = 3.6


@dataclass(frozen=True)
class ScheduledMap:
    maps: tuple[CurrentMap, ...]
    average: CurrentMap

    @property
    def period(self):
        return self.average.period

    @property
    def admittance(self):
        return self.average.admittance


def scheduled_map(impedances, offset, skew_rad):
    maps = tuple(CurrentMap.from_impedance(np.pi, z, offset) for z in impedances.swapaxes(0, 1))
    if skew_rad:
        maps = tuple(skewed(m, skew_rad) for m in maps)
    average = CurrentMap(np.pi, np.mean([m.admittance for m in maps], axis=0), offset)
    return ScheduledMap(maps, average)


def scheduled_observations(table, theta0, rpm, times, _probe_hz, volts):
    """Pair alpha and beta at adjacent known phases of the commanded load field."""
    speed = np.asarray(rpm)*2*np.pi/60
    theta = np.asarray(theta0)[..., None]+speed[..., None]*times
    phase_index = (2*np.arange(len(times))) % 6
    current_matrix = np.empty((*theta.shape, 2, 2), dtype=complex)
    for column in (0, 1):
        branch_theta = theta+column*speed[..., None]*PHASE_INTERVAL_S
        response = current_matrix[..., column]
        if isinstance(table, CurrentMap):
            response[...] = table.sample(branch_theta)[..., column]
        else:
            branches = (phase_index+column) % 6
            for phase in range(column, 6, 2):
                at_phase = branches == phase
                response[..., at_phase, :] = table.maps[phase].sample(branch_theta[..., at_phase])[..., column]
    return volts*real_features(current_matrix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fem-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out/"summary.json"
    if summary_path.exists():
        parser.error("choose a new output directory")

    start = perf_counter()
    prior_file = args.fem_dir/"summary.json"
    prior = json.loads(prior_file.read_text(encoding="utf8"))
    if prior["carrier_phases"] != 6 or prior["slip_hz"] != SLIP_HZ:
        raise ValueError("phase tracking expects the six phase FEM result at the specified load frequency")
    raw, summary = [], []
    for profile in ("bar_width", "modulated"):
        source = args.fem_dir/f"{profile}_loaded_impedance.npz"
        if prior["files_sha256"][source.name] != file_sha256(source):
            raise ValueError(f"FEM checksum mismatch for {source.name}")
        with np.load(source, allow_pickle=False) as data:
            first, second = data["calibration_phase_ohm"], data["holdout_phase_ohm"]
            offset = float(data["holdout_offset_rad"])
        for skew in (0., 1.):
            print(f"Scoring {profile} scheduled phases, {skew:g} pitch of skew", flush=True)
            pitch = skew*np.deg2rad(prior["pitch_deg"])
            calibration = scheduled_map(first, 0., pitch)
            truth = scheduled_map(second, offset, pitch)
            case = dict(profile=profile, skew_pitches=skew, window_s=WINDOW_S)
            for template, model, noise_set in (("phase_known", calibration, NOISE),
                                                ("mean_on_raw_phases", calibration.average, NOISE[:1])):
                for noise in noise_set:
                    score_case(model, truth, dict(**case, template=template, noise_a=noise), raw, summary,
                               sample_interval=SAMPLE_INTERVAL_S, observation_model=scheduled_observations)
    write_csv(args.out/"trials.csv", raw)
    write_csv(args.out/"speed_results.csv", summary)
    payload = dict(simulation_only=True, input_summary_sha256=file_sha256(prior_file),
                   window_s=WINDOW_S, sample_interval_s=SAMPLE_INTERVAL_S, phase_interval_s=PHASE_INTERVAL_S,
                   signed_rpm=prior["signed_rpm"], noise_a=NOISE, speed_results=summary, trial_count=len(raw),
                   elapsed_s=perf_counter()-start,
                   files_sha256={name: file_sha256(args.out/name) for name in ("trials.csv", "speed_results.csv")})
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf8")


if __name__ == "__main__":
    main()
