"""Score the PWM-slope and demodulated-current observers on matched AIR56 scenarios."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from fem.air56.current_tracking import anisotropic_current, real_features
from fem.air56.near_zero_tracking import ramp_profile
from fem.air56.pwm_section import LOAD_HZ, PROBE_HZ, PWM_PERIOD_S
from fem.air56.slope_tracking import ANGLE_GRID, FluxIncrement, causal_speed
from tools.run_air56_18_slot_openings import file_sha256
from tools.run_air56_loaded_contrast import NOISE
from tools.run_air56_moving_pwm import PERIODS
from tools.run_air56_pwm_slopes import BLOCK_S, DIRECT_ZERO_S, PHASES, SPEEDS
from tools.run_air56_skew_audit import write_csv
from tools.run_air56_standstill_angle import angle_error, periodic_interpolation


STEADY_TRIALS = 64
REST_TRIALS = 128
REVERSAL_TRIALS = 64
RECORD_S = 36.
ZERO_S = 16.
RAMP_S = 2.4
FRAMES = (4, 9, 18, 36)
PREHISTORY = 20


def phase_features(phasors):
    shape = phasors.shape[:-3]
    return anisotropic_current(real_features(phasors)).reshape(*shape, 24)


def phase_noise_factor():
    times = (np.arange(PERIODS)+.5)*PWM_PERIOD_S
    times = times[times >= .2]
    angles = 2*np.pi*times[:, None]*LOAD_HZ*np.arange(1, 41)
    design = np.column_stack((np.ones(len(times)), np.cos(angles), np.sin(angles)))
    fit = np.linalg.pinv(design)
    sidebands = np.arange(-5, 6)
    index = (PROBE_HZ/LOAD_HZ+2*sidebands).astype(int)
    envelope = np.exp(2j*PHASES[:, None]*sidebands) @ (fit[index]-1j*fit[index+40])
    projection = np.concatenate((envelope.real, envelope.imag), axis=0)
    singular = np.linalg.cholesky(projection @ projection.T)
    channels = np.eye(48).reshape(48, 4, 12) @ singular.T
    phasors = (channels[..., :6]+1j*channels[..., 6:]).transpose(0, 2, 1).reshape(48, 6, 2, 2)
    projected = phase_features(phasors)
    return np.linalg.cholesky(projected.T @ projected)


class PhaseMap:
    def __init__(self, nodes, phasors, factor):
        templates = periodic_interpolation(ANGLE_GRID, nodes, phase_features(phasors))
        self.center = templates.mean(axis=0)
        self.factor = factor
        self.templates = np.linalg.solve(factor, (templates-self.center).T).T.copy()
        self.power = .5*np.sum(self.templates**2, axis=1)

    def acquire(self, observations):
        shape = observations.shape[:-1]
        observed = np.linalg.solve(self.factor, (observations-self.center).reshape(-1, 24).T).T
        angle = np.empty(len(observed))
        for start in range(0, len(observed), 512):
            stop = min(start+512, len(observed))
            scores = observed[start:stop] @ self.templates.T-self.power
            angle[start:stop] = ANGLE_GRID[scores.argmax(axis=1)]
        return angle.reshape(shape)


def load_bank(location):
    meta_file = location/"summary.json"
    meta = json.loads(meta_file.read_text(encoding="utf8"))
    path = location/"observations.npz"
    if file_sha256(path) != meta["map_sha256"]:
        raise ValueError(f"invalid checksum: {path}")
    with np.load(path, allow_pickle=False) as data:
        payload = {k: data[k] for k in data.files}
    return payload, meta, file_sha256(meta_file)


def sample_bank(values, nodes, angle, velocity):
    by_speed = np.stack([periodic_interpolation(angle.ravel(), nodes, v).reshape(*angle.shape, *v.shape[1:])
                         for v in values])
    fraction = np.abs(velocity)/SPEEDS[-1]
    dims = (slice(None),)+(None,)*(by_speed.ndim-3)
    negative = np.where((velocity < 0.)[dims], by_speed[0], by_speed[2])
    return by_speed[1]+fraction[dims]*(negative-by_speed[1])


def scenarios(bank):
    time = (np.arange(-PREHISTORY, round(RECORD_S/BLOCK_S))+.5)*BLOCK_S
    rng = np.random.default_rng(802115)
    slopes, phases, angles, labels = [], [], [], []
    for mode, rpm, count in (("steady", 0., REST_TRIALS), ("steady", -.1, STEADY_TRIALS),
                             ("steady", .1, STEADY_TRIALS), ("reversal", -.1, REVERSAL_TRIALS),
                             ("reversal", .1, REVERSAL_TRIALS)):
        initial = rng.uniform(0., np.pi, (count, 1))
        if mode == "reversal":
            velocity, displacement = ramp_profile(time, ZERO_S-RAMP_S/2, RAMP_S, -rpm, rpm)
        else:
            velocity, displacement = np.full(len(time), rpm), rpm*2*np.pi/60*time
        angle = initial+displacement
        slopes.append(sample_bank(bank["slope"], bank["angle_rad"], angle, velocity))
        phases.append(sample_bank(phase_features(bank["phasor"]), bank["angle_rad"], angle, velocity))
        angles.append(angle)
        labels.append((mode, rpm, count))
    return np.concatenate(slopes), np.concatenate(phases), np.concatenate(angles), labels


def combine_slopes(data):
    weights = 1/data[..., 4]
    updated = np.sum(weights[..., None]*data[..., :4], axis=1)/weights.sum(axis=1)[..., None]
    return np.concatenate((updated, 1/weights.sum(axis=1)[..., None]), axis=-1)


def evaluate_rest(model, slope, phase, truth, case):
    for n in (1, 3, 9):
        if case["method"] == "qd_slopes":
            angle = model.acquire(combine_slopes(slope[:REST_TRIALS, :n]))
        else:
            angle = model.acquire(phase[:REST_TRIALS, :n].mean(axis=1))
        error = abs(angle_error(angle, truth[:REST_TRIALS, 0]))
        yield dict(**case, window_s=round(n*BLOCK_S, 3), trials=len(error),
                   median_deg=float(np.median(error)), p95_deg=float(np.percentile(error, 95)),
                   within_5_deg=int((error <= 5).sum()))


def double_confirmation(direction, desired, selector):
    match = (direction == desired) & selector
    return np.column_stack((match[:, :-1] & match[:, 1:], np.zeros(len(match), dtype=bool)))


def dynamic_metrics(angles, labels, case, steady, reversals):
    for n in FRAMES:
        rpm, direction = causal_speed(angles, n, BLOCK_S)
        time = (np.arange(rpm.shape[1])+n)*BLOCK_S
        first = 0
        settings = dict(**case, window_s=round(n*BLOCK_S, 3))
        for mode, target, count in labels:
            chosen = slice(first, first+count)
            first += count
            sign = np.sign(target)
            if mode == "steady":
                estimate = rpm[chosen, 0]
                classified = direction[chosen, 0]
                good = (classified == sign) if sign else np.zeros(count, dtype=bool)
                wrong = (classified != 0) & (classified != sign)
                steady.append(dict(**settings, true_rpm=target, trials=count, confirmed_correct=int(good.sum()),
                                   confirmed_wrong=int(wrong.sum()), no_sign=int((classified == 0).sum()),
                                   median_rpm=float(np.median(estimate)),
                                   p95_error_rpm=float(np.percentile(abs(estimate-target), 95))))
                continue
            prior = double_confirmation(direction[chosen], -sign, time <= ZERO_S-RAMP_S/2).any(axis=1)
            new = double_confirmation(direction[chosen], sign, time >= ZERO_S)
            valid = new.any(axis=1)
            delay = time[np.argmax(new[valid], axis=1)+1]-ZERO_S
            reversals.append(dict(**settings, target_rpm=target, trials=count, old_sign=int(prior.sum()),
                                  new_sign=int(valid.sum()), both_signs=int((prior & valid).sum()),
                                  median_delay_s=float(np.median(delay)) if len(delay) else None,
                                  p95_delay_s=float(np.percentile(delay, 95)) if len(delay) else None,
                                  wrong_after_plateau=int((direction[chosen, time >= ZERO_S+RAMP_S/2+n*BLOCK_S]
                                                           == -sign).any(axis=1).sum()),
                                  final_p95_error_rpm=float(np.percentile(abs(rpm[chosen, -1]-target), 95))))


def direct_checks(directory, skew, qd, phase_map, bank):
    source = directory/f"skew_{skew}"
    info = json.loads((source/"summary.json").read_text(encoding="utf8"))
    for name, checksum in info["files_sha256"].items():
        path = source/name
        if file_sha256(path) != checksum:
            raise ValueError(f"invalid direct reversal checksum: {path}")
        with np.load(path, allow_pickle=False) as data:
            truth, velocity, time = data["angle_rad"], data["rpm"], data["time_s"]
            slope, phase = data["slope"], phase_features(data["phasor"])
            virtual_slope = sample_bank(bank["slope"], bank["angle_rad"], truth[None, :], velocity)
            virtual_phase = sample_bank(phase_features(bank["phasor"]), bank["angle_rad"], truth[None, :], velocity)
            direct = (qd.sequence(slope[None, :]), phase_map.acquire(phase[None, :]))
            virtual = (qd.sequence(virtual_slope), phase_map.acquire(virtual_phase))
            for method, measured, modeled in zip(("qd_slopes", "phase_map"), direct, virtual):
                estimate, label = causal_speed(measured, 9, BLOCK_S)
                timestamp = time[8:]+BLOCK_S/2
                valid = double_confirmation(label, np.sign(velocity[-1]), timestamp >= DIRECT_ZERO_S)[0]
                delay = timestamp[np.argmax(valid)+1]-DIRECT_ZERO_S if valid.any() else None
                yield dict(skew_pitches=skew, direction=path.stem, method=method, frames=len(time), noise_adc_a=0.,
                           min_slope_pair_coverage=float(data["pair_coverage"].min()),
                           angle_vs_truth_p95_deg=float(np.percentile(abs(angle_error(measured, truth)), 95)),
                           angle_vs_bank_rms_deg=float(np.sqrt(np.mean(angle_error(measured, modeled)**2))),
                           window_s=3.6, reversal_delay_s=delay, final_estimate_rpm=float(estimate[0, -1]))


def run(args):
    rest, steady, reversals, source, direct = [], [], [], {}, []
    factor = phase_noise_factor()
    fits = {}
    for skew in (0, 1):
        prefix = args.maps/f"skew_{skew}"
        cal, low, lo_hash = load_bank(prefix/"mesh_0")
        test, fine, hi_hash = load_bank(prefix/"mesh_1")
        source[str(skew)] = {"calibration": lo_hash, "holdout": hi_hash}
        if not np.array_equal(test["rpm"], SPEEDS) or fine["prior_sha256"] != low["prior_sha256"]:
            raise ValueError("FEM maps must contain the same loaded operating point and the three validation speeds")
        qd = FluxIncrement.fit(cal["angle_rad"], cal["slope"][0])
        phase_map = PhaseMap(cal["angle_rad"], cal["phasor"][0], factor)
        slope, phase, truth, labels = scenarios(test)
        if args.direct:
            direct.extend(direct_checks(args.direct, skew, qd, phase_map, test))
        rng = np.random.default_rng(802116)
        slope_white = rng.standard_normal(slope[..., :2].shape)*np.sqrt(slope[..., 4:5])
        phase_white = rng.standard_normal(phase.shape) @ factor.T
        fits[str(skew)] = qd.coefficients.tolist()
        for noise in NOISE:
            noisy_slope = slope.copy()
            noisy_slope[..., :2] += noise*slope_white
            noisy_phase = phase+noise*phase_white
            post_slope, post_phase, post_truth = (data[:, PREHISTORY:] for data in (noisy_slope, noisy_phase, truth))
            for method, model in (("qd_slopes", qd), ("phase_map", phase_map)):
                case = dict(skew_pitches=skew, noise_adc_a=noise, method=method)
                rest.extend(evaluate_rest(model, post_slope, post_phase, post_truth, case))
                if method == "qd_slopes":
                    dynamic_metrics(qd.sequence(post_slope), labels, dict(**case, initial="cold"), steady, reversals)
                    angles = qd.sequence(noisy_slope)[:, PREHISTORY:]
                else:
                    angles = model.acquire(post_phase)
                dynamic_metrics(angles, labels, dict(**case, initial="warm"), steady, reversals)
            print(f"Scored skew {skew}, {1e3*noise:g} mA", flush=True)
    for name, rows in (("angle_at_rest", rest), ("steady_speed", steady), ("reversal", reversals)):
        write_csv(args.out/f"{name}.csv", rows)
    if direct:
        write_csv(args.out/"direct_reversal.csv", direct)
    info = dict(simulation_only=True, sources_sha256=source, fitted_inductance_h=fits, observation_s=BLOCK_S,
                reversal_zero_s=ZERO_S, reversal_ramp_s=RAMP_S, record_s=RECORD_S,
                warm_prehistory_s=PREHISTORY*BLOCK_S,
                noise="independent Gaussian raw alpha/beta current sample, propagated through linear readout",
                csv_sha256={p.name: file_sha256(p) for p in args.out.glob("*.csv")})
    (args.out/"summary.json").write_text(json.dumps(info, indent=2)+"\n", encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--direct", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        parser.error("choose an empty output directory")
    run(args)


if __name__ == "__main__":
    main()
