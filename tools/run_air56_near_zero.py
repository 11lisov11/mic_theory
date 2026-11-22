"""Build near-zero moving PWM truth records and evaluate standstill and reversals."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from fem.air56.current_tracking import real_features
from fem.air56.moving_pwm import common_model, moving_circuit, moving_run
from fem.air56.near_zero_tracking import BIN_SECONDS, PAIRS_PER_BIN, UNCERTAIN, Detector, phase_bins, ramp_profile
from fem.air56.pwm_section import PWM_PERIOD_S, demodulate
from tools.run_air56_18_slot_openings import file_sha256
from tools.run_air56_loaded_contrast import NOISE
from tools.run_air56_loaded_phase_tracking import PHASE_INTERVAL_S, SAMPLE_INTERVAL_S, scheduled_observations
from tools.run_air56_moving_pwm import (
    AXIAL_MODES, LEVELS, MOTION_BLOCK, PHASE_NODES, RANK, RECORD_CENTER_S, SECTOR_NODES, TARGET_POINTS,
    article_profile, build_sections, measured, read_maps,
)
from tools.run_air56_pwm_comparison import phase_current_map
from tools.run_air56_skew_audit import write_csv


SPEEDS = np.array([-.5, -.2, -.1, 0., .1, .2, .5])
WINDOWS = (1.2, 3.6, 7.2, 14.4, 21.6)
TRANSITION_WINDOWS = (3.6, 7.2, 14.4)
STEADY_TRIALS = 64
TRANSITION_TRIALS = 32
RAMP_START_S = 15.
RECORD_S = 33.6
DECISION_STRIDE = 2
SEED = 746100


def build_maps(args):
    start = perf_counter()
    source = args.loaded_dir/"summary.json"
    previous = json.loads(source.read_text(encoding="utf8"))
    geometry, closed = article_profile()
    design = replace(closed, bar_width_modulation=.60)
    peak = previous["current_peak_a"]["bar_width"]
    sections, fem_offset, working = build_sections(geometry, design, peak, mesh_index=1)
    offset = np.pi/(2*TARGET_POINTS)
    angles = offset+np.arange(TARGET_POINTS)*np.pi/TARGET_POINTS
    angular_step = np.pi/(3*SECTOR_NODES)
    y = np.empty((len(SPEEDS), len(angles), 6, 2, 2), dtype=complex)
    motional = 0.
    for first in range(0, len(angles), 2):
        targets = angles[first:first+2]
        low = int(np.floor((targets[0]-fem_offset)/angular_step))-1-2*args.skew
        high = int(np.ceil((targets[-1]-fem_offset)/angular_step))+1+2*args.skew
        model = common_model(sections, geometry.rotor_radius, RANK, fem_offset, tuple(range(low, high+1)))
        circuit = moving_circuit(model, geometry.stack_length, args.skew, PHASE_NODES, AXIAL_MODES)
        for s, speed in enumerate(SPEEDS):
            for i, theta in enumerate(targets, start=first):
                y[s, i], voltage = measured(circuit, peak, theta, speed)
                motional = max(motional, voltage)
        if (first+2) % 6 == 0:
            print(f"Skew {args.skew}: {first+2}/{len(angles)} angles, all seven speeds", flush=True)
    path = args.out/"near_zero_pwm.npz"
    np.savez_compressed(path, signed_rpm=SPEEDS, offset_rad=offset, admittance_a_per_v=y)
    torque = [r["torque_nm"] for r in working]
    payload = dict(simulation_only=True, skew_pitches=args.skew, signed_rpm=SPEEDS.tolist(),
                   geometry="AIR56 24/18, bar width modulation 0.60", loaded_summary_sha256=file_sha256(source),
                   mesh_level=LEVELS[-1], rank=RANK, degree_between_mesh_operators=180/(3*SECTOR_NODES),
                   angles=TARGET_POINTS, record_center_s=RECORD_CENTER_S, working_torque_mean_nm=float(np.mean(torque)),
                   motional_voltage_max_v=motional, elapsed_s=perf_counter()-start, map_sha256=file_sha256(path))
    (args.out/"summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf8")


def trajectory(bank, initial, before, after, ramp_seconds, time):
    """Interpolate the measured FEM probe response on the imposed trajectory."""
    response = np.empty((len(initial), len(time), 2, 2), dtype=complex)
    phases = (2*np.arange(len(time))) % 6
    for column in (0, 1):
        stamp = time+column*PHASE_INTERVAL_S
        velocity, displacement = ramp_profile(stamp, RAMP_START_S, ramp_seconds, before, after)
        lower = np.searchsorted(SPEEDS, velocity, side="right")-1
        lower = np.clip(lower, 0, len(SPEEDS)-2)
        blend = (velocity-SPEEDS[lower])/(SPEEDS[lower+1]-SPEEDS[lower])
        angles = initial[:, None]+displacement
        channel = response[..., column]
        for phase in range(column, 6, 2):
            for index in np.unique(lower):
                chosen = (phases+column == phase) & (lower == index)
                if not chosen.any():
                    continue
                a = bank[SPEEDS[index]].maps[phase].sample(angles[:, chosen])[..., column]
                b = bank[SPEEDS[index+1]].maps[phase].sample(angles[:, chosen])[..., column]
                w = blend[chosen][None, :, None]
                channel[:, chosen] = (1-w)*a+w*b
    return phase_bins(real_features(response))


def prepared_signals(bank):
    rng = np.random.default_rng(SEED)
    steady_time = np.arange(round(max(WINDOWS)/SAMPLE_INTERVAL_S))*SAMPLE_INTERVAL_S
    steady = []
    for speed in SPEEDS:
        starts = rng.uniform(0., np.pi, STEADY_TRIALS)
        waveform = scheduled_observations(bank[speed], starts, np.full(len(starts), speed), steady_time, 100., 1.)
        steady.append(phase_bins(waveform))
    steady = np.concatenate(steady)
    time = np.arange(round(RECORD_S/SAMPLE_INTERVAL_S))*SAMPLE_INTERVAL_S
    transitions, profiles = [], []
    for level in (.1, .2, .5):
        for sign in (-1, 1):
            target = level*sign
            for event, initial, duration in (("start", 0., 1.2), ("reversal", -target, 2.4)):
                phases = rng.uniform(0., np.pi, TRANSITION_TRIALS)
                transitions.append(trajectory(bank, phases, initial, target, duration, time))
                profiles.append(dict(event=event, initial_rpm=initial, target_rpm=target,
                                     reference_s=RAMP_START_S+(duration/2 if initial else 0.)))
    transition = np.concatenate(transitions)
    return (steady, rng.standard_normal(steady.shape), rng.normal(0, .0001, (len(steady), 1, 12))), \
        (transition, rng.standard_normal(transition.shape), rng.normal(0, .0001, (len(transition), 1, 12))), profiles


def noisy(components, noise):
    clean, white, bias = components
    return clean+noise/np.sqrt(3)*white+bias


def steady_results(detector, examples, case):
    directions, estimate, probability = detector.estimate(examples[:, :detector.bins], case["noise_a"])
    individual, summary = [], []
    for i, speed in enumerate(SPEEDS):
        index = slice(i*STEADY_TRIALS, (i+1)*STEADY_TRIALS)
        direction, rpm, move_p = directions[index], estimate[index], probability[index]
        sign = np.sign(speed)
        right = direction == sign
        errors = abs(rpm[right]-speed)
        summary.append(dict(**case, true_rpm=speed, trials=len(direction), correct=int(right.sum()),
                            rest=int(np.sum(direction==0)), moving=int(np.sum(abs(direction)==1)),
                            wrong_direction=int(np.sum(direction==-sign)) if speed else 0,
                            uncertain=int(np.sum(direction==UNCERTAIN)),
                            p95_rpm=float(np.quantile(errors,.95)) if len(errors) else None))
        individual.extend(dict(**case, true_rpm=speed, repeat=i,
                               direction=int(d) if d!=UNCERTAIN else None, estimated_rpm=n,
                               probability_motion=p) for i,(d,n,p) in enumerate(zip(direction,rpm,move_p)))
    return individual, summary


def transition_outcome(labels, times, profile):
    before, after, reference = profile["initial_rpm"], profile["target_rpm"], profile["reference_s"]
    sign = np.sign(after)
    # Confirmation requires two consecutive decisions. They are overlapping
    # causal windows, not independent pieces of evidence.
    confirmed = np.r_[UNCERTAIN, np.where(labels[:-1]==labels[1:], labels[1:], UNCERTAIN)]
    after_reference = np.r_[False, times[:-1]>=reference-1e-9]
    candidate = times[after_reference & (confirmed==sign)]
    delay = float(candidate[0]-reference) if candidate.size else None
    prior = (times<=RAMP_START_S+1e-9)
    prior_right = bool(np.any(prior & (confirmed==np.sign(before))))
    false_prestart = bool(np.any(prior & (abs(confirmed)==1))) if not before else False
    stale = times[(times>=reference-1e-9) & (confirmed==-sign)]
    stale_seconds = float(stale[-1]-reference) if stale.size else None
    return delay, prior_right, false_prestart, stale_seconds


def transition_results(detector, examples, profiles, case):
    windows = np.lib.stride_tricks.sliding_window_view(examples, detector.bins, axis=1)
    windows = windows[:, ::DECISION_STRIDE].swapaxes(-1,-2)
    times = (detector.bins+DECISION_STRIDE*np.arange(windows.shape[1]))*BIN_SECONDS
    decisions = detector.estimate(windows.reshape(-1, detector.bins,12), case["noise_a"])[0]
    decisions = decisions.reshape(len(examples), len(times))
    individual, summary = [], []
    for profile, group in zip(profiles, np.split(decisions, len(profiles))):
        result = [transition_outcome(row, times, profile) for row in group]
        delays = [d for d, _, _, _ in result if d is not None]
        old = [s for _, _, _, s in result if s is not None]
        summary.append(dict(**case, **profile, trials=len(group), new_sign_confirmed=len(delays),
                            prior_state_confirmed=sum(p for _,p,_,_ in result),
                            both_confirmed=sum(p and d is not None for d,p,_,_ in result),
                            false_motion_before_start=sum(f for _,_,f,_ in result),
                            wrong_sign_after_reference=len(old),
                            delay_median_s=float(np.median(delays)) if delays else None,
                            delay_p95_s=float(np.quantile(delays,.95)) if delays else None,
                            wrong_sign_last_p95_s=float(np.quantile(old,.95)) if old else None))
        individual.extend(dict(**case, **profile, repeat=i, delay_s=d, prior_state_confirmed=p,
                               false_motion_before_start=f, wrong_sign_last_s=s)
                          for i,(d,p,f,s) in enumerate(result))
    return individual, summary


def read_bank(source):
    summary_path = source/"summary.json"
    info = json.loads(summary_path.read_text(encoding="utf8"))
    path = source/"near_zero_pwm.npz"
    if file_sha256(path)!=info["map_sha256"]:
        raise ValueError("near-zero PWM map checksum mismatch")
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_allclose(data["signed_rpm"], SPEEDS, atol=0, rtol=0)
        bank = {v: phase_current_map(y,float(data["offset_rad"]))
                for v,y in zip(SPEEDS,data["admittance_a_per_v"])}
    return info, bank


def check_ramps(args):
    start = perf_counter()
    source = args.loaded_dir/"summary.json"
    prior = json.loads(source.read_text(encoding="utf8"))
    info, bank = read_bank(args.bank)
    if info["loaded_summary_sha256"]!=file_sha256(source):
        raise ValueError("ramp and bank must use the same load")
    skew = info["skew_pitches"]
    g, closed = article_profile()
    peak = prior["current_peak_a"]["bar_width"]
    sections, fem_offset, _ = build_sections(g, replace(closed,bar_width_modulation=.60), peak, mesh_index=1)
    theta = np.deg2rad(33.75)
    duration, ramp_start = 7.2, 2.4
    periods = round(duration/PWM_PERIOD_S)
    centers = np.array([1.8,2.3,2.6,3.,3.3,3.6,3.9,4.3,4.9,6.])
    lines = []
    for event, before, after, ramp_seconds in (("start", 0., .5, 1.2),("reversal", .5, -.5, 2.4)):
        print(f"Skew {skew}, full {duration:g} s switched waveform: {event}", flush=True)
        _, distance = ramp_profile(np.linspace(0,duration,80), ramp_start, ramp_seconds, before,after)
        step = np.pi/(3*SECTOR_NODES)
        low = int(np.floor((theta+distance.min()-fem_offset)/step))-1-2*skew
        high = int(np.ceil((theta+distance.max()-fem_offset)/step))+1+2*skew
        model = common_model(sections, g.rotor_radius, RANK, fem_offset, tuple(range(low,high+1)))
        system = moving_circuit(model, g.stack_length, skew, PHASE_NODES, AXIAL_MODES)
        pwm = [moving_run(*system,theta,before,peak,axis,periods,MOTION_BLOCK,0,ramp_start,ramp_seconds,after)
               for axis in (-1,0,1)]
        raw = np.array([r[0] for r in pwm])
        speeds, positions = ramp_profile(centers,ramp_start,ramp_seconds,before,after)
        probes = []
        for when,speed,position in zip(centers,speeds,positions):
            signal = demodulate(raw,periods,center_s=when)
            i = int(np.clip(np.searchsorted(SPEEDS,speed,side="right")-1,0,len(SPEEDS)-2))
            weight = (speed-SPEEDS[i])/(SPEEDS[i+1]-SPEEDS[i])
            y0,y1 = (np.array([m.sample(theta+position) for m in bank[SPEEDS[j]].maps]) for j in (i,i+1))
            approximation = (1-weight)*y0+weight*y1
            rms = float(np.sqrt(np.mean(abs(signal-approximation)**2)))
            probes.append(signal)
            lines.append(dict(skew_pitches=skew,event=event,time_s=when,speed_rpm=speed,
                              angle_deg=float(np.rad2deg(theta+position)),bank_vs_continuous_rms_a_per_v=rms))
        np.savez_compressed(args.out/f"{event}_pwm.npz", sample_period_s=PWM_PERIOD_S,
                            period_average_currents_a=raw, sample_times_s=centers,
                            carrier_response_a_per_v=np.array(probes), true_rpm_at_samples=speeds)
    write_csv(args.out/"ramp_checks.csv",lines)
    payload = dict(simulation_only=True, skew_pitches=skew, duration_s=duration,
                   bank_sha256=info["map_sha256"], elapsed_s=perf_counter()-start, checks=lines,
                   files_sha256={p.name:file_sha256(p) for p in args.out.iterdir() if p.suffix in (".csv",".npz")})
    (args.out/"summary.json").write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n",encoding="utf8")


def score(args):
    start = perf_counter()
    previous_path = args.calibration/"summary.json"
    previous = json.loads(previous_path.read_text(encoding="utf8"))
    if previous["profile"] != "bar_width":
        raise ValueError("near-zero study requires the bar-width calibration")
    steady, steady_summary, transitions, transition_summary = [], [], [], []
    sources = {}
    for source in args.banks:
        summary_path = source/"summary.json"
        info, bank = read_bank(source)
        skew = info["skew_pitches"]
        if skew in sources or info["loaded_summary_sha256"]!=previous["prior_sha256"]:
            raise ValueError("use one near-zero bank for each skew from the same loaded geometry")
        sources[skew] = file_sha256(summary_path)
        calibration = read_maps(args.calibration, previous, previous["levels"][0], skew)[0.]
        fixed_examples, dynamic_examples, profiles = prepared_signals(bank)
        for length in WINDOWS:
            print(f"Skew {skew}: decoding {length:g} s windows", flush=True)
            detector = Detector.build(calibration, SAMPLE_INTERVAL_S, scheduled_observations, round(length/BIN_SECONDS))
            for noise in NOISE:
                case = dict(skew_pitches=skew, noise_a=noise, window_s=length)
                rows, rows_summary = steady_results(detector, noisy(fixed_examples,noise), case)
                steady.extend(rows)
                steady_summary.extend(rows_summary)
                if length in TRANSITION_WINDOWS:
                    rows, rows_summary = transition_results(detector, noisy(dynamic_examples,noise), profiles, case)
                    transitions.extend(rows)
                    transition_summary.extend(rows_summary)
    if set(sources)!={0,1}:
        raise ValueError("both skew banks must be supplied")
    tables = dict(steady_trials=steady, steady_results=steady_summary,
                  transition_trials=transitions, transition_results=transition_summary)
    for name, data in tables.items():
        write_csv(args.out/f"{name}.csv",data)
    payload = dict(simulation_only=True, calibration_sha256=file_sha256(previous_path), banks_sha256=sources,
                   noise_a=NOISE, windows_s=WINDOWS, transition_windows_s=TRANSITION_WINDOWS, observation_bin_s=BIN_SECONDS,
                   moving_threshold_p=.95, decisions_stride_s=BIN_SECONDS*DECISION_STRIDE,
                   stationary_class_rpm=0., fixed_trial_count=len(steady), transition_trial_count=len(transitions),
                   elapsed_s=perf_counter()-start, steady=steady_summary, transitions=transition_summary,
                   files_sha256={f"{n}.csv":file_sha256(args.out/f"{n}.csv") for n in tables})
    (args.out/"summary.json").write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n",encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    maps = commands.add_parser("maps")
    maps.add_argument("--skew", type=int, choices=(0, 1), required=True)
    maps.add_argument("--loaded-dir", type=Path, required=True)
    observe = commands.add_parser("score")
    observe.add_argument("--banks", nargs=2, type=Path, required=True)
    observe.add_argument("--calibration", type=Path, required=True)
    check = commands.add_parser("check-ramps")
    check.add_argument("--bank", type=Path, required=True)
    check.add_argument("--loaded-dir", type=Path, required=True)
    for command in (maps,observe,check):
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/"summary.json").exists():
        parser.error("choose a new output directory")
    {"maps":build_maps,"score":score,"check-ramps":check_ramps}[args.command](args)


if __name__ == "__main__":
    main()
