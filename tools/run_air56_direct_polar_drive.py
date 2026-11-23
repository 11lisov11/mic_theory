"""Current-only sampled direct commutation of a passive polar salient cage in a nonlinear dq plant.

The plant integrates a shorted rotor circuit. The controller works in stationary
coordinates. Its inputs are two stator currents and its commanded voltages.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_air56_skew_audit import write_csv

H = 0.00025
LM = 0.45
LR = 0.006
RR = 14.0
RS = 9.17
LD = 0.043
LQ = 0.133
K = 0.0035
PSI = 0.04
J = 0.0001662
DRAG = 0.000179


@njit(cache=True)
def clip(x, low, high):
    return min(high, max(low, x))


@njit(cache=True)
def dynamics(x, ua, ub, load, scale, mag, drag, inertia, skew=0):
    d, q, rd, rq, theta, w = x
    c, s = np.cos(theta), np.sin(theta)
    # Analytic axial averaging for the first and second spatial harmonics.
    # Higher odd saturation harmonics retain the first-harmonic scale here.
    attenuation = 0.979815536 if skew else 1.0
    polar = 0.994930770 * mag if skew else mag
    ld = (LD + LQ + (LD - LQ) * attenuation) * 0.5 * scale
    lq = (LD + LQ - (LD - LQ) * attenuation) * 0.5 * scale
    ls_d = ld + K * polar * np.tanh(d)
    lsd = (LM + ld) * d + K * polar * np.log(np.cosh(min(20.0, abs(d)))) + LM * rd + PSI * polar
    lsq = (LM + lq) * q + LM * rq
    torque = 1.5 * (lsd * q - lsq * d)
    ratio = LM / (LM + LR)
    dd = (c * ua + s * ub - RS * d + w * lsq + ratio * RR * rd) / (ls_d + ratio * LR)
    dq = (-s * ua + c * ub - RS * q - w * lsd + ratio * RR * rq) / (lq + ratio * LR)
    return np.array(
        [
            dd,
            dq,
            (-RR * rd - LM * dd) / (LM + LR),
            (-RR * rq - LM * dq) / (LM + LR),
            w,
            (torque - load - drag * w) / inertia,
        ]
    )


@njit(cache=True)
def advance(x, ua, ub, load, scale, mag, drag, inertia, dt, skew):
    mid = x + 0.5 * dt * dynamics(x, ua, ub, load, scale, mag, drag, inertia, skew)
    return x + dt * dynamics(mid, ua, ub, load, scale, mag, drag, inertia, skew)


@njit(cache=True)
def bridge(x, ua, ub, load, scale, mag, drag, inertia, switched, skew):
    if not switched:
        return advance(x, ua, ub, load, scale, mag, drag, inertia, H, skew)
    # Space-vector dwell times for a two-level 310 V bus. Symmetric zero/active/active
    # sequence inside each sample. No direct access to shaft state in modulator.
    length = np.hypot(ua, ub)
    angle = np.arctan2(ub, ua) % (2 * np.pi)
    sector = int(angle / (np.pi / 3))
    offset = angle - sector * np.pi / 3
    t1 = H * np.sqrt(3) * length / 310 * np.sin(np.pi / 3 - offset)
    t2 = H * np.sqrt(3) * length / 310 * np.sin(offset)
    t0 = max(0.0, H - t1 - t2)
    a = sector * np.pi / 3
    v = 310 * 2 / 3
    for i in range(5):
        if i == 0 or i == 4:
            p, q, dt = 0.0, 0.0, t0 / 2
        elif i == 1 or i == 3:
            p, q, dt = v * np.cos(a), v * np.sin(a), t1 / 2
        else:
            p, q, dt = v * np.cos(a + np.pi / 3), v * np.sin(a + np.pi / 3), t2
        x = advance(x, p, q, load, scale, mag, drag, inertia, dt, skew)
    return x


@njit(cache=True)
def read_angle(samples, prior, fresh, estimator_k):
    # Each record has two current midpoints, commanded voltages and the measured
    # difference of the two increments. Smooth flux/back-EMF terms cancel to first order.
    n = len(samples)
    best = 1e30
    theta = prior
    span = 180.0 if fresh else 10.0
    step = 1.0 if fresh else 0.25
    for j in range(1 + int(2 * span / step)):
        trial = prior + np.deg2rad(j * step - span)
        c, s = np.cos(trial), np.sin(trial)
        loss = 0.0
        for row in samples:
            pa, pb = 0.0, 0.0
            for i in range(2):
                a, b = row[4 * i], row[4 * i + 1]
                va, vb = row[4 * i + 2] - 22.0 * a, row[4 * i + 3] - 22.0 * b
                id_ = a * c + b * s
                slope_d = (va * c + vb * s) / (LD + LM * LR / (LM + LR) + estimator_k * np.tanh(id_))
                slope_q = (-va * s + vb * c) / (LQ + LM * LR / (LM + LR))
                sign = 1 - 2 * i
                pa += sign * H * (c * slope_d - s * slope_q)
                pb += sign * H * (s * slope_d + c * slope_q)
            loss += (row[8] - pa) ** 2 + (row[9] - pb) ** 2
        if loss < best:
            best, theta = loss, trial
    return theta


@njit(cache=True)
def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


@njit(cache=True)
def torque_flux_rate(state, ua, ub):
    d, q, rd, rq, theta, w = state
    ld = LD + LM + K * np.tanh(d)
    ps = (LD + LM) * d + K * np.log(np.cosh(min(20.0, abs(d)))) + LM * rd + PSI
    qs = (LQ + LM) * q + LM * rq
    ds = dynamics(state, ua, ub, 0.0, 1.0, 1.0, DRAG, J)
    dp = ld * ds[0] + LM * ds[2]
    qp = (LQ + LM) * ds[1] + LM * ds[3]
    fl = max(0.015, np.hypot(ps, qs))
    return (
        np.array([1.5 * (dp * q + ps * ds[1] - qp * d - qs * ds[0]), (ps * dp + qs * qp) / fl]),
        1.5 * (ps * q - qs * d),
        fl,
    )


@njit(cache=True)
def direct_torque_voltage(state, torque, flux):
    zero, actual, amplitude = torque_flux_rate(state, 0.0, 0.0)
    ac = torque_flux_rate(state, 1.0, 0.0)[0] - zero
    bc = torque_flux_rate(state, 0.0, 1.0)[0] - zero
    desired = np.array([70.0 * (torque - actual), 45.0 * (flux - amplitude)]) - zero
    det = ac[0] * bc[1] - ac[1] * bc[0]
    if abs(det) < 0.01:
        return 0.0, 0.0
    ua = (bc[1] * desired[0] - bc[0] * desired[1]) / det
    ub = (-ac[1] * desired[0] + ac[0] * desired[1]) / det
    lim = min(1.0, 65.0 / max(1.0, np.hypot(ua, ub)))
    return ua * lim, ub * lim


@njit(cache=True)
def target(t, base, turns):
    # Absolute, externally prescribed setpoint. Includes a positioning move, holds
    # and two long +/-0.1 rpm segments, independent of any hidden plant variable.
    if turns:
        if t < 5.0:
            return base, 0.0
        if t < 18.0:
            return base + 0.55 * (t - 5.0), 0.55
        if t < 20.0:
            return base + 7.15, 0.0
        if t < 33.0:
            return base + 0.55 * (33.0 - t), -0.55
        return base, 0.0
    crawl = 0.1 * np.pi / 30
    if t < 1.0:
        return base, 0.0
    if t < 5.0:
        return base + 0.08 * (t - 1), 0.08
    if t < 9.0:
        return base + 0.32, 0.0
    if t < 21.0:
        return base + 0.32 + crawl * (t - 9), crawl
    if t < 33.0:
        return base + 0.32 + crawl * (33 - t), -crawl
    return base + 0.32, 0.0


@njit(cache=True)
def simulate(
    theta0,
    base,
    noise,
    seed,
    scale,
    mag,
    load,
    switched,
    oracle,
    estimator_k,
    turns,
    torque_mode=False,
    imperfect=False,
    tune=1.0,
    skew=0,
):
    np.random.seed(seed)
    x = np.array([0.0, 0.0, 0.0, 0.0, theta0, 0.0])
    pairs = 24
    block = 2 * pairs
    records = np.zeros((pairs, 10))
    startup = np.zeros((pairs * 8, 10))
    steps = int(38 / H)
    trace = np.zeros((steps // block, 11))
    current_sum = 0.0
    current_max = 0.0
    last = np.zeros(2)
    start = np.zeros(2)
    measured = np.zeros(2)
    previous_mid = np.zeros(2)
    previous_u = np.zeros(2)
    filt = np.zeros(2)
    ua0, ub0 = 0.0, 0.0
    estimate, speed, integrator = 0.0, 0.0, 0.0
    rotor_flux = np.zeros(2)
    disturbance = 0.0
    last_speed = 0.0
    recent_torque = 0.0
    torque_request = 0.0
    servo_ref = 0.0
    dtc = torque_mode
    acquired = False
    ktrace = 0
    for k in range(steps):
        t = k * H
        if k % block == 0:
            ref, ref_w = target(t, base, turns)
            if 0 < k <= block * 8:
                idx = k // block - 1
                startup[idx * pairs : (idx + 1) * pairs] = records
            if k >= block * 8:
                data = records if acquired else startup
                new = (
                    read_angle(data, estimate + block * H * speed, not acquired, estimator_k)
                    if not oracle
                    else x[4]
                )
                if acquired:
                    # alpha-beta tracker, unwrap relative to predicted angle.
                    predicted = estimate + speed * block * H
                    delta = wrap(new - predicted)
                    estimate = predicted + 0.80 * delta
                    speed += 0.20 * delta / (block * H)
                else:
                    estimate, speed = new, 0.0
                    acquired = True
                    servo_ref = estimate
                trace[ktrace] = np.array(
                    [
                        t,
                        x[4],
                        estimate,
                        x[5],
                        speed,
                        ref,
                        ref_w,
                        integrator,
                        np.hypot(x[0], x[1]),
                        np.sqrt(current_sum / block),
                        current_max,
                    ]
                )
                ktrace += 1
            current_sum = 0.0
            current_max = 0.0
            # Direct stationary-axis current command. Small integral rejects a
            # constant load. No flux angle, speed sensor or rotating current PI.
            if dtc and t < 0.7:
                servo_ref = estimate
            elif acquired:
                ref_step = clip(wrap(ref - servo_ref), -1.1 * block * H, 1.1 * block * H)
                servo_ref += ref_step
                ref_w = ref_step / (block * H)
            error = wrap(servo_ref - estimate)
            if not acquired:
                ia, ib = 0.0, 0.0
            elif not dtc:
                integrator = clip(integrator + block * H * 0.014 * error, -0.025, 0.025)
                tau = clip(0.018 * error + 0.0030 * (ref_w - speed) + integrator, -0.040, 0.040)
                amp = tau / (1.5 * PSI)
                ia = -amp * np.sin(estimate)
                ib = amp * np.cos(estimate)
            elif t < 0.7:
                ia = 0.9 * np.cos(estimate)
                ib = 0.9 * np.sin(estimate)
                integrator = 0.0
            else:
                disturbance = clip(
                    disturbance
                    + block * H / 0.12 * (recent_torque - DRAG * speed - disturbance)
                    - J * (speed - last_speed) / 0.12,
                    -1.0,
                    1.0,
                )
                last_speed = speed
                integrator = clip(integrator + block * H * 0.85 * tune * error, -0.65, 0.65)
                torque_request = clip(
                    0.24 * np.sqrt(tune) * error
                    + 0.012 * np.sqrt(tune) * (ref_w - speed)
                    + integrator
                    + 0.75 * disturbance,
                    -1.3,
                    1.3,
                )
        if k % 2 == 0:
            pair = k // 2
            a = (pair % 3) * np.pi / 3
            sign = 1 if (pair // 3) % 2 == 0 else -1
            pa, pb = sign * 82.0 * np.cos(a), sign * 82.0 * np.sin(a)
            ua0 = clip(48 * (ia - filt[0]) + RS * ia, -45.0, 45.0)
            ub0 = clip(48 * (ib - filt[1]) + RS * ib, -45.0, 45.0)
            if dtc and acquired:
                c, s = np.cos(estimate), np.sin(estimate)
                cur_dq = np.array([c * filt[0] + s * filt[1], -s * filt[0] + c * filt[1]])
                rotor_i = (rotor_flux - LM * cur_dq) / (LM + LR)
                rotor_flux -= 2 * H * RR * rotor_i
                if t >= 0.7:
                    state = np.array([cur_dq[0], cur_dq[1], rotor_i[0], rotor_i[1], estimate, speed])
                    ua0, ub0 = direct_torque_voltage(state, torque_request, 0.48)
                    recent_torque = torque_flux_rate(state, 0.0, 0.0)[1]
        ua = ua0 + (1 if k % 2 == 0 else -1) * pa
        ub = ub0 + (1 if k % 2 == 0 else -1) * pb
        applied_load = load * (clip((t - 6.0) / 3.0, 0.0, 1.0) if dtc else (1.0 if t > 6 else 0.0))
        va, vb = ua, ub
        if imperfect:
            # Independent plant-side voltage error, sensing gain and offsets.
            va = 0.985 * ua - 0.35 * np.sign(last[0])
            vb = 0.985 * ub - 0.35 * np.sign(last[1])
        x = bridge(x, va, vb, applied_load, scale, mag, DRAG, J, switched, skew)
        c, s = np.cos(x[4]), np.sin(x[4])
        measured = np.array([x[0] * c - x[1] * s, x[0] * s + x[1] * c]) + noise * np.random.randn(2)
        if imperfect:
            measured[0] = 1.005 * measured[0] + 0.0015
            measured[1] = 0.995 * measured[1] - 0.001
        current_sum += x[0] ** 2 + x[1] ** 2
        current_max = max(current_max, np.hypot(x[0], x[1]))
        mid = (measured + last) / 2
        if k % 2 == 0:
            start = last.copy()
            previous_mid = mid.copy()
            previous_u = np.array([ua, ub])
        else:
            row = (k % block) // 2
            records[row, 0:2] = previous_mid
            records[row, 2:4] = previous_u
            records[row, 4:6] = mid
            records[row, 6:8] = np.array([ua, ub])
            records[row, 8:10] = 2 * last - start - measured
            gain = 0.40 if dtc else 0.12
            filt = (1 - gain) * filt + gain * (mid + previous_mid) / 2
        last = measured.copy()
    return trace[:ktrace]


def summarize(trace):
    t, theta, est, w, west, ref, rw, integ, amp = trace.T[:9]
    e = np.rad2deg(np.angle(np.exp(1j * (est - theta))))
    tracking = np.rad2deg(np.angle(np.exp(1j * (ref - theta))))
    result = dict(
        first_error_deg=abs(e[0]),
        max_observer_deg=abs(e).max(),
        p95_observer_deg=np.percentile(abs(e), 95),
        p95_tracking_deg=np.percentile(abs(tracking[t > 5]), 95),
        max_current_sample_a=amp.max(),
        final_tracking_deg=tracking[-1],
    )
    if trace.shape[1] > 9:
        result["peak_current_vector_a"] = trace[:, 10].max()
        result["mean_phase_rms_loaded_a"] = np.sqrt(np.mean(trace[t > 9, 9] ** 2) / 2)
    result["max_rest_excursion_deg"] = np.rad2deg(np.ptp(theta[t >= 35.0]))
    for label, a, b in (("forward", 12.0, 20.0), ("reverse", 24.0, 32.0), ("rest", 35.0, 38.0)):
        mask = (t >= a) & (t < b)
        # Mean speed from net travel avoids counting probe-induced speed ripple as useful motion.
        result[label + "_actual_rpm"] = np.polyfit(t[mask], theta[mask], 1)[0] * 30 / np.pi
        result[label + "_read_rpm"] = np.polyfit(t[mask], est[mask], 1)[0] * 30 / np.pi
    for label, start, end in (("out", 5.0, 18.0), ("return", 20.0, 33.0)):
        i, j = np.searchsorted(t, [start, end])
        result[label + "_actual_turns"] = (theta[j] - theta[i]) / (2 * np.pi)
        result[label + "_read_turns"] = (est[j] - est[i]) / (2 * np.pi)
    frames = round(1 / (t[1] - t[0]))
    read = (est[frames:] - est[:-frames]) / (frames * (t[1] - t[0])) * 30 / np.pi
    actual = (theta[frames:] - theta[:-frames]) / (frames * (t[1] - t[0])) * 30 / np.pi
    result["p95_one_second_speed_error_rpm"] = np.percentile(abs(read - actual), 95)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    rows = []
    for seed in range(1 if args.smoke else 8):
        theta0 = 2 * np.pi * (seed + 0.37) / 8
        base = 1.20
        for mode, scale, mag, load, sw, oracle, k in (
            ("switched", 1.0, 1.0, 0.002, True, False, K),
            ("perturbed", 1.02, 0.85, 0.002, True, False, K),
            ("low_noise", 1.0, 1.0, 0.002, True, False, K),
            ("one_ma", 1.0, 1.0, 0.002, True, False, K),
            ("higher_load", 1.0, 1.0, 0.020, True, False, K),
            ("turn_count", 1.0, 1.0, 0.002, True, False, K),
            ("ideal_feedback", 1.0, 1.0, 0.002, True, True, K),
            ("no_polarity", 1.0, 1.0, 0.002, True, False, 0.0),
        ):
            if args.smoke and mode != "switched":
                continue
            noise = {"low_noise": 0.0003, "one_ma": 0.001}.get(mode, 0.003)
            trace = simulate(
                theta0, base, noise, 8711 + seed, scale, mag, load, sw, oracle, k, mode == "turn_count"
            )
            rows.append(dict(seed=seed, mode=mode, **summarize(trace)))
            if seed == 0:
                np.savez_compressed(args.out / f"drive_{mode}.npz", trace=trace)
            print(seed, mode, rows[-1]["first_error_deg"], rows[-1]["p95_tracking_deg"], flush=True)
    write_csv(args.out / ("drive_smoke.csv" if args.smoke else "drive.csv"), rows)
