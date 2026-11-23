"""Moving nonlinear P1 field, shared winding currents, and terminal-current feedback.

The plant and observer retain separate electromagnetic and mechanical states.
The actuator sees the observer state and terminal ADC only.
"""

from pathlib import Path
import argparse, json, sys, time
import numpy as np
from scipy.sparse import bmat, csc_matrix
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fem.air56.fourier_trace import TraceField
from fem.air56.p1_volume import P1Volume
from fem.air56.polar_start import capture_quality, resolving_pulses, resolve
from fem.air56.servo_pwm import segments, segment_tangents
from tools.run_air56_free_field import R, SkewBand, BandField, wrap
from tools.run_air56_field_observer import J, DRAG


class Machine:
    def __init__(self, level, modes, skew, ring=False):
        self.ring = ring
        poles = dict(pole_half_width_deg=90.0, remanence_t=0.8) if ring else {}
        self.slices = [
            (TraceField(level, modes, **poles) if modes else BandField(0.0, level, **poles))
            for _ in range(3 if skew else 1)
        ]
        self.offsets = SkewBand.offsets if skew else [0.0]
        self.n = len(self.slices[0].free)
        self.count = len(self.slices)
        self.size = (self.n + 1) * self.count + 2
        self.trapezoid = False
        self.torque_calibration = None
        self.current_angle = None
        self.factor_cache = {}
        self.volume = P1Volume(self.slices[0])

    def phase(self, angle):
        if self.current_angle == angle:
            return
        self.current_angle = angle
        for f, off in zip(self.slices, self.offsets):
            f.update(angle + off)

    def parts(self, x):
        return x[: self.size - 2].reshape(self.count, self.n + 1, -1)

    def equilibrium(self, angle, current=None):
        self.phase(angle)
        local = [f.equilibrium(current) for f in self.slices]
        torque = np.mean([f.torque(s[:-3]) for f, s in zip(self.slices, local)])
        return np.r_[np.concatenate([s[:-2] for s in local]), torque, local[0][-2:], angle, 0.0, 0.0]

    def electromagnetic(self, x):
        return np.concatenate((x[: self.size - 2], x[-5:-3]), axis=0)

    def evaluate(self, s, old, u, h, linearize):
        residual = []
        circuit = R * s[-2:] - u
        factors = []
        for j, (f, z) in enumerate(zip(self.slices, s[:-2].reshape(self.count, -1))):
            a = z[:-1]
            rate = (a - old[j]) / h
            force, t = self.volume.field(a)
            force += f.extra @ a - f.pm
            residual.append(
                np.r_[force + f.mass @ rate - f.g * z[-1] - f.W @ s[-2:], f.g @ rate - f.total * z[-1]]
            )
            circuit = circuit + (f.lam / self.count) * (f.W.T @ rate)
            if linearize:
                k = bmat(
                    [
                        [self.volume.matrix(t) + f.extra + f.mass / h, -f.floating],
                        [f.floating.T / h, csc_matrix([[-f.total]])],
                    ],
                    format="csc",
                )
                lu = splu(k, permc_spec="MMD_AT_PLUS_A")
                coupling = np.vstack([f.W, np.zeros((1, 2))])
                factors.append((lu, lu.solve(coupling)))
        return np.r_[np.concatenate(residual), circuit], factors

    def solve(self, factors, rhs, h):
        rhs = np.asarray(rhs).reshape(self.size, -1)
        right = rhs[-2:].copy()
        k = R * np.eye(2)
        pieces = []
        for j, (f, (lu, c)) in enumerate(zip(self.slices, factors)):
            y = lu.solve(self.parts(rhs)[j])
            response = (f.lam / (self.count * h)) * f.W.T
            right -= response @ y[:-1]
            k += response @ c[:-1]
            pieces.append(y)
        di = np.linalg.solve(k, right)
        return np.vstack([*(y + c @ di for y, (_, c) in zip(pieces, factors)), di])

    def step(self, x, u, h, load=None, tangent=False, guess=None):
        theta, w, dist = x[-3:]
        self.phase(theta)
        old = self.parts(x)[:, :-1, 0]
        s = self.electromagnetic(x) if guess is None else guess.copy()
        bucket = round(np.log2(h) * 4)
        if bucket not in self.factor_cache and len(self.factor_cache) > 12:
            self.factor_cache.pop(next(iter(self.factor_cache)))
        cached = self.factor_cache.get(bucket)
        factors = cached[1] if cached and abs(wrap(theta - cached[0])) < 0.04 else None
        res = self.evaluate(s, old, u, h, False)[0]
        exact = False
        for _ in range(50):
            norm = np.linalg.norm(res)
            if np.max(abs(res)) < 1e-6:
                if tangent:
                    _, factors = self.evaluate(s, old, u, h, True)
                    self.factor_cache[bucket] = (theta, factors)
                break
            if factors is None:
                _, factors = self.evaluate(s, old, u, h, True)
                self.factor_cache[bucket] = (theta, factors)
                exact = True
            direction = self.solve(factors, -res, h)[:, 0]
            for scale in 2.0 ** -np.arange(13 if exact else 3):
                test = s + scale * direction
                next_res = self.evaluate(test, old, u, h, False)[0]
                progress = np.linalg.norm(next_res) / norm
                if progress < 1.0:
                    s = test
                    res = next_res
                    if progress > 0.55:
                        factors = None
                    exact = False
                    break
            else:
                if exact:
                    raise RuntimeError("field line search")
                factors = None
        else:
            raise RuntimeError("field iterations")
        torque = np.mean([f.torque(z[:-1, 0]) for f, z in zip(self.slices, self.parts(s))])
        if self.torque_calibration is not None:
            torque += (
                sum(self.torque_calibration.evaluate(theta + off, s[-2:])[0] for off in self.offsets)
                / self.count
            )
        # The algebraic magnetic field belongs to the geometry of the last
        # converged electrical step. Carry its endpoint torque across rotation.
        end_torque = torque
        angle, speed, torque = self.mechanics(
            theta, w, dist if load is None else load, x[-6], end_torque, h
        )
        after = np.r_[s[:-2], end_torque, s[-2:], angle, speed, dist]
        return after, torque, (factors if tangent else None)

    def differentiate(self, old, new, factors, dx, du, h, dh=None):
        # All columns use the same converged implicit winding/field Jacobian.
        rhs = np.zeros((self.size, dx.shape[1]))
        rhs[-2:] = du
        torque = np.zeros(dx.shape[1])
        derivatives = []
        for j, (f, z, da) in enumerate(zip(self.slices, self.parts(new), self.parts(dx))):
            a = z[:-1, 0]
            hist = da[:-1]
            if dh is not None:
                hist = hist + (a - self.parts(old)[j, :-1, 0])[:, None] * (dh / h)
            g = 1j * f.orders * f.rot.conj() * f.coupling
            v = a[f.nodes]
            outer = f.outer @ v[f.count :]
            inner = f.inner @ v[: f.count]
            dk = np.zeros_like(a)
            dk[f.nodes] = np.r_[
                -np.real(f.inner.conj().T @ (g * outer)), -np.real(f.outer.conj().T @ (g.conj() * inner))
            ]
            q = self.parts(rhs)[j]
            q[:-1] = f.mass @ hist / h - dk[:, None] * dx[-3]
            q[-1] = f.g @ hist / h
            rhs[-2:] += (f.lam / (self.count * h)) * (f.W.T @ hist)
            fixed = -f.length * np.sum(f.orders**2 * f.coupling * np.real((f.rot * inner).conj() * outer))
            torque += fixed * dx[-3] / self.count
            derivatives.append(-f.length * dk / self.count)
        electrical = self.solve(factors, rhs, h)
        for j, v in enumerate(derivatives):
            torque += v @ self.parts(electrical)[j, :-1, :]
        if self.torque_calibration is not None:
            for offset in self.offsets:
                _, angular, axes = self.torque_calibration.evaluate(old[-3] + offset, new[-5:-3])
                torque += (angular * dx[-3] + axes @ electrical[-2:]) / self.count
        endpoint = torque
        angle, speed, torque = self.mechanics(dx[-3], dx[-2], dx[-1], dx[-6], endpoint, h)
        if dh is not None:
            if self.trapezoid:
                mean_torque = 0.5 * (old[-6] + new[-6])
                rate = (mean_torque - old[-1] - 0.5 * DRAG * (old[-2] + new[-2])) / (J + 0.5 * h * DRAG)
                turn_rate = (
                    0.5 * (old[-2] + new[-2] + h * rate)
                    + h / (6 * J) * (old[-6] - new[-6] + DRAG * (new[-2] - old[-2]))
                    + h * h * DRAG * rate / (12 * J)
                )
            else:
                rate = (new[-6] - old[-1] - DRAG * old[-2]) / J
                turn_rate = new[-2] + h * rate
            speed += rate * dh
            angle += turn_rate * dh
        return np.vstack([electrical[:-2], endpoint, electrical[-2:], angle, speed, dx[-1]]), torque

    def mechanics(self, theta, speed, load, start_torque, end_torque, h):
        if not self.trapezoid:
            end_speed = speed + h * (end_torque - DRAG * speed - load) / J
            return theta + h * end_speed, end_speed, end_torque
        avg = (start_torque + end_torque) / 2
        end_speed = (speed + h * (avg - load - 0.5 * DRAG * speed) / J) / (1 + 0.5 * h * DRAG / J)
        # Integrate a linearly changing acceleration through the electrical interval.
        angle = (
            theta
            + h * (speed + end_speed) / 2
            + h * h * (start_torque - end_torque + DRAG * (end_speed - speed)) / (12 * J)
        )
        return angle, end_speed, avg


def posterior(x, factors, y, noise):
    measured = factors[-5:-3, :]
    innovation = y - x[-5:-3]
    mat = measured @ measured.T + noise**2 * np.eye(2)
    x = x + factors @ (measured.T @ np.linalg.solve(mat, innovation))
    # Right square-root update, avoiding any dense matrix of nodal covariance.
    eig, vec = np.linalg.eigh(np.eye(measured.shape[1]) - measured.T @ np.linalg.solve(mat, measured))
    factors = factors @ (vec * np.sqrt(np.maximum(0, eig))) @ vec.T
    return x, factors, float(innovation @ np.linalg.solve(mat, innovation))


def compress(columns, rank):
    # Use physical units to keep field, shaft and load uncertainty in the retained basis.
    scaled = columns.copy()
    scaled[:-5] *= 10.0 / np.sqrt(len(scaled))
    scaled[-6] = 0.0
    scaled[-3] *= 20.0
    scaled[-2] *= 0.2
    scaled[-1] *= 100.0
    _, vectors = np.linalg.eigh(scaled.T @ scaled)
    return columns @ vectors[:, -rank:]


def command(model, x, ref, wref, k, h, stiffness=0.32, integral=0.0, bias=None, probe_v=5.0):
    err = wrap(ref - x[-3])
    demand = x[-1] + stiffness * err + 0.014 * np.sqrt(stiffness / 0.32) * (wref - x[-2]) + integral
    demand = np.clip(demand, -0.12, 0.12)
    current = x[-5:-3]
    probe = probe_v * np.array([np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)])
    u = (R - 10.0) * current + probe
    if bias is not None:
        u = 0.3 * u + 0.7 * (bias + probe)
    cap = 155.0
    zeros = np.zeros((len(x), 2))
    guess = None
    best = None
    for iteration in range(6):
        try:
            pred, tau, fac = model.step(x, u, h, tangent=True, guess=guess)
        except RuntimeError:
            if best is not None:
                break
            pred, tau, fac = model.step(x, u, h, tangent=True)

        slope, grad = model.differentiate(x, pred, fac, zeros, np.eye(2), h)
        if model.trapezoid:
            # Regulate endpoint torque. Inverting the trapezoidal average would
            # reflect the preceding endpoint about the command every update.
            tau, grad = pred[-6], slope[-6]
        err = demand - tau
        if best is None or abs(err) < best[0]:
            best = (abs(err), u.copy(), pred, tau, fac)
        if abs(err) < 0.00010:
            break
        delta = grad * err / (grad @ grad + 3e-11)
        if iteration == 5:
            break
        prev = u.copy()
        u += delta * min(1.0, 35.0 / max(np.linalg.norm(delta), 1e-12))
        u *= min(1.0, cap / max(np.linalg.norm(u), 1e-12))
        guess = model.electromagnetic(pred) + model.electromagnetic(slope) @ (u - prev)
    return best[1:]


def predictive_pwm(model, state, ref, wref, k, args, integral, bias):
    demand = np.clip(
        state[-1]
        + args.stiffness * wrap(ref - state[-3])
        + 0.014 * np.sqrt(args.stiffness / 0.32) * (wref - state[-2])
        + integral,
        -0.12,
        0.12,
    )
    # A bias near the preceding solution discourages alternating large currents.
    probe = args.probe_v * np.array([np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)])
    u = (
        probe + (R - 8) * state[-5:-3]
        if bias is None
        else probe + 0.85 * bias + 0.15 * (R - 8) * state[-5:-3]
    )
    best = None
    for _ in range(4):
        trial = state.copy()
        sens = np.zeros((len(state), 2))
        impulse = 0.0
        grad = np.zeros(2)
        for _carrier in range(args.pwm):
            for voltage, h, dh in segment_tangents(u, args.h / args.pwm):
                old = trial
                trial, tau, fac = model.step(old, voltage, h, tangent=True)
                sens, dtau = model.differentiate(old, trial, fac, sens, np.zeros((2, 2)), h, dh)
                impulse += h * tau / args.h
                grad += (h * dtau + tau * dh) / args.h
        # Endpoint weighting damps the alternating endpoint-torque mode of
        # exact impulse inversion. The measured output remains terminal current.
        err = demand - (0.7 * impulse + 0.3 * trial[-6])
        grad = 0.7 * grad + 0.3 * sens[-6]
        if best is None or abs(err) < best[0]:
            best = (abs(err), u.copy(), trial, impulse)
        if abs(err) < 0.00015:
            break
        delta = err * grad / (grad @ grad + 1e-10)
        u += delta * min(1.0, 12.0 / max(1e-10, np.linalg.norm(delta)))
        u *= min(1.0, 80.0 / max(1e-10, np.linalg.norm(u)))
    return *best[1:], None


def profile(t, base, leg, start=1.0):
    ramp = 0.25
    w = 0.1 * np.pi / 30
    # Integral of a speed command ramped between rest, forward, reverse and rest.
    t0 = start
    t1 = t0 + leg
    t2 = t1 + leg
    pos = base
    speed = 0.0
    for transition, change in [(t0, w), (t1, -2 * w), (t2, w)]:
        d = max(t - transition, 0.0)
        pos += change * (0.5 * min(d, ramp) ** 2 / ramp + max(0.0, d - ramp))
        speed += change * min(d / ramp, 1.0)
    return pos, speed


def voltage_stages(voltage, interval, switched):
    return segments(voltage, interval) if switched else [(voltage, interval)]


def acquire(model, archive, switched=False):
    # Pre-run lookup contains known calibration poses, never a plant trajectory.
    path = (
        archive
        / f"cold_{model.count}_{model.n}_{len(model.slices[0].orders)}{'_ring' if model.ring else ''}{'_bridge' if switched else ''}.npz"
    )
    if path.exists():
        with np.load(path) as d:
            return d["angle"], d["current"]
    angles = np.deg2rad(np.arange(0, 360, 5.0))
    rows = []
    for pose in angles:
        x = model.equilibrium(pose)
        samples = []
        for k in range(12):
            u = cold_voltage(k)
            for v, h in voltage_stages(u, 0.00025, switched):
                x, _, _ = model.step(x, v, h)
                x[-3:] = [pose, 0, 0]
            samples.extend(x[-5:-3])
        rows.append(samples)
    np.savez_compressed(path, angle=angles, current=rows)
    return angles, np.array(rows)


def cold_voltage(k):
    p = k // 2
    a = p % 3 * np.pi / 3
    return 82 * (-1) ** (k % 2 + p // 3) * np.array([np.cos(a), np.sin(a)])


def sample_terminal(
    start,
    end,
    predicted_start,
    predicted_end,
    duration,
    args,
    rng,
    measured_history=None,
    predicted_history=None,
):
    # Keep conversions after a 2 us switch blanking interval.
    n = min(args.adc_samples, 1 + int(max(0.0, duration - 2e-6) / args.adc_spacing))
    left = np.arange(n) * args.adc_spacing / duration
    samples = end + left[:, None] * (start - end)

    def at_conversions(history):
        values = np.asarray(history)
        times = np.linspace(0.0, 1.0, len(values))
        return np.column_stack([np.interp(1.0 - left, times, v) for v in values.T])

    if measured_history is not None:
        samples = at_conversions(measured_history)
    if args.adc_correlation:
        samples += rng.normal(0, args.noise * np.sqrt(args.adc_correlation), 2)
    samples += rng.normal(0, args.noise * np.sqrt(1 - args.adc_correlation), samples.shape)
    sample = samples.mean(axis=0) + np.mean(left) * (predicted_end - predicted_start)
    if predicted_history is not None:
        sample = samples.mean(axis=0) + predicted_end - at_conversions(predicted_history).mean(axis=0)
    sd = args.noise * np.sqrt(args.adc_correlation + (1 - args.adc_correlation) / n)
    return sample, max(0.00005, sd)


def advance_plant(model, state, voltage, duration, load, subdivisions, angles=None, currents=None):
    torque = 0.0
    for _ in range(subdivisions):
        state, mean, _ = model.step(state, voltage, duration / subdivisions, load=load)
        torque += mean
        if angles is not None:
            angles.append(state[-3])
        if currents is not None:
            currents.append(state[-5:-3].copy())
    return state, torque / subdivisions


def advance_observer(model, state, factors, voltage, duration, subdivisions, currents=None):
    step = duration / subdivisions
    average = 0.0
    for _ in range(subdivisions):
        prior = state
        state, torque, jacobian = model.step(prior, voltage, step, tangent=True)
        factors, _ = model.differentiate(
            prior, state, jacobian, factors, np.zeros((2, factors.shape[1])), step
        )
        if currents is not None:
            currents.append(state[-5:-3].copy())
        average += torque
    return state, factors, average / subdivisions


def run(args):
    if args.plant_substeps < 1:
        raise ValueError("plant substeps must be positive")
    if args.observer_substeps < 1 or (
        args.observer_substeps > 1 and (not args.pwm or not args.capture_bank)
    ):
        raise ValueError(
            "refined post-capture observation requires switched control and its native capture bank"
        )
    if args.model_current_rms < 0:
        raise ValueError("model current RMS must be nonnegative")
    if args.fit_start_load and (not args.resolve or args.diagnostic_prior is not None):
        raise ValueError("load fitting requires blind polarity resolution")
    if args.check_capture and not (args.fit_start_load or args.capture_bank):
        raise ValueError("the capture check requires the two-parameter startup fit")
    if args.capture_bank and (
        not args.resolve
        or not args.switched_start
        or args.start_volts != 35
        or args.diagnostic_prior is not None
        or args.ring
        or args.modes
        or args.fit_start_load
    ):
        raise ValueError(
            "the native table requires the two inserts and switched 82/35 V blind acquisition"
        )
    if args.adc_samples < 1 or args.adc_spacing <= 0 or not 0 <= args.adc_correlation <= 1:
        raise ValueError("invalid ADC settings")
    if args.pwm < 0 or ((args.inside_pwm or args.shoot_pwm) and not args.pwm):
        raise ValueError("within-carrier measurements require a switched inverter")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    plant = Machine(args.level, args.modes, args.skew, args.ring)
    observer_level = 3 if args.capture_bank else args.observer_level or args.level
    obs = Machine(
        observer_level,
        args.modes if args.observer_modes is None else args.observer_modes,
        args.skew,
        args.ring,
    )
    obs.trapezoid = args.trapezoid or args.observer_substeps > 1
    plant.trapezoid = args.trapezoid or args.plant_substeps > 1
    if args.calibrate:
        acquire(obs, out, args.switched_start)
        return
    rng = np.random.default_rng(args.seed)
    base = np.deg2rad(args.target)
    x = plant.equilibrium(np.deg2rad(args.initial))
    startup_load = args.load if args.preloaded else 0.0
    startup_trace = [[0.0, x[-3], x[-2], startup_load]]
    pulse_time = 0.0
    resolved = None
    polarity_margin = np.nan
    reduced_cost = np.nan
    accepted = None
    bank_fit = None
    if args.diagnostic_prior is None:
        reads = []
        pulses = []
        stage_reads = []
        startup_peak = 0.0
        for k in range(12):
            for v, h in voltage_stages(cold_voltage(k), 0.00025, args.switched_start):
                x, _ = advance_plant(plant, x, v, h, startup_load, args.plant_substeps)
                pulse_time += h
                startup_trace.append([pulse_time, x[-3], x[-2], startup_load])
                pulses.append((v, h))
                startup_peak = max(startup_peak, np.linalg.norm(x[-5:-3]))
                reading = x[-5:-3] + rng.normal(0, args.noise, 2)
                stage_reads.extend(reading)
            reads.extend(reading)
        if not args.capture_bank:
            angles, lib = acquire(obs, out, args.switched_start)
            grid = np.deg2rad(np.arange(0, 360, 0.1))
            table = np.array([np.interp(grid, angles, v, period=2 * np.pi) for v in lib.T]).T
            cost = np.sum((np.array(reads) - table) ** 2, axis=1)
            initial = grid[np.argmin(cost)]
        if args.resolve:
            for u, nominal_h in resolving_pulses():
                for v, h in voltage_stages(u * (args.start_volts / 115), nominal_h, args.switched_start):
                    x, _ = advance_plant(plant, x, v, h, startup_load, args.plant_substeps)
                    pulse_time += h
                    startup_trace.append([pulse_time, x[-3], x[-2], startup_load])
                    stage_reads.extend(x[-5:-3] + rng.normal(0, args.noise, 2))
                    startup_peak = max(startup_peak, np.linalg.norm(x[-5:-3]))
                    pulses.append((v, h))
            if args.capture_bank:
                from research.air56_rotor_fingerprinting.preloaded_native import read_bank

                bank_fit, accepted = read_bank(args.capture_bank, args.skew, stage_reads, args.noise)
                initial = np.deg2rad(bank_fit.initial_deg)
                reduced_cost, polarity_margin = bank_fit.redcost, bank_fit.margin
                observer_level = 3 + bank_fit.profile
                capture = np.array([np.deg2rad(bank_fit.angle), bank_fit.speed, bank_fit.load])
                if accepted and observer_level != 3:
                    obs = Machine(observer_level, 0, args.skew)
                    obs.trapezoid = args.trapezoid or args.observer_substeps > 1
            else:
                other = abs(wrap(grid - initial)) > np.pi * 2 / 3
                centers = [initial, grid[np.where(other, cost, np.inf).argmin()]]
                (cost, initial, resolved, acquisition_derivative), polarity_margin = resolve(
                    obs,
                    centers,
                    pulses,
                    stage_reads,
                    max(0.0001, args.noise),
                    fit_load=args.fit_start_load,
                )
                capture = resolved[-3:]
                if args.fit_start_load:
                    reduced_cost, accepted = capture_quality(
                        cost, max(0.0001, args.noise), len(stage_reads), polarity_margin
                    )
            if (args.check_capture or args.capture_bank) and not accepted:
                np.savez_compressed(
                    out / (args.name + ".npz"),
                    trace=np.empty((0, 19)),
                    cold_angle=initial,
                    capture_estimate=capture,
                    startup_trace=startup_trace,
                    selected_observer_level=observer_level,
                    startup_peak_a=startup_peak,
                    polarity_margin=polarity_margin,
                    capture_reduced_cost=reduced_cost,
                    capture_status="rejected",
                    settings=json.dumps(vars(args), default=str),
                )
                return

    else:
        initial = np.deg2rad(args.diagnostic_prior)
        startup_peak = 0.0
    est = obs.equilibrium(initial)
    d = 0.00005
    q = (obs.equilibrium(initial + d) - obs.equilibrium(initial - d)) / (2 * d)
    S = np.zeros((len(est), 3))
    S[:, 0] = q * 0.025
    S[-2, 1] = 0.35
    S[-1, 2] = 0.025
    if resolved is not None:
        est = resolved
        if args.fit_start_load:
            S = np.column_stack((acquisition_derivative, S[:, 1]))
        else:
            S[:, 0] = acquisition_derivative * 0.005
    elif args.diagnostic_prior is None:
        if bank_fit is not None:
            est[-1] = bank_fit.load
        for v, h in pulses:
            est, S, _ = advance_observer(obs, est, S, v, h, args.observer_substeps)
    if args.torque_calibration:
        from tools.calibrate_air56_terminal_torque import TorqueCorrection

        obs.torque_calibration = TorqueCorrection(out, args.ring)
    capture_estimate = est[-3:].copy()
    log = []
    speed_filtered = 0.0
    trim = 0.0
    position_i = 0.0
    pwm_bias = 0.0
    bias = None
    rotor_moments = []
    demands = []
    quiet_time = 0.0
    mean_nis = 2.0
    start = time.perf_counter()
    for k in range(int(round(args.seconds / args.h))):
        t = (k + 1) * args.h
        target, wref = profile(t, base, args.leg, args.motion_start)
        load = args.load if args.preloaded else args.load * np.clip((t - 0.03) / 0.15, 0, 1)
        if args.oracle_diagnostic:
            est = x.copy()
            est[-1] = load
        if args.rotating:
            phase = 2 * np.pi * (k + 0.5) / args.carrier_steps
            if k and k % args.carrier_steps == 0:
                trim = np.clip(
                    trim + 0.35 * (np.mean(demands) - np.mean(rotor_moments)) / 0.04, -1.8, 1.8
                )
                rotor_moments = []
                demands = []
            speed_filtered += (1 - np.exp(-args.h / 0.012)) * (est[-2] - speed_filtered)
            err = wrap(target - est[-3])
            position_i = np.clip(position_i + args.h * 0.12 * err, -0.04, 0.04)
            demand = np.clip(
                est[-1] + 0.12 * err + 0.008 * (wref - speed_filtered) + position_i, -0.075, 0.075
            )
            q = np.clip(trim + demand / 0.04, -1.8, 1.8)
            u = args.volts * np.array([np.cos(phase), q * np.sin(phase)])
            pred, tau, fac = obs.step(est, u, args.h, tangent=True)
            rotor_moments.append(tau)
            demands.append(demand)
        else:
            position_i = np.clip(
                position_i + args.h * args.integral_gain * wrap(target - est[-3]), -0.03, 0.03
            )
            if args.shoot_pwm:
                u, pred, tau, fac = predictive_pwm(obs, est, target, wref, k, args, position_i, bias)
            else:
                u, pred, tau, fac = command(
                    obs,
                    est,
                    target,
                    wref,
                    k,
                    args.h,
                    args.stiffness,
                    position_i - pwm_bias,
                    bias,
                    args.probe_v,
                )
            if args.continuous_bias or args.shoot_pwm:
                bias = u - args.probe_v * np.array([np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)])
        angles_in_step = [x[-3]]
        measured_history = predicted_history = None
        if args.pwm:
            average_input_torque = tau
            predicted = est
            actual = tau = 0.0
            stages = segments(u, args.h / args.pwm)
            for carrier in range(args.pwm):
                for j, (v, dt) in enumerate(stages):
                    start_current = x[-5:-3].copy()
                    observer_start_current = predicted[-5:-3].copy()
                    measured_history = [start_current] if args.plant_substeps > 1 else None
                    predicted_history = [observer_start_current] if args.observer_substeps > 1 else None
                    predicted, S, mt = advance_observer(
                        obs, predicted, S, v, dt, args.observer_substeps, predicted_history
                    )
                    x, at = advance_plant(
                        plant, x, v, dt, load, args.plant_substeps, angles_in_step, measured_history
                    )
                    tau += dt * mt / args.h
                    actual += dt * at / args.h
                    last = carrier == args.pwm - 1 and j == len(stages) - 1
                    if args.inside_pwm and not last and dt > 2e-6:
                        mid, mid_sd = sample_terminal(
                            start_current,
                            x[-5:-3],
                            observer_start_current,
                            predicted[-5:-3],
                            dt,
                            args,
                            rng,
                            measured_history,
                            predicted_history,
                        )
                        predicted, S, _ = posterior(
                            predicted, S, mid, np.hypot(mid_sd, args.model_current_rms)
                        )
            pred = predicted
            pwm_bias += (1 - np.exp(-args.h / 0.04)) * (tau - average_input_torque - pwm_bias)
            aperture_interval = stages[-1][1]
        else:
            start_current = x[-5:-3].copy()
            observer_start_current = est[-5:-3]
            measured_history = [start_current] if args.plant_substeps > 1 else None
            x, actual = advance_plant(
                plant, x, u, args.h, load, args.plant_substeps, angles_in_step, measured_history
            )
            S, _ = obs.differentiate(est, pred, fac, S, np.zeros((2, S.shape[1])), args.h)
            aperture_interval = args.h
        adc, sd = sample_terminal(
            start_current,
            x[-5:-3],
            observer_start_current,
            pred[-5:-3],
            aperture_interval,
            args,
            rng,
            measured_history,
            predicted_history,
        )
        new = np.zeros((len(est), 2))
        if args.adaptive:
            quiet_time = 0.0 if mean_nis > 5.0 else quiet_time + args.h
            opening = np.exp(-max(quiet_time - 0.30, 0) / 0.12)
        else:
            opening = 1.0
        new[-1, 0] = (0.000010 + (args.load_process - 0.000010) * opening) * np.sqrt(args.h)
        new[-2, 1] = (0.0001 + 0.0029 * opening) * np.sqrt(args.h)
        S = compress(np.column_stack((S, new)), 7)
        est, S, nis = posterior(pred, S, adc, np.hypot(max(sd, 0.00005), args.model_current_rms))
        mean_nis += (1 - np.exp(-args.h / 0.05)) * (nis - mean_nis)
        log.append(
            [
                t,
                x[-3],
                est[-3],
                target,
                x[-2],
                est[-2],
                wref,
                load,
                est[-1],
                actual,
                tau,
                *u,
                *adc,
                mean_nis,
                opening,
                min(*angles_in_step, x[-3]),
                max(*angles_in_step, x[-3]),
            ]
        )
        if k % 40 == 0 or k + 1 == int(round(args.seconds / args.h)):
            record = out / (args.name + ".npz")
            temp = record.with_suffix(".partial.npz")
            np.savez_compressed(
                temp,
                trace=log,
                cold_angle=initial,
                startup_peak_a=startup_peak,
                startup_trace=startup_trace,
                capture_estimate=capture_estimate,
                native_capture_deg=bank_fit.angle if bank_fit is not None else np.nan,
                capture_reduced_cost=reduced_cost,
                capture_status="accepted" if args.check_capture or args.capture_bank else "unchecked",
                selected_observer_level=observer_level,
                polarity_margin=polarity_margin,
                settings=json.dumps(vars(args), default=str),
                elapsed_s=time.perf_counter() - start,
            )
            temp.replace(record)
            print(k, np.round(np.rad2deg([x[-3], est[-3], target]), 3), round(est[-1], 4), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", default="trial")
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--modes", type=int, default=30)
    p.add_argument("--observer-level", type=int)
    p.add_argument("--observer-modes", type=int)
    p.add_argument("--initial", type=float, default=27)
    p.add_argument("--target", type=float, default=27)
    p.add_argument("--diagnostic-prior", type=float)
    p.add_argument("--load", type=float, default=0.02)
    p.add_argument(
        "--preloaded", action="store_true", help="Apply the plant load from the first probing pulse"
    )
    p.add_argument(
        "--capture-bank", type=Path, help="Use read-only native startup tables for the two-insert rotor"
    )
    p.add_argument(
        "--fit-start-load",
        action="store_true",
        help="Fit unknown angle and signed disturbance from startup currents",
    )
    p.add_argument(
        "--check-capture",
        action="store_true",
        help="End this trial before servo enable if the startup fit is incompatible",
    )
    p.add_argument("--load-process", type=float, default=0.012)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--leg", type=float, default=2.7)
    p.add_argument("--noise", type=float, default=0.003)
    p.add_argument("--seed", type=int, default=2119)
    p.add_argument("--h", type=float, default=0.004)
    p.add_argument("--skew", action="store_true")
    p.add_argument("--oracle-diagnostic", action="store_true")
    p.add_argument("--ring", action="store_true")
    p.add_argument("--resolve", action="store_true")
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--continuous-bias", action="store_true")
    p.add_argument("--probe-v", type=float, default=5.0)
    p.add_argument("--stiffness", type=float, default=0.32)
    p.add_argument("--integral-gain", type=float, default=0.0)
    p.add_argument("--motion-start", type=float, default=1.0)
    p.add_argument("--pwm", type=int, default=0, help="Switched SVM periods per control update")
    p.add_argument("--torque-calibration", action="store_true")
    p.add_argument("--shoot-pwm", action="store_true")
    p.add_argument("--switched-start", action="store_true")
    p.add_argument("--start-volts", type=float, default=115.0)
    p.add_argument("--trapezoid", action="store_true")
    p.add_argument("--observer-substeps", type=int, default=1)
    p.add_argument(
        "--model-current-rms",
        type=float,
        default=0.0,
        help="Observer-only current model discrepancy in ampere RMS",
    )
    p.add_argument(
        "--plant-substeps",
        type=int,
        default=1,
        help="Refine each plant dwell; >1 enables trapezoidal shaft integration only in the plant",
    )
    p.add_argument("--inside-pwm", action="store_true")
    p.add_argument("--adc-samples", type=int, default=1)
    p.add_argument("--adc-spacing", type=float, default=2.0e-6)
    p.add_argument("--adc-correlation", type=float, default=0.0)
    p.add_argument("--rotating", action="store_true")
    p.add_argument("--carrier-steps", type=int, default=8)
    p.add_argument("--volts", type=float, default=25.0)
    p.add_argument("--calibrate", action="store_true")
    run(p.parse_args())
