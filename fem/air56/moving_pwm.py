"""Switched transients with one FEM state basis in rotor/stator material coordinates."""

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from fem.air56.pwm_section import (
    AB_TO_PHASE, BUS_V, DEAD_TICKS, LOAD_HZ, PHASE_TO_AB, PROBE_HZ, PWM_PERIOD_S,
    STATOR_OHM, TIMER_HZ, TIMER_TICKS, edges_for_period,
)
from fem.air56.sliding_section import TAU


@dataclass(frozen=True)
class MaterialModel:
    stiffness: np.ndarray
    cage: np.ndarray
    winding: np.ndarray
    linkage: np.ndarray
    offset: float
    angular_step: float


def stator_permutations(full, rotor_radius):
    points = full.basis.mesh.p[:, full.free].T
    stator = np.flatnonzero(np.linalg.norm(points, axis=1) > rotor_radius+1e-10)
    locator = cKDTree(points[stator])
    result = []
    for sector in range(6):
        delta = sector*np.pi/3
        c, s = np.cos(delta), np.sin(delta)
        rotation = np.array([[c, -s], [s, c]])
        distance, rotated = locator.query(points[stator] @ rotation.T)
        if distance.max() > 1e-9 or np.unique(rotated).size != len(stator):
            raise RuntimeError("stator node rotation did not preserve the finite element grid")
        perm = np.arange(len(points))
        perm[stator] = stator[rotated]
        np.testing.assert_allclose(full.winding[perm], full.winding @ rotation.T, rtol=1e-10, atol=1e-11)
        result.append((perm, rotation, delta))
    return result


def common_model(sections, rotor_radius, rank, offset, angular_indices):
    """Project remeshed operators on a fixed space of material node values."""
    full = sections[0][0]
    rotations = stator_permutations(full, rotor_radius)
    for point, _, _, _ in sections:
        if not np.array_equal(full.free, point.free):
            raise RuntimeError("moving sectors must use the same material node indexing")
        np.testing.assert_allclose(point.winding, full.winding, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(point.integral, full.integral, rtol=1e-12, atol=1e-12)

    angular_step = np.pi/(3*len(sections))
    registered = []
    for angle in angular_indices:
        sector, index = divmod(angle, len(sections))
        perm, rotation, delta = rotations[sector % 6]
        _, matrices, linkage, snapshots = sections[index]
        registered.append((sector, perm, rotation, delta, matrices, linkage, snapshots[np.argsort(perm)]))
    snapshots = [row[-1] for row in registered]
    rng = np.random.default_rng(44)
    sketch = sum(x @ rng.standard_normal((x.shape[1], rank+16)) for x in snapshots)
    span = np.linalg.qr(sketch)[0]
    # One power iteration resolves closely spaced energetic directions.
    span = np.linalg.qr(sum(x @ (x.T @ span) for x in snapshots))[0]
    left, _, _ = np.linalg.svd(np.concatenate([span.T @ x for x in snapshots], axis=1), full_matrices=False)
    vectors = span @ left[:, :rank]
    projected_integral = vectors.T @ full.integral
    inertia = vectors.T @ (full.cage @ vectors)-np.outer(projected_integral, projected_integral)/full.conductance
    winding = vectors.T @ full.winding
    operators, load_linkage = [], []
    for sector, perm, rotation, delta, matrices, linkage, _ in registered:
        transformed = vectors[perm]
        stiffness = [transformed.T @ (matrix @ transformed) for matrix in matrices]
        operators.append(np.roll(stiffness, 4*sector, axis=0))
        load_linkage.append((rotation @ linkage)*np.exp(-1j*delta))
    return MaterialModel(np.array(operators), inertia, winding, np.array(load_linkage),
                         offset+angular_indices[0]*angular_step, angular_step)


def moving_circuit(model, stack_length, skew, phase_nodes, axial_modes=3):
    n, phase_count, rank, _ = model.stiffness.shape
    length = (2/3)*stack_length
    offset = model.offset
    if skew:
        weights = np.array([1, 4, 2, 4, 1])/12
        x = np.linspace(-1., 1., 5)
        axial = np.linalg.qr(np.sqrt(weights[:, None])*np.polynomial.legendre.legvander(x, axial_modes-1))[0]
        n -= 4
        offset += 2*model.angular_step
        slices = [model.stiffness[i:i+n] for i in range(5)]
        load_linkage = sum(w*model.linkage[i:i+n] for i, w in enumerate(weights))
        phase_k = sum(np.einsum("a,b,nfij->nfaibj", a, a, k) for a, k in zip(axial, slices))
        phase_k = phase_k.reshape(n, phase_count, axial_modes*rank, axial_modes*rank)
        winding = np.kron((np.sqrt(weights) @ axial)[:, None], model.winding)
        inertia = np.kron(np.eye(axial_modes), model.cage)
    else:
        phase_k, load_linkage, winding, inertia = model.stiffness, model.linkage, model.winding, model.cage
    inertia = inertia+length/STATOR_OHM*(winding @ winding.T)
    value, modes = np.linalg.eigh(inertia)
    scale = modes/np.sqrt(np.maximum(value, value.max()*1e-11))
    dimension = len(value)
    poles = np.empty((n, phase_nodes, dimension))
    forward = np.empty((n, phase_nodes, dimension, dimension))
    source = scale.T @ winding/STATOR_OHM
    drive = np.empty((n, phase_nodes, dimension, 2))
    flux = np.empty((n, phase_nodes, 2, dimension))
    for index, phase in enumerate(np.arange(phase_nodes)*phase_count/phase_nodes):
        lo = int(phase)
        part = phase-lo
        stiffness = (1-part)*phase_k[:, lo]+part*phase_k[:, (lo+1) % phase_count]
        eig, basis = np.linalg.eigh(scale.T @ stiffness @ scale)
        poles[:, index] = eig
        forward[:, index] = basis
        drive[:, index] = basis.swapaxes(1, 2) @ source
        flux[:, index] = (length*winding.T @ scale) @ basis
    if poles.min() <= 0:
        raise RuntimeError("non-positive moving-mesh transient pole")
    inverse = np.ascontiguousarray(forward.swapaxes(-1, -2))
    decay = np.exp(-poles*PWM_PERIOD_S)
    hold = -np.expm1(-poles*PWM_PERIOD_S)/poles
    legs = drive @ (BUS_V*PHASE_TO_AB)
    return poles, decay, hold, forward, inverse, drive, legs, flux, load_linkage, offset, model.angular_step


@njit(cache=True)
def advance_modal(state, poles, decay, hold, drive, legs, rise, fall, base_voltage):
    for i in range(len(state)):
        rate = poles[i]
        delta = -hold[i]*(drive[i, 0]*base_voltage[0]+drive[i, 1]*base_voltage[1])
        for leg in range(3):
            # Closed-form integration of the exact high-side pulse on this mode.
            end_decay = np.exp(-rate*(TIMER_TICKS-fall[leg])/TIMER_HZ)
            width = (fall[leg]-rise[leg])/TIMER_HZ
            delta += legs[i, leg]*end_decay*(-np.expm1(-rate*width))/rate
        state[i] = decay[i]*state[i]+delta


@njit(cache=True)
def segment_change(state, poles, drive, flux, voltage, duration, guard):
    """Advance one switching state and return the guarded current increment."""
    change = np.zeros(2)
    aperture = duration-2*guard
    for i in range(len(state)):
        rate = poles[i]
        forcing = drive[i, 0]*voltage[0]+drive[i, 1]*voltage[1]
        derivative = forcing-rate*state[i]
        if aperture > 0.:
            increment = -np.exp(-rate*guard)*np.expm1(-rate*aperture)*derivative/STATOR_OHM
            change[0] += flux[0, i]*increment
            change[1] += flux[1, i]*increment
        state[i] -= np.expm1(-rate*duration)*derivative/rate
    return change


@njit(cache=True)
def slope_period(state_a, state_b, poles, drive, flux, rise, fall, voltage, weight, row):
    """Record the active-minus-passive slope and its voltage-time increment."""
    guard = .35e-6
    min_aperture = .50e-6
    cuts = np.sort(np.concatenate((np.array([0, TIMER_TICKS]), rise, fall)))
    active_time, passive_time = 0., 0.
    n_active, n_passive = 0, 0
    active_i, passive_i, volt_seconds = np.zeros(2), np.zeros(2), np.zeros(2)
    for j in range(7):
        left, right = cuts[j], cuts[j+1]
        if left == right:
            continue
        duration = (right-left)/TIMER_HZ
        aperture = duration-2*guard
        middle = .5*(left+right)
        on = (rise <= middle) & (middle < fall)
        applied = PHASE_TO_AB @ (BUS_V*on.astype(np.float64))
        sampling_guard = guard if aperture >= min_aperture else duration/2
        a = segment_change(state_a, poles[0], drive[0], flux[0], applied-voltage, duration, sampling_guard)
        b = segment_change(state_b, poles[1], drive[1], flux[1], applied-voltage, duration, sampling_guard)
        if aperture < min_aperture:
            continue
        change = (1-weight)*a+weight*b
        if 0 < np.sum(on) < 3:
            active_time += aperture
            active_i += change
            volt_seconds += aperture*applied
            n_active += 1
        else:
            passive_time += aperture
            passive_i += change
            n_passive += 1
    if n_active and n_passive:
        ratio = active_time/passive_time
        row[:2] = active_i-ratio*passive_i
        row[2:4] = volt_seconds/PWM_PERIOD_S
        row[4] = 2*(n_active+ratio**2*n_passive)


@njit(cache=True)
def moving_run(poles, decay, hold, forward, backward, drive, legs, flux, linkage,
               offset, angular_step, theta_initial, rpm, current_peak, probe_axis, periods, motion_block, capture_periods,
               ramp_start_s=0., ramp_duration_s=0., terminal_rpm=0., square_probe=False,
               slope_records=None, alternating_probe_v=0.):
    """Propagate between remeshed FE operators without restarting cage history."""
    n, phase_nodes, dimension = poles.shape
    state_a, state_b = np.zeros(dimension), np.zeros(dimension)
    flux_before = np.zeros(2)
    current_increment = np.zeros(2)
    result = np.empty((periods, 2))
    events = np.zeros((capture_periods*7, 6))
    previous_angle, previous_phase, previous_weight = -1, -1, 0.
    recorded, changes = 0, 0
    coefficient_derivative_norm = 0.
    for sample in range(periods):
        t = (sample+.5)*PWM_PERIOD_S
        active_rpm, integral_rpm = rpm, rpm*t
        if ramp_duration_s > 0.:
            fraction = min(max((t-ramp_start_s)/ramp_duration_s, 0.), 1.)
            excursion = .5*ramp_duration_s*fraction*fraction+max(t-ramp_start_s-ramp_duration_s,0.)
            active_rpm += (terminal_rpm-rpm)*fraction
            integral_rpm += (terminal_rpm-rpm)*excursion
        mechanical = TAU*active_rpm/60
        theta = theta_initial+TAU*integral_rpm/60
        index = (theta-offset)/angular_step
        if index < -1e-8 or index > n-1+1e-8:
            raise ValueError("rotor trajectory left the prepared moving-band interval")
        index = min(max(index, 0.), n-1.)
        lower = min(int(index), n-2)
        upper = lower+1
        weight = index-lower
        field_phase = int(np.rint(2*LOAD_HZ*t*phase_nodes)) % phase_nodes
        if previous_angle >= 0 and (field_phase != previous_phase or lower != previous_angle or sample % motion_block == 0):
            old_upper = previous_angle+1
            combined = (1-previous_weight)*(forward[previous_angle, previous_phase] @ state_a) \
                       +previous_weight*(forward[old_upper, previous_phase] @ state_b)
            state_a = backward[lower, field_phase] @ combined
            state_b = backward[upper, field_phase] @ combined
            changes += 1
        previous_angle, previous_phase, previous_weight = lower, field_phase, weight
        e = complex(np.cos(TAU*LOAD_HZ*t), np.sin(TAU*LOAD_HZ*t))
        load = current_peak*np.array([e.real, e.imag])
        link = (1-weight)*linkage[lower]+weight*linkage[upper]
        motional = mechanical*(linkage[upper]-linkage[lower])/angular_step
        coefficient_derivative_norm = max(coefficient_derivative_norm, np.linalg.norm(motional))
        voltage = STATOR_OHM*load+((1j*TAU*LOAD_HZ*link+motional)*e).real
        command = voltage.copy()
        if alternating_probe_v:
            command[(sample//2) % 2] += alternating_probe_v if sample % 2 == 0 else -alternating_probe_v
        elif probe_axis >= 0:
            probe = np.cos(TAU*PROBE_HZ*t)
            command[probe_axis] += (np.pi/4)*np.sign(probe) if square_probe else probe
        reference = AB_TO_PHASE @ load
        estimate = reference+AB_TO_PHASE @ current_increment
        compensated = AB_TO_PHASE @ command+BUS_V*DEAD_TICKS/TIMER_TICKS*np.sign(reference)
        rise, fall = edges_for_period(compensated, estimate, DEAD_TICKS)
        if slope_records is None:
            for node in (lower, upper):
                state = state_a if node == lower else state_b
                advance_modal(state, poles[node, field_phase], decay[node, field_phase], hold[node, field_phase],
                              drive[node, field_phase], legs[node, field_phase], rise, fall, voltage)
        else:
            slope_period(state_a, state_b, poles[lower:upper+1, field_phase], drive[lower:upper+1, field_phase],
                         flux[lower:upper+1, field_phase], rise, fall, voltage, weight, slope_records[sample])
        after = (1-weight)*(flux[lower, field_phase] @ state_a)+weight*(flux[upper, field_phase] @ state_b)
        applied = PHASE_TO_AB @ (BUS_V*(fall-rise)/TIMER_TICKS)
        current_increment = ((applied-voltage)-(after-flux_before)/PWM_PERIOD_S)/STATOR_OHM
        flux_before = after
        result[sample] = load+current_increment
        if sample < capture_periods:
            cuts = np.sort(np.concatenate((np.array([0, TIMER_TICKS]), rise, fall)))
            for j in range(7):
                left, right = cuts[j], cuts[j+1]
                if left == right:
                    continue
                middle = .5*(left+right)
                on = (rise <= middle) & (middle < fall)
                step_v = PHASE_TO_AB @ (BUS_V*on.astype(np.float64))-voltage
                events[recorded] = [float(sample), (sample*TIMER_TICKS+right)/TIMER_HZ,
                                    (right-left)/TIMER_HZ, step_v[0], step_v[1], theta]
                recorded += 1
    return result, events[:recorded], changes, coefficient_derivative_norm
