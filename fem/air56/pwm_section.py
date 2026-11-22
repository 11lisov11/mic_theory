"""Circuit-coupled transient projection of saturated AIR56 sections."""

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.linalg import block_diag
from scipy.sparse.linalg import splu
from skfem import Basis, ElementTriP1

from fem.air56.loaded_section import cage_matrices, magnetic_force, material, tensor_matrix
from fem.air56.sliding_section import TAU, winding_rhs


TIMER_TICKS = 10624
TIMER_HZ = 170e6
PWM_PERIOD_S = TIMER_TICKS/TIMER_HZ
DEAD_TICKS = round(1.35e-6*TIMER_HZ)
BUS_V = 322.
LOAD_HZ = 5.
PROBE_HZ = 100.
STATOR_OHM = 27.5/3
PHASE_NODES = 60
PHASE_TO_AB = (2/3)*np.array([[1., -.5, -.5], [0., np.sqrt(3)/2, -np.sqrt(3)/2]])
AB_TO_PHASE = 1.5*PHASE_TO_AB.T


@dataclass(frozen=True)
class Projection:
    stiffness: np.ndarray
    cage: np.ndarray
    winding: np.ndarray
    linkage: np.ndarray


@dataclass(frozen=True)
class FullSection:
    cage: object
    integral: np.ndarray
    conductance: float
    winding: np.ndarray
    basis: object
    free: np.ndarray
    steel: np.ndarray
    loaded_field: np.ndarray


def section_matrices(geometry, section, fundamental):
    basis = Basis(section.mesh, ElementTriP1(), intorder=2)
    free = basis.complement_dofs(basis.get_dofs())
    winding = winding_rhs(geometry, section, basis)[free]
    sigma = np.where(section.cage, geometry.aluminium_sigma, 0.)
    cage, integral, total = cage_matrices(basis, sigma, free)
    tangent = fundamental["tangent"]
    matrices = tuple(tensor_matrix(basis, (v[:, k] for v in tangent))[free][:, free].tocsc()
                     for k in range(tangent[0].shape[1]))
    state = fundamental["state"]
    count = len(free)
    potential = np.zeros(basis.N, dtype=complex)
    potential[free] = state[:count]+1j*state[count:2*count]
    linkage = (2/3)*geometry.stack_length*(winding.T @ potential[free])
    full = FullSection(cage, integral, total, winding, basis, free, section.steel,
                       basis.interpolate(potential).grad[..., 0])
    return full, matrices, linkage


def response_snapshots(full, matrices, phases=(0, 3, 6, 9)):
    columns = []
    for phase in phases:
        stiffness = matrices[phase]
        for freq in (0., 100., 600., 4000., 16000., 64000.):
            iw = 1j*TAU*freq
            raw = splu((stiffness+iw*full.cage).tocsc()).solve(np.column_stack((full.winding, iw*full.integral)))
            potential, neutral = raw[:, :2], raw[:, 2]
            if freq:
                potential += neutral[:, None]*((full.integral @ potential)/(full.conductance-full.integral @ neutral))
            columns.extend((potential.real, potential.imag))
    snapshots = np.column_stack(columns)
    norms = np.linalg.norm(snapshots, axis=0)
    active = norms > 1e-12*norms.max()
    return snapshots[:, active]/norms[active]


def projected_section(geometry, section, fundamental, rank):
    full, matrices, linkage = section_matrices(geometry, section, fundamental)
    modes, singular, _ = np.linalg.svd(response_snapshots(full, matrices), full_matrices=False)
    vectors = modes[:, :min(rank, np.count_nonzero(singular > 1e-9*singular[0]))]
    projected_integral = vectors.T @ full.integral
    mass = vectors.T @ (full.cage @ vectors)-np.outer(projected_integral, projected_integral)/full.conductance
    reduced = Projection(np.array([vectors.T @ (k @ vectors) for k in matrices]),
                         mass, vectors.T @ full.winding, linkage)
    return reduced, full


def rotate_projection(section, sector):
    angle = sector*np.pi/3
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return Projection(np.roll(section.stiffness, 4*sector, axis=0), section.cage,
                      section.winding @ rotation.T, (rotation @ section.linkage)*np.exp(-1j*angle))


def circuit(sections, weights, stack_length):
    length = (2/3)*stack_length
    winding = np.vstack([np.sqrt(w)*s.winding for w, s in zip(weights, sections)])
    inertia = block_diag(*(s.cage for s in sections))+length/STATOR_OHM*(winding @ winding.T)
    inertias, directions = np.linalg.eigh(inertia)
    scale = directions/np.sqrt(np.maximum(inertias, inertias.max()*1e-11))
    phase_count = len(sections[0].stiffness)
    systems = []
    for phase in np.arange(PHASE_NODES)*phase_count/PHASE_NODES:
        lower = int(phase)
        fraction = phase-lower
        blocks = [(1-fraction)*s.stiffness[lower]+fraction*s.stiffness[(lower+1) % phase_count]
                  for s in sections]
        systems.append(scale.T @ block_diag(*blocks) @ scale)
    poles, modes = np.linalg.eigh(systems)
    if poles.min() <= 0:
        raise RuntimeError("non-positive transient pole")
    source = scale.T @ winding/STATOR_OHM
    forward = np.ascontiguousarray(modes)
    backward = np.ascontiguousarray(modes.swapaxes(1, 2))
    drive = backward @ source
    voltage_flux = length*(winding.T @ scale)
    load_linkage = sum(w*s.linkage for w, s in zip(weights, sections))
    return poles, forward, backward, drive, voltage_flux, load_linkage


@njit(cache=True)
def edges_for_period(command, current_sign, dead_ticks):
    common = -.5*(np.max(command)+np.min(command))
    duty = np.minimum(.97, np.maximum(.03, .5+(command+common)/BUS_V))
    half_width = np.rint(.5*TIMER_TICKS*duty).astype(np.int64)
    rise = TIMER_TICKS//2-half_width
    fall = TIMER_TICKS//2+half_width
    for leg in range(3):
        if current_sign[leg] >= 0:
            rise[leg] += dead_ticks
        else:
            fall[leg] += dead_ticks
    return rise, fall


@njit(cache=True)
def switched_run(poles, forward, backward, drive, voltage_flux, load_linkage, current_peak,
                 probe_axis, periods, capture_periods, dead_ticks):
    """Advance on the transistor edges. Output exact period-average winding current."""
    modal = np.zeros(poles.shape[1])
    previous_cell = -1
    old_delta_i = np.zeros(2)
    trace = np.empty((periods, 2))
    events = np.zeros((capture_periods*7, 5))
    recorded = 0
    absolute_delta = 0.
    sign_disagreement = 0
    sample_square = 0.
    for period in range(periods):
        t = (period+.5)*PWM_PERIOD_S
        e = complex(np.cos(TAU*LOAD_HZ*t), np.sin(TAU*LOAD_HZ*t))
        load_current = np.array([e.real, e.imag])*current_peak
        working_voltage = STATOR_OHM*load_current+(1j*TAU*LOAD_HZ*load_linkage*e).real
        commanded = working_voltage.copy()
        if probe_axis >= 0:
            commanded[probe_axis] += np.cos(TAU*PROBE_HZ*t)
        base_phase = AB_TO_PHASE @ load_current
        estimate = base_phase+AB_TO_PHASE @ old_delta_i
        for j in range(3):
            sign_disagreement += (base_phase[j] >= 0) != (estimate[j] >= 0)
        # Feedforward uses the working current reference. Clamping during
        # dead time uses the previous period's total FEM current prediction.
        phase_command = AB_TO_PHASE @ commanded+BUS_V*dead_ticks/TIMER_TICKS*np.sign(base_phase)
        rise, fall = edges_for_period(phase_command, estimate, dead_ticks)
        cuts = np.sort(np.concatenate((np.array([0, TIMER_TICKS]), rise, fall)))
        cell = int(np.rint(2*LOAD_HZ*t*PHASE_NODES)) % PHASE_NODES
        if previous_cell >= 0 and cell != previous_cell:
            modal = backward[cell] @ (forward[previous_cell] @ modal)
        previous_cell = cell
        before = modal.copy()
        integrated_voltage = np.zeros(2)
        for n in range(7):
            left, right = cuts[n], cuts[n+1]
            if left == right:
                continue
            middle = .5*(left+right)
            pole = (rise <= middle) & (middle < fall)
            voltage = PHASE_TO_AB @ (BUS_V*pole.astype(np.float64))-working_voltage
            dt = (right-left)/TIMER_HZ
            decay = np.exp(-poles[cell]*dt)
            response = -np.expm1(-poles[cell]*dt)/poles[cell]
            modal = decay*modal+response*(drive[cell] @ voltage)
            integrated_voltage += voltage*dt
            if period < capture_periods:
                events[recorded] = np.array([float(period), (period*TIMER_TICKS+right)/TIMER_HZ,
                                             dt, voltage[0], voltage[1]])
                recorded += 1
        old_delta_i = (integrated_voltage-voltage_flux @ (forward[cell] @ (modal-before)))/(STATOR_OHM*PWM_PERIOD_S)
        trace[period] = load_current+old_delta_i
        absolute_delta = max(absolute_delta, np.sqrt(old_delta_i @ old_delta_i))
        sample_square += old_delta_i @ old_delta_i
    return trace, events[:recorded], absolute_delta, np.sqrt(sample_square/periods), sign_disagreement


def demodulate(traces, periods, *, center_s=None):
    """Fit the paired 100 Hz response and the working-field sidebands."""
    times = (np.arange(periods)+.5)*PWM_PERIOD_S
    use = times >= .2 if center_s is None else abs(times-center_s)<.1
    harmonics = np.arange(1, 41)*LOAD_HZ
    angle = TAU*times[use, None]*harmonics
    design = np.column_stack((np.ones(use.sum()), np.cos(angle), np.sin(angle)))
    samples = (traces[1:, use]-traces[0, use]).transpose(1, 2, 0).reshape(use.sum(), 4)
    mixing = np.linalg.lstsq(design, samples, rcond=None)[0]
    phasor = mixing[1:41]-1j*mixing[41:]
    sideband = np.arange(-5, 6)
    carrier_index = (PROBE_HZ/LOAD_HZ+2*sideband).astype(int)-1
    working_phases = np.arange(6)*np.pi/6
    envelope = np.exp(2j*working_phases[:, None]*sideband) @ phasor[carrier_index]
    return envelope.reshape(6, 2, 2)


def direct_nonlinear_trace(section, events, stack_length, current_peak, subdivisions, *, moving_sections=None):
    """Full-space, nonlinear B-H integration on a captured PWM edge schedule."""
    rows = int(events[-1, 0])+1
    result = np.zeros((rows, 2))
    stack = (2/3)*stack_length
    winding, cage = section.winding, section.cage
    basis, free = section.basis, section.free
    low_rank = np.column_stack((np.sqrt(stack/STATOR_OHM)*winding,
                               section.integral/np.sqrt(section.conductance)))
    signs = np.array([1., 1., -1.])
    state = np.zeros(len(free))
    full = np.zeros(basis.N)
    max_residual, max_increment = 0., 0.
    for period, end, duration, va, vb in events:
        row = int(period)
        if moving_sections is not None:
            section = moving_sections[row]
            basis = section.basis
        dt = duration/subdivisions
        forcing = winding @ np.array([va, vb])/STATOR_OHM
        for n in range(subdivisions):
            t = end-duration+(n+1)*dt
            old = state.copy()
            base_x, base_y = (section.loaded_field*np.exp(1j*TAU*LOAD_HZ*t)).real
            base_mu, _ = material(base_x, base_y, section.steel)
            target = max(1., np.linalg.norm(forcing))
            for iteration in range(12):
                full[free] = state
                dx, dy = basis.interpolate(full).grad[..., 0]
                gx, gy = base_x+dx, base_y+dy
                nu, tangent = material(gx, gy, section.steel)
                h = magnetic_force.assemble(basis, hx=(nu*gx-base_mu*base_x)[:, None],
                                            hy=(nu*gy-base_mu*base_y)[:, None])[free]
                rate = (state-old)/dt
                residual = h+cage @ rate+low_rank @ (signs*(low_rank.T @ rate))-forcing
                relative = np.linalg.norm(residual)/target
                if relative <= 2e-7:
                    max_residual = max(max_residual, relative)
                    max_increment = max(max_increment, float(np.hypot(dx, dy)[section.steel].max()))
                    break
                nonlinear = tensor_matrix(basis, tangent)[free][:, free]+cage/dt
                z = splu(nonlinear.tocsc()).solve(np.column_stack((-residual, low_rank)))
                right, response = z[:, 0], z[:, 1:]
                small = dt*np.diag(signs)+low_rank.T @ response
                state += right-response @ np.linalg.solve(small, low_rank.T @ right)
            else:
                raise RuntimeError(f"direct PWM Newton failed at PWM period {row}, residual {relative:g}")
            result[row] += (np.array([va, vb])*dt-stack*winding.T @ (state-old))/(STATOR_OHM*PWM_PERIOD_S)
    time = (np.arange(rows)+.5)*PWM_PERIOD_S
    result += current_peak*np.column_stack((np.cos(TAU*LOAD_HZ*time), np.sin(TAU*LOAD_HZ*time)))
    return result, max_residual, max_increment
