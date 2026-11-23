"""Parallel, independent FEM evolutions between successive ADC observations."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack

import numpy as np


def _initialize(level, modes, skew, ring):
    from tools.run_air56_terminal_servo import Machine

    global _machine
    _machine = Machine(level, modes, skew, ring)
    _machine.trapezoid = True


def _step(state, voltage, dt, load, subdivisions, log_angle, log_current):
    from tools.run_air56_terminal_servo import advance_plant

    angles = [] if log_angle else None
    currents = [state[-5:-3].copy()] if log_current else None
    state, torque = advance_plant(_machine, state, voltage, dt, load, subdivisions, angles, currents)
    return state, torque, angles, currents


def _impulse(state, stages, interval, subdivisions):
    from tools.run_air56_terminal_servo import advance_plant

    total = 0.0
    for v, h in stages:
        state, tau = advance_plant(_machine, state, v, h, state[-1], subdivisions)
        total += h * tau / interval
    return total, state[-6]


def _signature(angle, load, pulses, subdivisions):
    from tools.run_air56_terminal_servo import advance_plant

    state = _machine.equilibrium(angle)
    state[-1] = load
    trace = []
    for v, h in pulses:
        state, _ = advance_plant(_machine, state, v, h, load, subdivisions)
        trace.append(state[-5:-3].copy())
    return np.asarray(trace)


class DwellWorkers(ExitStack):
    """One isolated field/cache per branch. Only the plant worker receives a plant state."""

    def __init__(self, observer_level, plant_level, plant_modes, skew):
        super().__init__()

        def worker(level, modes):
            return self.enter_context(
                ProcessPoolExecutor(
                    max_workers=1, initializer=_initialize, initargs=(level, modes, skew, False)
                )
            )

        self.observer = {n: worker(observer_level, 0) for n in (1, 7)}
        self.plant = worker(plant_level, plant_modes)

    def reference(self, state, voltage, dt, load, steps):
        return self.plant.submit(_step, state, voltage, dt, load, steps, True, True)

    def predictions(self, state, voltage, dt, history):
        return {
            n: pool.submit(_step, state, voltage, dt, state[-1], n, False, history)
            for n, pool in self.observer.items()
        }

    def impulses(self, state, stages, interval):
        return {n: pool.submit(_impulse, state, stages, interval, n) for n, pool in self.observer.items()}

    def signatures(self, angle, load, pulses):
        return {n: pool.submit(_signature, angle, load, pulses, n) for n, pool in self.observer.items()}
