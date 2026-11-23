"""Rotor-field energy harvesting and synchronous backscatter through the power windings."""

from pathlib import Path
import sys
import argparse
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fem.air56.sliding_section import make_section, solve
from tools.run_air56_skew_audit import article_profile, write_csv


def harmonic_ports(root):
    g, design = article_profile()
    angles = np.arange(24) * 2 * np.pi / 24
    z = np.empty((3, 24, 2, 2), complex)
    transfer = np.empty_like(z)
    for i, angle in enumerate(angles):
        section = make_section(g, angle, 7, design, 1)
        for j, hz in enumerate((200, 1000, 5000)):
            s = solve(g, hz, section, 27.5 / 3, transfer=True)
            if max(s[k] for k in ("residual", "energy", "reciprocity")) > 1e-7:
                raise RuntimeError("harmonic port FEM checks")
            z[j, i] = s["z"]
            transfer[j, i] = s["transfer"]
    np.savez_compressed(root / "harmonic_ports.npz", z=z, transfer=transfer)


def circuit(root, stock):
    primary = np.load(stock)
    doubled = np.load(root / "harmonic_ports.npz")
    time = np.arange(256) * 2 * np.pi / 256
    harvest, signals = [], []
    for turns, length in ((1, 0.040), (20, 0.004), (100, 0.040)):
        area = 0.004 * length
        r = 1.72e-8 * 2 * (length + 0.004) * turns / (0.0003 * 35e-6)
        # 50 uH reference for a compact 20-turn tag; extended shape is a scenario.
        inductance = 0.125e-6 * turns**2
        for hz in (0, 5, 50):
            for field_t in (0.25, 0.8):
                peak = 2 * np.pi * hz * turns * area * field_t
                harvest.append(
                    dict(
                        turns=turns,
                        length_mm=length * 1000,
                        relative_hz=hz,
                        b_peak_t=field_t,
                        open_circuit_peak_v=peak,
                        max_ac_power_uw=1e6 * peak**2 / (8 * r),
                        exceeds_150mv=peak > 0.15,
                        copper_volume_mm3=2 * (length + 0.004) * turns * 0.0003 * 35e-6 * 1e9,
                    )
                )
        for j, hz in enumerate((100, 500, 2500)):
            w = 2 * np.pi * hz
            m1 = primary["field_t_per_a"][1, j, :, 0, :] * turns * area
            z1 = primary["z"][1, j]
            z2 = doubled["z"][j]
            m2 = doubled["transfer"][j, :, 0, :] * turns * area
            for voltage in (1.0, 5.0):
                primary_i = np.linalg.inv(z1)[:, :, 0] * voltage
                voltage_oc = -1j * w * np.sum(m1 * primary_i, axis=1)
                # Return between a matched resistive load and open state.
                current_tag = voltage_oc / (2 * r + 1j * w * inductance)
                emf = (2 / 3) * 1j * w * m1 * current_tag[:, None]
                return_i = np.linalg.solve(z1, emf[..., None])[..., 0]
                for vf in (0.0, 0.15):
                    # Periodic series RL + one-way element. Include the complex
                    # transfer phases on both carrier and second-harmonic paths.
                    emf_t = (voltage_oc[:, None] * np.exp(1j * time)).real
                    decay = np.exp(-r / (256 * hz * inductance))
                    amps = np.zeros(len(voltage_oc))
                    for _ in range(8):
                        history = []
                        for sample in emf_t.T:
                            amps = np.maximum(0, decay * amps + (1 - decay) * (sample - vf) / r)
                            history.append(amps.copy())
                    second = 2 * np.exp(-2j * time) @ np.array(history) / len(time)
                    v2 = (2 / 3) * 2j * w * m2 * second[:, None]
                    i2 = np.linalg.solve(z2, v2[..., None])[..., 0]
                    half = np.linalg.norm(i2 - np.roll(i2, len(i2) // 2, axis=0), axis=1) * 1e6
                    signals.append(
                        dict(
                            turns=turns,
                            length_mm=length * 1000,
                            hz=hz,
                            probe_peak_v=voltage,
                            diode_drop_v=vf,
                            max_induced_v=abs(voltage_oc).max(),
                            matched_best_ac_uw=1e6 * max(abs(voltage_oc)) ** 2 / (8 * r),
                            max_switched_current_ua=1e6 * np.linalg.norm(return_i, axis=1).max(),
                            max_second_harmonic_ua=1e6 * np.linalg.norm(i2, axis=1).max(),
                            min_opposed_difference_ua=half.min(),
                            max_opposed_difference_ua=half.max(),
                        )
                    )
    write_csv(root / "harvest.csv", harvest)
    write_csv(root / "backscatter.csv", signals)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stock", type=Path, required=True)
    a = p.parse_args()
    if not (a.out / "harmonic_ports.npz").exists():
        harmonic_ports(a.out)
    circuit(a.out, a.stock)
