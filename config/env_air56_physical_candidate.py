"""Catalog-based AIR56B2 simulation prior. Separate from the historical AIR56."""

from dataclasses import replace
import json
from pathlib import Path

from config.env import MotorParams, create_default_env


REGISTRY = Path(__file__).resolve().parents[1] / "research/air56_interlaminar_marker/profiles.json"
PROFILE = json.loads(REGISTRY.read_text(encoding="utf-8"))["AIR56_PHYSICAL_CANDIDATE"]
MOTOR = MotorParams(**PROFILE["electrical_prior"]["motor"])
HF_INDUCTANCE_H = MOTOR.Ls_sigma + MOTOR.Lm*MOTOR.Lr_sigma/(MOTOR.Lm+MOTOR.Lr_sigma)
BASE = create_default_env()
ENV = replace(BASE, motor=MOTOR, foc=replace(BASE.foc, id_ref=0.76, iq_limit=1.5),
              inverter=replace(BASE.inverter, Vdc=310.0, f_pwm=10_000.0))
