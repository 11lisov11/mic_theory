from __future__ import annotations

from dataclasses import dataclass
import math

from config.env import MotorParams


@dataclass(frozen=True)
class CurrentFluxMotorParams:
    """Induction-motor parameters for the independent i_s/psi_r model."""

    rs_ohm: float
    rr_ohm: float
    ls_h: float
    lr_h: float
    lm_h: float
    inertia_kgm2: float
    viscous_b_nms: float
    pole_pairs: int
    coulomb_friction_nm: float = 0.0
    current_limit_a: float = 20.0

    @classmethod
    def from_motor_params(
        cls,
        params: MotorParams,
        *,
        coulomb_friction_nm: float = 0.0,
        current_limit_a: float | None = None,
    ) -> "CurrentFluxMotorParams":
        return cls(
            rs_ohm=float(params.Rs),
            rr_ohm=float(params.Rr),
            ls_h=float(params.Ls_sigma + params.Lm),
            lr_h=float(params.Lr_sigma + params.Lm),
            lm_h=float(params.Lm),
            inertia_kgm2=float(params.J),
            viscous_b_nms=float(params.B),
            pole_pairs=int(params.p),
            coulomb_friction_nm=float(coulomb_friction_nm),
            current_limit_a=float(
                current_limit_a
                if current_limit_a is not None
                else 6.0 * float(getattr(params, "I_n", 1.0) or 1.0)
            ),
        )

    def __post_init__(self) -> None:
        for name in (
            "rs_ohm",
            "rr_ohm",
            "ls_h",
            "lr_h",
            "lm_h",
            "inertia_kgm2",
            "current_limit_a",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.lm_h**2 >= self.ls_h * self.lr_h:
            raise ValueError("inductance matrix must be positive definite")
        if self.pole_pairs <= 0:
            raise ValueError("pole_pairs must be positive")


@dataclass(frozen=True)
class CurrentFluxMotorState:
    i_s_alpha_a: float = 0.0
    i_s_beta_a: float = 0.0
    psi_r_alpha_wb: float = 0.0
    psi_r_beta_wb: float = 0.0
    omega_m_rad_s: float = 0.0
    theta_m_rad: float = 0.0


@dataclass(frozen=True)
class CurrentFluxMotorOutput:
    state: CurrentFluxMotorState
    torque_nm: float
    current_abs_a: float


def _state_add(state: CurrentFluxMotorState, derivative: tuple[float, ...], scale: float) -> CurrentFluxMotorState:
    values = tuple(float(getattr(state, field)) + scale * derivative[index] for index, field in enumerate(state.__dataclass_fields__))
    return CurrentFluxMotorState(*values)


class CurrentFluxInductionMotorRk4:
    """Structurally independent stationary-frame model integrated with RK4.

    State variables are stator current and rotor flux. The existing research
    plant uses stator/rotor flux and explicit Euler integration, so this model
    provides a separate numerical and state-space implementation for holdout
    tests. It is still a simulation model, not experimental validation.
    """

    def __init__(self, params: CurrentFluxMotorParams, state: CurrentFluxMotorState | None = None) -> None:
        self.params = params
        self.state = state or CurrentFluxMotorState()

    def _derivative(
        self,
        state: CurrentFluxMotorState,
        v_alpha_v: float,
        v_beta_v: float,
        load_torque_nm: float,
    ) -> tuple[float, float, float, float, float, float]:
        p = self.params
        sigma = 1.0 - p.lm_h**2 / (p.ls_h * p.lr_h)
        sigma_ls = max(sigma * p.ls_h, 1e-12)
        rotor_rate = p.rr_ohm / p.lr_h
        omega_e = p.pole_pairs * state.omega_m_rad_s
        dpsi_a = (
            -rotor_rate * state.psi_r_alpha_wb
            - omega_e * state.psi_r_beta_wb
            + rotor_rate * p.lm_h * state.i_s_alpha_a
        )
        dpsi_b = (
            -rotor_rate * state.psi_r_beta_wb
            + omega_e * state.psi_r_alpha_wb
            + rotor_rate * p.lm_h * state.i_s_beta_a
        )
        coupling = p.lm_h / p.lr_h
        di_a = (
            float(v_alpha_v)
            - p.rs_ohm * state.i_s_alpha_a
            - coupling * dpsi_a
        ) / sigma_ls
        di_b = (
            float(v_beta_v)
            - p.rs_ohm * state.i_s_beta_a
            - coupling * dpsi_b
        ) / sigma_ls
        torque = 1.5 * p.pole_pairs * coupling * (
            state.psi_r_alpha_wb * state.i_s_beta_a
            - state.psi_r_beta_wb * state.i_s_alpha_a
        )
        friction = p.viscous_b_nms * state.omega_m_rad_s
        if abs(state.omega_m_rad_s) > 1e-6:
            friction += math.copysign(p.coulomb_friction_nm, state.omega_m_rad_s)
        domega = (torque - float(load_torque_nm) - friction) / p.inertia_kgm2
        return di_a, di_b, dpsi_a, dpsi_b, domega, state.omega_m_rad_s

    def torque_nm(self, state: CurrentFluxMotorState | None = None) -> float:
        state = state or self.state
        coupling = self.params.lm_h / self.params.lr_h
        return 1.5 * self.params.pole_pairs * coupling * (
            state.psi_r_alpha_wb * state.i_s_beta_a
            - state.psi_r_beta_wb * state.i_s_alpha_a
        )

    def step(
        self,
        v_alpha_v: float,
        v_beta_v: float,
        load_torque_nm: float,
        dt_s: float,
    ) -> CurrentFluxMotorOutput:
        dt = float(dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        values = (v_alpha_v, v_beta_v, load_torque_nm)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("voltage and load inputs must be finite")

        state = self.state
        k1 = self._derivative(state, v_alpha_v, v_beta_v, load_torque_nm)
        k2 = self._derivative(_state_add(state, k1, 0.5 * dt), v_alpha_v, v_beta_v, load_torque_nm)
        k3 = self._derivative(_state_add(state, k2, 0.5 * dt), v_alpha_v, v_beta_v, load_torque_nm)
        k4 = self._derivative(_state_add(state, k3, dt), v_alpha_v, v_beta_v, load_torque_nm)
        derivative = tuple((k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0 for i in range(6))
        next_state = _state_add(state, derivative, dt)
        current_abs = math.hypot(next_state.i_s_alpha_a, next_state.i_s_beta_a)
        if not all(math.isfinite(float(value)) for value in next_state.__dict__.values()):
            raise FloatingPointError("independent plant state became non-finite")
        if current_abs > self.params.current_limit_a:
            raise FloatingPointError(
                f"independent plant current {current_abs:.6g} A exceeds numerical guard {self.params.current_limit_a:.6g} A"
            )
        self.state = next_state
        return CurrentFluxMotorOutput(
            state=next_state,
            torque_nm=self.torque_nm(next_state),
            current_abs_a=current_abs,
        )


__all__ = [
    "CurrentFluxInductionMotorRk4",
    "CurrentFluxMotorOutput",
    "CurrentFluxMotorParams",
    "CurrentFluxMotorState",
]
