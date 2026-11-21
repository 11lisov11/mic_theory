import json
from pathlib import Path

import numpy as np
import pytest


SLOW_TEST_FILES = {
    "test_ao2_hardening_sweep_smoke.py",
    "test_frozen_mini_baselines.py",
    "test_reproduce_ieee_step28_smoke.py",
    "test_robust_motor_hardening_smoke.py",
    "test_run_integration_pipeline_smoke.py",
    "test_run_step27_extended_repro_smoke.py",
    "test_step27_pipeline_smoke.py",
    "test_train_3motors_pipeline_joint_and_finetune_smoke.py",
    "test_train_3motors_pipeline_resume_eval_first_smoke.py",
    "test_train_3motors_pipeline_smoke.py",
    "test_train_any_motor_pipeline_smoke.py",
}

SLOW_TEST_PREFIXES = (
    "test_run_external_step27",
    "test_promote_external_step27",
)


def pytest_configure() -> None:
    np.random.seed(0)
    try:
        import torch

        torch.manual_seed(0)
    except Exception:
        pass


def pytest_collection_modifyitems(items) -> None:
    slow_marker = pytest.mark.slow
    for item in items:
        filename = item.path.name
        if filename in SLOW_TEST_FILES or any(item.name.startswith(prefix) for prefix in SLOW_TEST_PREFIXES):
            item.add_marker(slow_marker)


@pytest.fixture
def smoke_checkpoint_registry(tmp_path: Path) -> Path:
    """Supply a valid synthetic actor without requiring unpublished checkpoints."""
    import torch

    from mic_ai.ai.agents.ppo_voltage import PPOVoltageAgent
    from mic_ai.ai.train_ai_id_ref import build_feature_keys

    with torch.random.fork_rng(devices=[]):
        agent = PPOVoltageAgent(
            feature_keys=build_feature_keys(True, False),
            action_dim=1,
            device="cpu",
            hidden_sizes=(64, 64),
        )
    state = {name: torch.zeros_like(tensor) for name, tensor in agent.net.state_dict().items()}
    motors = {}
    for motor in ("air56", "al31", "ao2"):
        checkpoint = tmp_path / f"{motor}_actor.pth"
        torch.save(state, checkpoint)
        motors[motor] = str(checkpoint)
    registry = tmp_path / "checkpoint_registry.json"
    registry.write_text(json.dumps({"motors": motors}), encoding="utf-8")
    return registry
