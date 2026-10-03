from __future__ import annotations

import math

import numpy as np
import pytest

from control.posterior_confidence_gate import accept_estimate, hpd_width_deg


def test_width_gate_is_nested_and_uses_only_posterior() -> None:
    concentrated = np.asarray([0.96, 0.02, 0.01, 0.01])
    diffuse = np.asarray([0.25, 0.25, 0.25, 0.25])
    assert hpd_width_deg(concentrated, 12.0) == pytest.approx(3.0)
    assert hpd_width_deg(diffuse, 12.0) == pytest.approx(12.0)
    assert accept_estimate(concentrated, 12.0, 3.0)
    assert not accept_estimate(diffuse, 12.0, 3.0)


@pytest.mark.parametrize("posterior", [np.asarray([0.0, 0.0]),
                                            np.asarray([math.nan, 1.0]),
                                            np.asarray([-1.0, 2.0])])
def test_invalid_posterior_rejected(posterior: np.ndarray) -> None:
    with pytest.raises(ValueError):
        hpd_width_deg(posterior, 12.0)


def test_invalid_gate_threshold_rejected() -> None:
    with pytest.raises(ValueError):
        accept_estimate(np.asarray([0.5, 0.5]), 12.0, math.inf)
