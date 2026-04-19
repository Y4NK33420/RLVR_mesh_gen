from __future__ import annotations

from r4.lagrangian import PIConstraintController


def test_lambda_increases_when_observed_cost_above_target() -> None:
    controller = PIConstraintController(lambda_value=0.01, kp=0.1, ki=0.02, target_cost=0.05)

    new_lambda = controller.update(observed_cost=0.25)

    assert new_lambda > 0.01


def test_lambda_respects_min_bound() -> None:
    controller = PIConstraintController(
        lambda_value=0.01,
        kp=0.1,
        ki=0.02,
        target_cost=0.20,
        min_lambda=0.0,
    )

    for _ in range(8):
        controller.update(observed_cost=0.0)

    assert controller.lambda_value >= 0.0
