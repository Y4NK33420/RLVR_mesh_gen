from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PIConstraintController:
    """PI controller for updating the Lagrangian penalty on constraint violations."""

    lambda_value: float = 0.01
    kp: float = 0.1
    ki: float = 0.02
    target_cost: float = 0.0
    min_lambda: float = 0.0
    max_lambda: float | None = None
    _integral_error: float = field(default=0.0, init=False, repr=False)

    def update(self, observed_cost: float) -> float:
        error = observed_cost - self.target_cost
        self._integral_error += error

        delta = self.kp * error + self.ki * self._integral_error
        self.lambda_value += delta

        if self.lambda_value < self.min_lambda:
            self.lambda_value = self.min_lambda
        if self.max_lambda is not None and self.lambda_value > self.max_lambda:
            self.lambda_value = self.max_lambda

        return self.lambda_value
