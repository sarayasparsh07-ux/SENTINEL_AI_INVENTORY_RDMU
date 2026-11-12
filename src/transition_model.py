"""State transition model T(s' | s, a) with validated, action-specific matrices.

Two optional environment modifiers turn the base matrices into the *effective*
matrices used for both prediction and simulation:

* supplier delivery risk  rho in [0, 1]
    A supplier-dependent action (Standard Reorder, Expedite) fails to land with
    probability rho, in which case the period evolves as if nothing had been
    ordered:      T'_a = (1 - rho) * T_a + rho * T_monitor
    Reallocation uses internal stock and is unaffected.

* demand volatility  kappa in [0, 0.5]
    After the action's effect, inventory slips one extra level with
    probability kappa:      T_eff_a = T'_a @ D(kappa),
    D(kappa) = (1 - kappa) * I + kappa * Shift,  Shift: H->L, L->C, C->C.

Both operations are mixtures / products of row-stochastic matrices, so the
effective matrices remain row-stochastic.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .config import (ACTION_KEYS, N_ACTIONS, N_STATES, STATES,
                     SUPPLIER_DEPENDENT_ACTIONS, SentinelConfig, action_index)

_SHIFT = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])


def validate_stochastic_matrix(matrix: np.ndarray, name: str = "matrix", atol: float = 1e-8) -> None:
    """Raise ValueError unless `matrix` is a valid row-stochastic matrix."""
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    if not np.all(np.isfinite(m)):
        raise ValueError(f"{name} contains non-finite values.")
    if np.any(m < -atol) or np.any(m > 1 + atol):
        raise ValueError(f"{name} contains values outside [0, 1].")
    sums = m.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=atol):
        bad = int(np.argmax(np.abs(sums - 1.0)))
        raise ValueError(f"{name}: row {bad} sums to {sums[bad]:.6f}, expected 1.")


class TransitionModel:
    """Action-specific transition probabilities with environment modifiers."""

    def __init__(self, base: Dict[str, np.ndarray], demand_volatility: float = 0.0,
                 supplier_risk: float = 0.0):
        if not (0.0 <= demand_volatility <= 0.5):
            raise ValueError("demand_volatility must lie in [0, 0.5].")
        if not (0.0 <= supplier_risk <= 1.0):
            raise ValueError("supplier_risk must lie in [0, 1].")
        self.base = {k: np.asarray(base[k], dtype=float).copy() for k in ACTION_KEYS}
        for key, m in self.base.items():
            if m.shape != (N_STATES, N_STATES):
                raise ValueError(f"Transition matrix '{key}' must be {N_STATES} x {N_STATES}.")
            validate_stochastic_matrix(m, f"Transition matrix '{key}'")
        self.demand_volatility = float(demand_volatility)
        self.supplier_risk = float(supplier_risk)
        self._effective = np.stack([self._build(k) for k in ACTION_KEYS])
        for i, key in enumerate(ACTION_KEYS):
            validate_stochastic_matrix(self._effective[i], f"Effective transition matrix '{key}'")

    @classmethod
    def from_config(cls, config: SentinelConfig) -> "TransitionModel":
        return cls(config.transitions, config.demand_volatility, config.supplier_risk)

    def _build(self, key: str) -> np.ndarray:
        m = self.base[key]
        if key in SUPPLIER_DEPENDENT_ACTIONS and self.supplier_risk > 0:
            m = (1.0 - self.supplier_risk) * m + self.supplier_risk * self.base["monitor"]
        if self.demand_volatility > 0:
            depletion = (1.0 - self.demand_volatility) * np.eye(N_STATES) + self.demand_volatility * _SHIFT
            m = m @ depletion
        return m

    def matrix(self, action) -> np.ndarray:
        """Effective T_a with rows = current state and columns = next state."""
        return self._effective[action_index(action)].copy()

    def tensor(self) -> np.ndarray:
        """Array of shape (actions, states, states)."""
        return self._effective.copy()

    def predict(self, belief: np.ndarray, action) -> np.ndarray:
        """Prediction step  b^-(s') = sum_s T(s'|s,a) b(s)."""
        return np.asarray(belief, dtype=float) @ self._effective[action_index(action)]

    def as_frame(self, action):
        import pandas as pd
        return pd.DataFrame(self.matrix(action), index=[f"from {s}" for s in STATES],
                            columns=[f"to {s}" for s in STATES])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"TransitionModel(actions={N_ACTIONS}, demand_volatility={self.demand_volatility}, "
                f"supplier_risk={self.supplier_risk})")
