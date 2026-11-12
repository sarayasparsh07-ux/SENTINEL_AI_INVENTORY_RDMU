"""Observation model: four evidence channels with reliability-adjusted likelihoods.

Nominal likelihoods  Z_k(o | s)  describe a channel that is working properly.
An unreliable channel is modelled with a contamination mixture:

    Z_{k,r}(o | s) = r * Z_k(o | s) + (1 - r) * u_k(o),      u_k(o) = 1 / |O_k|

Generative reading: with probability r the channel reports a genuine reading
drawn from Z_k(. | s); with probability 1 - r it reports state-independent
noise. Consequences (all unit-tested):

* Z_{k,r}(. | s) is a valid distribution over observations for every state,
  so the simulator can sample from exactly the model the agent assumes.
* r = 1 gives the nominal model; r = 0 gives a likelihood that is identical
  across states, so the posterior equals the predicted belief.
* For r < 1 every observation has positive probability in every state.

An observation is EVIDENCE about the hidden state. It is not the hidden state.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .config import (CHANNEL_KEYS, CHANNELS, EPSILON, N_STATES, STATES,
                     SentinelConfig, observation_index)
from .transition_model import validate_stochastic_matrix


def floor_and_renormalise(matrix: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    """Floor likelihoods at epsilon and renormalise each row over observations."""
    m = np.maximum(np.asarray(matrix, dtype=float), epsilon)
    return m / m.sum(axis=1, keepdims=True)


class ObservationModel:
    """Per-channel likelihood matrices Z_k[s, o] plus channel reliabilities."""

    def __init__(self, likelihoods: Dict[str, np.ndarray], reliability: Dict[str, float],
                 epsilon: float = EPSILON):
        self.epsilon = float(epsilon)
        self.nominal: Dict[str, np.ndarray] = {}
        self.reliability: Dict[str, float] = {}
        for key in CHANNEL_KEYS:
            z = np.asarray(likelihoods[key], dtype=float)
            n_obs = len(CHANNELS[key]["observations"])
            if z.shape != (N_STATES, n_obs):
                raise ValueError(f"Likelihood matrix '{key}' must be {N_STATES} x {n_obs}.")
            validate_stochastic_matrix(z, f"Likelihood matrix '{key}'")
            self.nominal[key] = floor_and_renormalise(z, self.epsilon)
            r = float(reliability[key])
            self._check_reliability(r)
            self.reliability[key] = r

    @classmethod
    def from_config(cls, config: SentinelConfig) -> "ObservationModel":
        return cls(config.likelihoods, config.reliability, config.epsilon)

    @staticmethod
    def _check_reliability(r: float) -> None:
        if not np.isfinite(r) or not (0.0 <= r <= 1.0):
            raise ValueError(f"Reliability must lie in [0, 1]; got {r!r}.")

    # ------------------------------------------------------------------ helpers
    def n_observations(self, channel: str) -> int:
        return self.nominal[channel].shape[1]

    def labels(self, channel: str):
        return list(CHANNELS[channel]["observations"])

    def effective_matrix(self, channel: str, reliability: Optional[float] = None) -> np.ndarray:
        """Contamination-mixture matrix Z_{k,r}[s, o]."""
        if channel not in self.nominal:
            raise ValueError(f"Unknown observation channel: {channel!r}")
        r = self.reliability[channel] if reliability is None else float(reliability)
        self._check_reliability(r)
        z = self.nominal[channel]
        noise = np.full_like(z, 1.0 / z.shape[1])
        return r * z + (1.0 - r) * noise

    def likelihood_vector(self, channel: str, observation, reliability: Optional[float] = None) -> np.ndarray:
        """Vector [Z_r(o | Healthy), Z_r(o | Low), Z_r(o | Critical)] for one observation."""
        idx = observation_index(channel, observation)
        return self.effective_matrix(channel, reliability)[:, idx].copy()

    def sample(self, channel: str, state: int, rng: np.random.Generator,
               reliability: Optional[float] = None) -> int:
        """Draw one observation index for a given TRUE state (simulator use only)."""
        probs = self.effective_matrix(channel, reliability)[int(state)]
        return int(rng.choice(len(probs), p=probs))

    def as_frame(self, channel: str, reliability: Optional[float] = None):
        import pandas as pd
        return pd.DataFrame(self.effective_matrix(channel, reliability),
                            index=[f"state = {s}" for s in STATES], columns=self.labels(channel))

    def validate(self) -> None:
        for key in CHANNEL_KEYS:
            validate_stochastic_matrix(self.nominal[key], f"Nominal likelihood '{key}'")
            validate_stochastic_matrix(self.effective_matrix(key), f"Effective likelihood '{key}'")
