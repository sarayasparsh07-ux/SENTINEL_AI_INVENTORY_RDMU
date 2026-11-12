"""Exact discrete Bayes filter for the belief state (prediction + correction).

Stage 1 - prediction (only when an action has actually been executed):
    b^-(s') = sum_s T(s' | s, a) b(s)

Stage 2 - correction with ALL observations received in the period:
    b'(s') = eta * [prod_k Z_k(o_k | s')] * b^-(s'),
    eta = 1 / sum_s' [prod_k Z_k(o_k | s')] b^-(s')        (eta^-1 = P(o_1..o_K | b, a))

The product over channels is the conditional-independence assumption: given
the hidden state, the channels err independently. The correction is carried
out in log-space with the log-sum-exp trick so that many small likelihoods
cannot underflow. Evidence that the model regards as impossible (zero total
probability) is rejected: the predicted belief is returned and flagged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .config import N_STATES

SUM_TOLERANCE = 1e-9


class BeliefError(ValueError):
    """Raised when a belief vector or likelihood vector is invalid."""


def validate_belief(belief, name: str = "belief", atol: float = 1e-8) -> np.ndarray:
    """Return the belief as a float array, or raise BeliefError."""
    b = np.asarray(belief, dtype=float)
    if b.shape != (N_STATES,):
        raise BeliefError(f"{name} must have {N_STATES} entries; got shape {b.shape}.")
    if not np.all(np.isfinite(b)):
        raise BeliefError(f"{name} contains non-finite values.")
    if np.any(b < -atol) or np.any(b > 1 + atol):
        raise BeliefError(f"{name} has probabilities outside [0, 1]: {b}.")
    if abs(b.sum() - 1.0) > 1e-6:
        raise BeliefError(f"{name} must sum to 1; it sums to {b.sum():.8f}.")
    return b


def normalise(weights) -> np.ndarray:
    """Normalise non-negative weights to a probability vector."""
    w = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise BeliefError("Weights must be finite and non-negative.")
    total = w.sum()
    if total <= 0:
        raise BeliefError("Cannot normalise: all weights are zero.")
    return w / total


def validate_likelihood(likelihood, name: str = "likelihood") -> np.ndarray:
    z = np.asarray(likelihood, dtype=float)
    if z.shape != (N_STATES,):
        raise BeliefError(f"{name} must have {N_STATES} entries; got shape {z.shape}.")
    if not np.all(np.isfinite(z)) or np.any(z < 0) or np.any(z > 1 + 1e-12):
        raise BeliefError(f"{name} must contain probabilities in [0, 1]: {z}.")
    return z


def predict(belief, transition_matrix) -> np.ndarray:
    """Prediction step b^- = b @ T_a (rows of T_a = current state)."""
    b = validate_belief(belief)
    t = np.asarray(transition_matrix, dtype=float)
    if t.shape != (N_STATES, N_STATES):
        raise BeliefError("Transition matrix must be 3 x 3.")
    predicted = b @ t
    return predicted / predicted.sum()   # guard against float drift only


@dataclass
class Correction:
    """Result of one correction step."""
    posterior: np.ndarray
    joint_likelihood: np.ndarray       # prod_k Z_k(o_k | s') for each state
    evidence_probability: float        # eta^-1 = P(observations | predicted belief)
    log_evidence: float
    impossible: bool                   # True when the evidence has zero probability under the model
    n_observations: int

    @property
    def surprise(self) -> float:
        """-log P(observations | b^-), a diagnostic of how unexpected the evidence was."""
        return float("inf") if self.impossible else -self.log_evidence


def _safe_log(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(x)


def correct(predicted_belief, likelihood_vectors: Optional[Iterable] = None) -> Correction:
    """Joint Bayesian correction in log-space.

    `likelihood_vectors` is an iterable of length-3 vectors, one per observation,
    each giving Z(o | s') for s' = Healthy, Low, Critical.
    """
    prior = validate_belief(predicted_belief, "predicted belief")
    vectors: List[np.ndarray] = [validate_likelihood(v) for v in (likelihood_vectors or [])]
    if not vectors:
        return Correction(prior.copy(), np.ones(N_STATES), 1.0, 0.0, False, 0)

    log_joint = np.zeros(N_STATES)
    for v in vectors:
        log_joint = log_joint + _safe_log(v)
    log_unnormalised = _safe_log(prior) + log_joint

    finite = np.isfinite(log_unnormalised)
    if not finite.any():
        # Every state is ruled out: the evidence is impossible under the model.
        return Correction(prior.copy(), np.exp(log_joint), 0.0, float("-inf"), True, len(vectors))

    peak = log_unnormalised[finite].max()
    log_evidence = peak + np.log(np.sum(np.exp(log_unnormalised[finite] - peak)))
    posterior = np.zeros(N_STATES)
    posterior[finite] = np.exp(log_unnormalised[finite] - log_evidence)
    posterior = posterior / posterior.sum()

    if abs(posterior.sum() - 1.0) > SUM_TOLERANCE or np.any(posterior < 0) or np.any(posterior > 1 + 1e-12):
        raise BeliefError(f"Posterior failed the probability checks: {posterior}")
    return Correction(posterior, np.exp(log_joint), float(np.exp(log_evidence)), float(log_evidence),
                      False, len(vectors))


def correct_sequential(predicted_belief, likelihood_vectors: Sequence) -> List[np.ndarray]:
    """Apply the observations one at a time (no prediction in between).

    Returns the list of intermediate posteriors. Under conditional independence
    the last element equals the joint correction, in any order (unit-tested).
    """
    belief = validate_belief(predicted_belief, "predicted belief")
    steps: List[np.ndarray] = []
    for v in likelihood_vectors:
        result = correct(belief, [v])
        belief = result.posterior
        steps.append(belief.copy())
    return steps


@dataclass
class BeliefUpdate:
    """Full two-stage update, kept for display in the Bayesian update explorer."""
    prior: np.ndarray
    predicted: np.ndarray
    joint_likelihood: np.ndarray
    posterior: np.ndarray
    evidence_probability: float
    impossible: bool


def bayes_update(belief, transition_matrix, likelihood_vectors: Optional[Iterable] = None) -> BeliefUpdate:
    """One prediction followed by one joint correction (one simulation period)."""
    prior = validate_belief(belief)
    predicted = predict(prior, transition_matrix)
    corr = correct(predicted, likelihood_vectors)
    return BeliefUpdate(prior, predicted, corr.joint_likelihood, corr.posterior,
                        corr.evidence_probability, corr.impossible)


def entropy(belief) -> float:
    """Shannon entropy in nats (0 * log 0 treated as 0)."""
    b = np.asarray(belief, dtype=float)
    nz = b[b > 0]
    return float(-(nz * np.log(nz)).sum())


def belief_concentration(belief) -> float:
    """1 - H(b) / ln(3): 0 for a uniform belief, 1 when one state is certain.

    This is the dashboard's "belief concentration" index. It measures how
    peaked the belief is; it is not the probability of being right.
    """
    return float(1.0 - entropy(belief) / np.log(N_STATES))


def total_variation(p, q) -> float:
    """Total variation distance between two beliefs (0 = identical, 1 = disjoint)."""
    return float(0.5 * np.abs(np.asarray(p, dtype=float) - np.asarray(q, dtype=float)).sum())
