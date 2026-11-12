"""Evaluation metrics for beliefs and policies (simulation only).

Belief quality needs the hidden ground truth, so every function here is an
EVALUATION tool: nothing in this module is used by the operational agent.

Scoring conventions (all negatively oriented: lower is better):

* Multiclass Brier score, original Brier (1950) form
      BS = (1/N) sum_n sum_r (f_nr - y_nr)^2          range [0, 2]
  e.g. belief [0.60, 0.30, 0.10] scores 0.26 / 0.86 / 1.26 when the truth is
  Healthy / Low / Critical.
* Log loss  -(1/N) sum_n ln b_n(true state), beliefs clipped at 1e-12.
* Top-label calibration: confidence = max_s b(s), accuracy = [argmax = truth],
  Expected Calibration Error = sum_bins (n_bin / N) |accuracy - confidence|.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .config import N_STATES, STATES

BELIEF_COLUMNS = ["b_healthy", "b_low", "b_critical"]
LOG_CLIP = 1e-12


def _as_arrays(beliefs, true_states):
    b = np.atleast_2d(np.asarray(beliefs, dtype=float))
    y = np.atleast_1d(np.asarray(true_states, dtype=int))
    if b.shape[1] != N_STATES or b.shape[0] != y.shape[0]:
        raise ValueError("beliefs must be (n, 3) and true_states must have length n.")
    return b, y


def brier_scores(beliefs, true_states) -> np.ndarray:
    """Per-forecast multiclass Brier score (original 0-2 form)."""
    b, y = _as_arrays(beliefs, true_states)
    one_hot = np.eye(N_STATES)[y]
    return ((b - one_hot) ** 2).sum(axis=1)


def brier_score(beliefs, true_states) -> float:
    return float(brier_scores(beliefs, true_states).mean())


def log_loss(beliefs, true_states) -> float:
    b, y = _as_arrays(beliefs, true_states)
    p_true = np.clip(b[np.arange(len(y)), y], LOG_CLIP, 1.0)
    return float(-np.log(p_true).mean())


def map_accuracy(beliefs, true_states) -> float:
    """Share of periods in which the most probable state equals the true state."""
    b, y = _as_arrays(beliefs, true_states)
    return float((b.argmax(axis=1) == y).mean())


def calibration_table(beliefs, true_states, n_bins: int = 10, mode: str = "top_label",
                      state: Optional[int] = None) -> pd.DataFrame:
    """Reliability-diagram data.

    mode="top_label": predicted = max belief, outcome = 1 if argmax is correct.
    mode="state":     predicted = b(state),   outcome = 1 if truth == state.
    """
    b, y = _as_arrays(beliefs, true_states)
    if mode == "top_label":
        predicted = b.max(axis=1)
        outcome = (b.argmax(axis=1) == y).astype(float)
    elif mode == "state":
        if state is None:
            raise ValueError("state must be given when mode='state'.")
        predicted = b[:, int(state)]
        outcome = (y == int(state)).astype(float)
    else:
        raise ValueError("mode must be 'top_label' or 'state'.")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(predicted, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        mask = idx == k
        if mask.any():
            rows.append({"bin_lower": edges[k], "bin_upper": edges[k + 1],
                         "mean_predicted": float(predicted[mask].mean()),
                         "observed_frequency": float(outcome[mask].mean()),
                         "count": int(mask.sum())})
    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> float:
    if table.empty:
        return float("nan")
    weights = table["count"] / table["count"].sum()
    return float((weights * (table["observed_frequency"] - table["mean_predicted"]).abs()).sum())


def count_policy_switches(actions: Sequence[int]) -> int:
    a = np.asarray(list(actions))
    return int((a[1:] != a[:-1]).sum()) if len(a) > 1 else 0


def recovery_times(states: Sequence[int]) -> list:
    """Number of periods from entering Critical until the next Healthy period."""
    times, start = [], None
    for t, s in enumerate(states):
        if s == 2 and start is None:
            start = t
        elif s == 0 and start is not None:
            times.append(t - start)
            start = None
    return times


def summarise_episode(frame: pd.DataFrame) -> Dict[str, float]:
    """Scalar metrics for one episode frame produced by `run_episode`."""
    beliefs = frame[BELIEF_COLUMNS].to_numpy()
    truth = frame["hidden_state"].to_numpy()
    recoveries = recovery_times(truth)
    return {
        "total_cost": float(frame["cost"].sum()),
        "mean_cost": float(frame["cost"].mean()),
        "stockout_rate": float(frame["stockout"].mean()),
        "service_level": float(frame["units_served"].sum() / frame["demand_units"].sum()),
        "critical_share": float((truth == 2).mean()),
        "healthy_share": float((truth == 0).mean()),
        "intervention_rate": float((frame["action"] != 0).mean()),
        "policy_switches": float(count_policy_switches(frame["action"])),
        "mean_recovery_periods": float(np.mean(recoveries)) if recoveries else float("nan"),
        "brier": brier_score(beliefs, truth),
        "log_loss": log_loss(beliefs, truth),
        "accuracy": map_accuracy(beliefs, truth),
    }


def mean_confidence_interval(values, level: float = 0.95) -> Dict[str, float]:
    """Mean with a two-sided Student-t confidence interval across replications."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan"), "half_width": float("nan"), "n": 0}
    mean = float(x.mean())
    if n == 1:
        return {"mean": mean, "low": mean, "high": mean, "half_width": 0.0, "n": 1}
    half = float(stats.t.ppf(0.5 + level / 2.0, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return {"mean": mean, "low": mean - half, "high": mean + half, "half_width": half, "n": n}


def summarise_replications(results: pd.DataFrame, metrics: Optional[Sequence[str]] = None,
                           level: float = 0.95) -> pd.DataFrame:
    """Mean and confidence half-width per policy for each metric."""
    metrics = list(metrics or ["total_cost", "stockout_rate", "service_level", "critical_share",
                               "intervention_rate", "brier", "log_loss", "accuracy"])
    rows = []
    for policy, group in results.groupby("policy", sort=False):
        row = {"policy": policy, "replications": len(group)}
        for m in metrics:
            ci = mean_confidence_interval(group[m], level)
            row[m] = ci["mean"]
            row[f"{m}_ci"] = ci["half_width"]
        rows.append(row)
    return pd.DataFrame(rows)


def paired_difference(results: pd.DataFrame, policy_a: str, policy_b: str, metric: str = "total_cost",
                      level: float = 0.95) -> Dict[str, float]:
    """Mean paired difference (a - b) across common-random-number replications."""
    a = results[results["policy"] == policy_a].set_index("replication")[metric]
    b = results[results["policy"] == policy_b].set_index("replication")[metric]
    return mean_confidence_interval((a - b).dropna().to_numpy(), level)


def gap_closed(cost_baseline: float, cost_agent: float, cost_oracle: float) -> float:
    """Share of the baseline-to-oracle cost gap that the agent closes.

    1.0 = as good as the full-information myopic oracle, 0.0 = no better than
    the baseline, negative = worse than the baseline.
    """
    denominator = cost_baseline - cost_oracle
    if abs(denominator) < 1e-12:
        return float("nan")
    return float((cost_baseline - cost_agent) / denominator)


def state_name(index: int) -> str:
    return STATES[int(index)]
