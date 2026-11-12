"""Scripted alert scenarios shared by the notebook (A-H) and the dashboard presets.

Scenarios are deterministic: every alert is scripted, nothing is sampled, so
the notebook and the dashboard reproduce exactly the same numbers.

A stage is processed in this order:
    1. if `advance` is True, an action is EXECUTED (the current recommendation
       when `execute` is None, otherwise the named action) and one belief
       prediction is made for the new period;
    2. the stage's alerts are injected and jointly corrected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Alert = Tuple  # (channel, observation label) or (channel, observation label, reliability)


@dataclass
class Stage:
    alerts: List[Alert] = field(default_factory=list)
    advance: bool = False
    execute: Optional[str] = None      # None = execute the current recommendation
    note: str = ""


@dataclass
class Scenario:
    key: str
    name: str
    description: str
    stages: List[Stage]
    notebook_label: str = ""


SCENARIOS: Dict[str, Scenario] = {}


def _add(scenario: Scenario) -> None:
    SCENARIOS[scenario.key] = scenario


_add(Scenario(
    "normal_ops", "Normal Inventory Operations",
    "All four channels report nominal conditions over three periods. Belief should concentrate on "
    "Healthy and the agent should keep monitoring.",
    [
        Stage([("sensor", "Normal inventory reading"), ("supplier", "Delivery on schedule"),
               ("demand", "Normal demand"), ("system", "No anomaly")], note="Nominal readings"),
        Stage([("sensor", "Normal inventory reading"), ("demand", "Normal demand")], advance=True),
        Stage([("sensor", "Normal inventory reading"), ("supplier", "Delivery on schedule")], advance=True),
    ], "A"))

_add(Scenario(
    "unreliable_sensor", "Low Reading from an Unreliable Sensor",
    "A single low-inventory reading arrives from a sensor that is only 30% reliable. The belief "
    "should move far less than it would for a trustworthy sensor.",
    [Stage([("sensor", "Low inventory reading", 0.30)], note="Sensor reliability 30%")], "B"))

_add(Scenario(
    "supplier_delay", "Unexpected Supplier Delay",
    "The supplier reports a significant delay, followed one period later by a cancellation and a "
    "low sensor reading.",
    [
        Stage([("supplier", "Significant delivery delay")], note="Delay notice"),
        Stage([("supplier", "Supplier cancellation"), ("sensor", "Low inventory reading")], advance=True,
              note="Cancellation and low reading after executing the recommendation"),
    ], "C"))

_add(Scenario(
    "demand_surge", "Demand Surge",
    "Demand moves from moderately elevated to an unexpected surge, with a low sensor reading "
    "arriving alongside the surge.",
    [
        Stage([("demand", "Moderately elevated demand")], note="Early demand signal"),
        Stage([("demand", "Unexpected demand surge"), ("sensor", "Low inventory reading")], advance=True,
              note="Surge after executing the recommendation"),
    ], "D"))

_add(Scenario(
    "critical_warning", "Critical Inventory Warning",
    "The sensor issues a critical inventory warning and the monitoring system raises a critical "
    "stock warning in the same period.",
    [Stage([("sensor", "Critical inventory warning"), ("system", "Critical stock warning")],
           note="Two critical alerts")], "E"))

_add(Scenario(
    "conflicting", "Conflicting Sensor Evidence",
    "The sensor reads normal while the monitoring system raises a critical stock warning and the "
    "supplier reports a significant delay. The channels disagree; Bayes weighs them by likelihood.",
    [Stage([("sensor", "Normal inventory reading"), ("system", "Critical stock warning"),
            ("supplier", "Significant delivery delay"), ("demand", "Normal demand")],
           note="Channels disagree")], "F"))

_add(Scenario(
    "consecutive", "Multiple Consecutive Alerts",
    "Five periods of mostly worsening evidence while the operator only monitors (manual override). "
    "Shows the belief drifting towards Critical and the recommendation escalating.",
    [
        Stage([("demand", "Moderately elevated demand")], note="Period 0"),
        Stage([("supplier", "Minor delivery delay"), ("sensor", "Normal inventory reading")],
              advance=True, execute="monitor", note="Period 1"),
        Stage([("sensor", "Low inventory reading"), ("demand", "High demand")],
              advance=True, execute="monitor", note="Period 2"),
        Stage([("supplier", "Significant delivery delay"), ("system", "Replenishment / low-stock warning")],
              advance=True, execute="monitor", note="Period 3"),
        Stage([("sensor", "Critical inventory warning"), ("system", "Critical stock warning"),
               ("demand", "Unexpected demand surge")], advance=True, execute="monitor", note="Period 4"),
    ], "G"))

_add(Scenario(
    "recovery", "Successful Replenishment and Recovery",
    "A critical picture triggers an intervention; the recommended action is executed and the "
    "following periods bring improving evidence.",
    [
        Stage([("sensor", "Critical inventory warning"), ("system", "Critical stock warning"),
               ("supplier", "Significant delivery delay")], note="Critical evidence"),
        Stage([("sensor", "Low inventory reading"), ("supplier", "Delivery on schedule")], advance=True,
              note="Recommended action executed; partial recovery signals"),
        Stage([("sensor", "Normal inventory reading"), ("system", "No anomaly"), ("demand", "Normal demand")],
              advance=True, note="Recommended action executed; recovery signals"),
        Stage([("sensor", "Normal inventory reading")], advance=True, note="Stable"),
    ], "H"))

_add(Scenario(
    "simultaneous", "Multiple Simultaneous Alerts",
    "All four channels deliver adverse evidence in the same period. One joint correction combines "
    "them under the conditional-independence assumption.",
    [Stage([("sensor", "Low inventory reading"), ("supplier", "Significant delivery delay"),
            ("demand", "High demand"), ("system", "Replenishment / low-stock warning")],
           note="Four adverse alerts at once")], ""))

# Order of the seven presets required on the dashboard, followed by two extras.
DASHBOARD_PRESETS: List[str] = [
    "normal_ops", "supplier_delay", "demand_surge", "critical_warning", "conflicting",
    "simultaneous", "recovery", "unreliable_sensor", "consecutive",
]

# Scenarios A-H for the notebook, in order.
NOTEBOOK_SCENARIOS: List[str] = [
    "normal_ops", "unreliable_sensor", "supplier_delay", "demand_surge", "critical_warning",
    "conflicting", "consecutive", "recovery",
]

# The three scenarios used for the notebook <-> dashboard numerical cross-check.
CROSS_CHECK_SCENARIOS: List[str] = ["supplier_delay", "conflicting", "recovery"]


def replay_with_primitives(scenario: Scenario, config=None) -> dict:
    """Replay a scenario using ONLY the low-level maths (no SentinelEngine).

    This is the independent path used for the notebook <-> dashboard numerical
    cross-check: it calls `predict`, `correct` and `evaluate_actions` directly
    and returns the final period's prior, predicted belief, joint likelihood,
    posterior, expected cost per action and recommended action.
    """
    import numpy as np

    from .belief_updater import correct, predict
    from .config import action_index, default_config
    from .cost_model import CostModel
    from .decision_policy import evaluate_actions
    from .observation_model import ObservationModel
    from .transition_model import TransitionModel

    config = config or default_config()
    transition = TransitionModel.from_config(config)
    observation = ObservationModel.from_config(config)
    cost = CostModel.from_config(config)

    prior = np.asarray(config.initial_belief, dtype=float).copy()
    predicted = prior.copy()
    posterior = prior.copy()
    joint = np.ones(3)
    period_vectors = []                                  # evidence of the current period
    for stage in scenario.stages:
        if stage.advance:
            recommended = evaluate_actions(posterior, transition, cost).action
            executed = recommended if stage.execute is None else action_index(stage.execute)
            prior = posterior.copy()
            predicted = predict(prior, transition.matrix(executed))     # ONE prediction per period
            period_vectors = []
        period_vectors = period_vectors + [
            observation.likelihood_vector(a[0], a[1], a[2] if len(a) > 2 else None) for a in stage.alerts]
        result = correct(predicted, period_vectors)                      # ONE joint correction
        posterior, joint = result.posterior, result.joint_likelihood
    decision = evaluate_actions(posterior, transition, cost)
    return {"prior": prior, "predicted": predicted, "joint_likelihood": joint, "posterior": posterior,
            "q_values": decision.q_values, "action": int(decision.action)}
