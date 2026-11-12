"""Centralised model parameters for the SENTINEL POMDP.

EVERY number in this file is a SYNTHETIC SIMULATION ASSUMPTION chosen for
teaching purposes. None of it is estimated from real supply-chain data.

The only values taken from the assignment are the three hidden states, the
four information channels and the illustrative initial belief
b0 = [0.60, 0.30, 0.10] shown on the professor's infographic.

Timing convention used everywhere in the project (frozen before coding):

    period t :  hidden state s_t, agent belief b_t
        1. the agent evaluates Q(b_t, a) and RECOMMENDS a*            (no state change)
        2. an action a_t is EXECUTED (the recommendation or an override)
        3. the hidden state moves  s_{t+1} ~ T(. | s_t, a_t)
        4. the cost C(s_t, a_t, s_{t+1}) is incurred;  R = -C
        5. each channel k emits   o_{t+1}^k ~ Z_k(. | s_{t+1})
        6. the belief is updated with ONE prediction and ONE joint correction
              b^-_{t+1}(s') = sum_s T(s'|s,a_t) b_t(s)
              b_{t+1}(s')  ∝ prod_k Z_k(o^k_{t+1} | s') * b^-_{t+1}(s')

Alerts injected by hand between two executed actions belong to the current
period: they only trigger the correction (step 6, second line) and never a
transition, because no action has been executed.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
STATES: Tuple[str, ...] = ("Healthy", "Low", "Critical")
STATE_LONG: Tuple[str, ...] = ("Healthy Inventory", "Low Inventory", "Critical Inventory")
STATE_DESCRIPTIONS: Tuple[str, ...] = (
    "Adequate inventory available; no immediate replenishment concern.",
    "Inventory is below the preferred operating level; replenishment may become necessary.",
    "Inventory is dangerously low; stock-out and service disruption risk.",
)

ACTION_KEYS: Tuple[str, ...] = ("monitor", "reorder", "expedite", "reallocate")
ACTIONS: Tuple[str, ...] = (
    "Continue Monitoring",
    "Standard Reorder",
    "Expedite Supplier Delivery",
    "Adjust Production / Reallocate Stock",
)
ACTION_SHORT: Tuple[str, ...] = ("Monitor", "Reorder", "Expedite", "Reallocate")
ACTION_DESCRIPTIONS: Tuple[str, ...] = (
    "Keep observing without committing inventory expenditure. Appropriate while "
    "belief mass sits on Healthy and the expected shortage cost is small.",
    "Place a normal replenishment order with standard lead time. Appropriate when "
    "Low inventory is the most likely condition.",
    "Pay a premium for fast replenishment. Appropriate when the probability of "
    "Critical inventory is high enough that the stock-out penalty dominates.",
    "Re-plan production or move stock between locations. Uses internal capacity, "
    "so it does not depend on the external supplier; appropriate for a mixed "
    "Low/Critical picture or when supplier risk is high.",
)
# Actions whose effect depends on the external supplier delivering.
SUPPLIER_DEPENDENT_ACTIONS: Tuple[str, ...] = ("reorder", "expedite")

CHANNEL_KEYS: Tuple[str, ...] = ("sensor", "supplier", "demand", "system")

N_STATES = len(STATES)
N_ACTIONS = len(ACTIONS)

COST_UNIT = "SCU"  # simulation cost units (synthetic; not a real currency)

# --------------------------------------------------------------------------
# Initial belief (from the infographic) and discount factor
# --------------------------------------------------------------------------
INITIAL_BELIEF = np.array([0.60, 0.30, 0.10])

# gamma completes the POMDP tuple. The primary SENTINEL policy is myopic
# (one-step), i.e. it behaves as if gamma = 0. gamma is used only by the
# optional QMDP comparison policy and by discounted-return reporting.
DISCOUNT_FACTOR = 0.95

# --------------------------------------------------------------------------
# Transition model  T[a][s, s'] = P(s_{t+1} = s' | s_t = s, a_t = a)
# Rows = current state, columns = next state (Healthy, Low, Critical).
# --------------------------------------------------------------------------
BASE_TRANSITIONS: Dict[str, np.ndarray] = {
    # No replenishment: demand slowly depletes stock; Critical is "sticky".
    "monitor": np.array([
        [0.85, 0.13, 0.02],
        [0.05, 0.70, 0.25],
        [0.01, 0.09, 0.90],
    ]),
    # Normal order, standard lead time: solid recovery from Low, partial from Critical.
    "reorder": np.array([
        [0.93, 0.06, 0.01],
        [0.50, 0.42, 0.08],
        [0.15, 0.45, 0.40],
    ]),
    # Premium fast delivery: strongest recovery, especially from Critical.
    "expedite": np.array([
        [0.96, 0.035, 0.005],
        [0.75, 0.22, 0.03],
        [0.60, 0.32, 0.08],
    ]),
    # Internal rebalancing: quick partial relief, weaker full recovery than expediting.
    "reallocate": np.array([
        [0.90, 0.09, 0.01],
        [0.60, 0.34, 0.06],
        [0.25, 0.50, 0.25],
    ]),
}

# --------------------------------------------------------------------------
# Observation model  Z_k[s, o] = P(o | s) for channel k  (nominal, r = 1)
# Rows = hidden state (Healthy, Low, Critical); columns = observation symbols.
# "severity" is a display grade for each symbol (it is NOT a multiplier:
# severity is modelled by giving each severity level its own symbol and its
# own likelihood column, which keeps the update strictly Bayesian).
# --------------------------------------------------------------------------
CHANNELS: Dict[str, dict] = {
    "sensor": {
        "label": "Sensor readings",
        "icon": "◈",
        "observations": [
            "Normal inventory reading",
            "Low inventory reading",
            "Critical inventory warning",
            "Sensor malfunction / unreliable reading",
        ],
        "severity": ["Nominal", "Warning", "Critical", "Fault"],
        "likelihood": np.array([
            [0.80, 0.12, 0.03, 0.05],
            [0.15, 0.68, 0.12, 0.05],
            [0.04, 0.21, 0.70, 0.05],
        ]),
        "default_reliability": 0.90,
    },
    "supplier": {
        "label": "Supplier updates",
        "icon": "⛟",
        "observations": [
            "Delivery on schedule",
            "Minor delivery delay",
            "Significant delivery delay",
            "Partial shipment",
            "Supplier cancellation",
        ],
        "severity": ["Nominal", "Advisory", "Warning", "Warning", "Critical"],
        "likelihood": np.array([
            [0.70, 0.17, 0.05, 0.06, 0.02],
            [0.38, 0.27, 0.15, 0.14, 0.06],
            [0.15, 0.22, 0.28, 0.20, 0.15],
        ]),
        "default_reliability": 0.85,
    },
    "demand": {
        "label": "Demand signals",
        "icon": "▲",
        "observations": [
            "Normal demand",
            "Moderately elevated demand",
            "High demand",
            "Unexpected demand surge",
        ],
        "severity": ["Nominal", "Advisory", "Warning", "Critical"],
        "likelihood": np.array([
            [0.62, 0.24, 0.10, 0.04],
            [0.35, 0.32, 0.22, 0.11],
            [0.18, 0.26, 0.32, 0.24],
        ]),
        "default_reliability": 0.80,
    },
    "system": {
        "label": "System alerts",
        "icon": "⚑",
        "observations": [
            "No anomaly",
            "Inventory anomaly detected",
            "Replenishment / low-stock warning",
            "Critical stock warning",
            "Monitoring system malfunction",
        ],
        "severity": ["Nominal", "Advisory", "Warning", "Critical", "Fault"],
        "likelihood": np.array([
            [0.78, 0.09, 0.07, 0.02, 0.04],
            [0.22, 0.20, 0.44, 0.10, 0.04],
            [0.06, 0.16, 0.24, 0.50, 0.04],
        ]),
        "default_reliability": 0.90,
    },
}

# Likelihood floor: the model never asserts that an observation is impossible.
EPSILON = 1e-9

# --------------------------------------------------------------------------
# Cost model (all in synthetic simulation cost units, SCU, per period)
#
#   C(s, a, s') = c_action(a) + c_current(s, a) + c_post(s')
#
#   c_action(a)     direct cost of executing the action
#   c_current(s, a) cost that depends on the CURRENT state: ordering or
#                   re-planning while stock is already Healthy creates surplus
#                   holding / wasted premium / needless disruption
#   c_post(s')      cost of the state the system LANDS in: holding cost when
#                   Healthy, service degradation when Low, stock-out penalty
#                   when Critical
#   Reward:  R(s, a, s') = -C(s, a, s')
# --------------------------------------------------------------------------
ACTION_COST = np.array([0.0, 15.0, 40.0, 22.0])          # monitor, reorder, expedite, reallocate
CURRENT_STATE_COST = np.array([                          # rows = current state, cols = action
    [0.0, 6.0, 10.0, 5.0],   # Healthy: surplus holding / wasted premium / disruption
    [0.0, 0.0, 0.0, 0.0],    # Low
    [0.0, 0.0, 0.0, 0.0],    # Critical
])
POST_STATE_COST = np.array([4.0, 20.0, 150.0])           # Healthy holding, Low degradation, Critical stock-out

# --------------------------------------------------------------------------
# Synthetic, state-conditioned unit proxies (used ONLY by the simulator for
# stock-out and service-level metrics; the operational agent never sees them).
# Each period an inventory SNAPSHOT is drawn uniformly inside the hidden state's
# band; there is no stock roll-forward between periods.
# --------------------------------------------------------------------------
WAREHOUSE_CAPACITY_UNITS = 1000
# State bands for the unit snapshot (rows: Healthy, Low, Critical; columns: lower, upper bound).
# As a partition of the capacity range [0, 1,000] the bands are non-overlapping and gap-free:
#     Healthy [600, 1,000]     Low [250, 600)     Critical [0, 250)
# The simulator draws a snapshot as lower + (upper - lower) * U with U in [0, 1), so every draw lies in
# the half-open interval [lower, upper); the single point 1,000 belongs to Healthy but has probability 0.
INVENTORY_BANDS = np.array([[600.0, 1000.0], [250.0, 600.0], [0.0, 250.0]])


def inventory_band_label(state: int) -> str:
    """Interval notation for a state's unit band, e.g. '[250, 600)'. The top band is closed at capacity."""
    lo, hi = INVENTORY_BANDS[int(state)]
    closing = "]" if hi >= WAREHOUSE_CAPACITY_UNITS else ")"
    return f"[{lo:,.0f}, {hi:,.0f}{closing}"


def check_inventory_bands(bands=None, capacity: float = WAREHOUSE_CAPACITY_UNITS) -> List[str]:
    """Return problems if the bands do not partition [0, capacity] without gaps or overlaps."""
    b = np.asarray(INVENTORY_BANDS if bands is None else bands, dtype=float)
    problems: List[str] = []
    if np.any(b[:, 0] >= b[:, 1]):
        problems.append("Every band needs lower < upper.")
    ordered = b[np.argsort(b[:, 0])]
    if ordered[0, 0] != 0.0:
        problems.append("The lowest band must start at 0 units.")
    if ordered[-1, 1] != capacity:
        problems.append(f"The highest band must end at the capacity ({capacity:,.0f} units).")
    for (lo1, hi1), (lo2, hi2) in zip(ordered[:-1], ordered[1:]):
        if hi1 < lo2:
            problems.append(f"Gap between {hi1:,.0f} and {lo2:,.0f} units.")
        elif hi1 > lo2:
            problems.append(f"Overlap between {lo2:,.0f} and {hi1:,.0f} units.")
    if not (b[0, 0] > b[1, 0] > b[2, 0]):
        problems.append("Bands must be ordered Healthy > Low > Critical.")
    return problems
# Per-period demand (units) given the TRUE demand level (same order as the demand symbols)
DEMAND_MEAN_UNITS = np.array([180.0, 240.0, 310.0, 420.0])
DEMAND_SD_UNITS = np.array([25.0, 30.0, 35.0, 50.0])

# Simulation defaults
DEFAULT_SEED = 42
DEFAULT_HORIZON = 60
SIM_START = "2026-11-11 08:00"   # simulated clock origin; one period = one operating day


@dataclass
class SentinelConfig:
    """All editable model parameters in one object (shared by notebook and dashboard)."""

    initial_belief: np.ndarray = field(default_factory=lambda: INITIAL_BELIEF.copy())
    transitions: Dict[str, np.ndarray] = field(default_factory=lambda: deepcopy(BASE_TRANSITIONS))
    likelihoods: Dict[str, np.ndarray] = field(
        default_factory=lambda: {k: CHANNELS[k]["likelihood"].copy() for k in CHANNEL_KEYS})
    reliability: Dict[str, float] = field(
        default_factory=lambda: {k: float(CHANNELS[k]["default_reliability"]) for k in CHANNEL_KEYS})
    action_cost: np.ndarray = field(default_factory=lambda: ACTION_COST.copy())
    current_state_cost: np.ndarray = field(default_factory=lambda: CURRENT_STATE_COST.copy())
    post_state_cost: np.ndarray = field(default_factory=lambda: POST_STATE_COST.copy())
    # Environment modifiers (0 = base matrices)
    demand_volatility: float = 0.0   # kappa: extra one-level depletion probability per period
    supplier_risk: float = 0.0       # rho: probability a supplier-dependent action fails to land
    discount: float = DISCOUNT_FACTOR
    epsilon: float = EPSILON
    seed: int = DEFAULT_SEED
    horizon: int = DEFAULT_HORIZON

    def copy(self) -> "SentinelConfig":
        return deepcopy(self)

    def validate(self) -> List[str]:
        """Return a list of human-readable problems (empty list = valid)."""
        problems: List[str] = []
        b = np.asarray(self.initial_belief, dtype=float)
        if b.shape != (N_STATES,):
            problems.append("Initial belief must have exactly three entries.")
        elif np.any(~np.isfinite(b)) or np.any(b < 0) or np.any(b > 1):
            problems.append("Initial belief entries must lie between 0% and 100%.")
        elif abs(b.sum() - 1.0) > 1e-6:
            problems.append(f"Initial belief must sum to 100% (currently {100 * b.sum():.2f}%).")
        for key in ACTION_KEYS:
            m = np.asarray(self.transitions.get(key), dtype=float)
            if m.shape != (N_STATES, N_STATES):
                problems.append(f"Transition matrix '{key}' must be 3 x 3.")
                continue
            if np.any(~np.isfinite(m)) or np.any(m < 0) or np.any(m > 1):
                problems.append(f"Transition matrix '{key}' has entries outside [0, 1].")
            sums = m.sum(axis=1)
            for i, s in enumerate(sums):
                if abs(s - 1.0) > 1e-6:
                    problems.append(
                        f"Transition matrix '{key}', row '{STATES[i]}' sums to {s:.4f}, not 1.")
        for key in CHANNEL_KEYS:
            z = np.asarray(self.likelihoods.get(key), dtype=float)
            n_obs = len(CHANNELS[key]["observations"])
            if z.shape != (N_STATES, n_obs):
                problems.append(f"Likelihood matrix '{key}' must be 3 x {n_obs}.")
                continue
            if np.any(~np.isfinite(z)) or np.any(z < 0) or np.any(z > 1):
                problems.append(f"Likelihood matrix '{key}' has entries outside [0, 1].")
            for i, s in enumerate(z.sum(axis=1)):
                if abs(s - 1.0) > 1e-6:
                    problems.append(
                        f"Likelihood matrix '{key}', row '{STATES[i]}' sums to {s:.4f}, not 1.")
            r = self.reliability.get(key, None)
            if r is None or not np.isfinite(r) or not (0.0 <= r <= 1.0):
                problems.append(f"Reliability of channel '{key}' must lie in [0, 1].")
        for name, arr, shape in (
            ("Action costs", self.action_cost, (N_ACTIONS,)),
            ("Current-state costs", self.current_state_cost, (N_STATES, N_ACTIONS)),
            ("Post-transition state costs", self.post_state_cost, (N_STATES,)),
        ):
            a = np.asarray(arr, dtype=float)
            if a.shape != shape:
                problems.append(f"{name} must have shape {shape}.")
            elif np.any(~np.isfinite(a)) or np.any(a < 0):
                problems.append(f"{name} must be finite and non-negative.")
        if not (0.0 <= self.demand_volatility <= 0.5):
            problems.append("Demand volatility must lie in [0, 0.5].")
        if not (0.0 <= self.supplier_risk <= 1.0):
            problems.append("Supplier delivery risk must lie in [0, 1].")
        if not (0.0 <= self.discount < 1.0):
            problems.append("Discount factor must lie in [0, 1).")
        if int(self.horizon) < 1:
            problems.append("Simulation duration must be at least 1 period.")
        return problems


def default_config() -> SentinelConfig:
    return SentinelConfig()


def observation_index(channel: str, observation) -> int:
    """Resolve an observation given as index or label into an index (raises ValueError)."""
    if channel not in CHANNELS:
        raise ValueError(f"Unknown observation channel: {channel!r}")
    labels = CHANNELS[channel]["observations"]
    if isinstance(observation, (int, np.integer)):
        if 0 <= int(observation) < len(labels):
            return int(observation)
        raise ValueError(f"Observation index {observation} out of range for channel '{channel}'.")
    if observation in labels:
        return labels.index(observation)
    raise ValueError(f"Unknown observation {observation!r} for channel '{channel}'.")


def action_index(action) -> int:
    """Resolve an action given as index, key or label into an index (raises ValueError)."""
    if isinstance(action, (int, np.integer)):
        if 0 <= int(action) < N_ACTIONS:
            return int(action)
        raise ValueError(f"Action index {action} out of range.")
    if action in ACTION_KEYS:
        return ACTION_KEYS.index(action)
    if action in ACTIONS:
        return ACTIONS.index(action)
    if action in ACTION_SHORT:
        return ACTION_SHORT.index(action)
    raise ValueError(f"Unknown action: {action!r}")
