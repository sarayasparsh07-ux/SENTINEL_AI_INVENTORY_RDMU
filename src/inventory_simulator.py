"""Synthetic supply-chain environment with a HIDDEN inventory state.

The environment owns the ground truth. It exposes exactly one thing to an
agent: the observations returned by `step()`. Everything else (hidden state,
inventory-unit snapshot, demand, stock-outs) is stored in `truth_log` for
evaluation only.

IMPORTANT - what the "units" are. The hidden STATE is the only quantity with
dynamics (it moves according to T). The inventory units are NOT a physical
stock balance: there is no roll-forward equation (units_next = units - demand
+ deliveries). In each period a synthetic, state-conditioned inventory
SNAPSHOT is drawn independently and uniformly inside the current state's band,
and demand is drawn independently. Stock-out, fill rate and service level are
computed per period from that snapshot and demand. They are proxies that make
the three-state abstraction measurable in units; they are not the output of a
physically continuous inventory simulation.

Generative process for one period (see the timing convention in config.py):

    s_{t+1} ~ T_eff(. | s_t, a_t)                              hidden transition
    for every channel k:
        genuine_k ~ Z_k(. | s_{t+1})                           nominal reading
        o_k = genuine_k with prob. r_k, else uniform noise      contamination mixture
    synthetic unit proxies (never observed, drawn afresh each period):
        inventory snapshot ~ Uniform(band of s_{t+1})            no carry-over between periods
        demand units       ~ Normal(mean, sd of the GENUINE demand level), floored at 1
        fill rate          = min(snapshot, demand) / demand
        stock-out          = demand > snapshot

Model-mismatch options let the truth differ from the agent's assumptions
(different channel reliabilities, a system alert that merely echoes the
sensor, faster depletion, an unreliable supplier).

Random numbers are drawn from separate, fixed-length streams, so two policies
run with the same seed face the same underlying randomness (common random
numbers) even though their actions differ.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from .belief_updater import correct
from .config import (ACTIONS, CHANNEL_KEYS, CHANNELS, DEMAND_MEAN_UNITS, DEMAND_SD_UNITS,
                     INVENTORY_BANDS, N_STATES, SIM_START, STATES, SentinelConfig, action_index)
from .cost_model import CostModel
from .decision_policy import Policy, PolicyContext
from .observation_model import ObservationModel
from .transition_model import TransitionModel

# Sensor symbol -> system-alert symbol used when the system alert merely echoes the sensor.
_SENSOR_TO_SYSTEM_ECHO = {0: 0, 1: 2, 2: 3, 3: 4}


@dataclass
class EnvironmentOptions:
    """How the TRUE environment may deviate from the agent's assumed model."""
    true_reliability: Optional[Dict[str, float]] = None     # None -> same as the agent assumes
    system_echo_probability: float = 0.0                    # P(system alert is a copy of the sensor reading)
    true_demand_volatility: Optional[float] = None          # None -> same as the agent assumes
    true_supplier_risk: Optional[float] = None              # None -> same as the agent assumes
    initial_state: Optional[int] = None                     # None -> sampled from the initial belief
    label: str = "Well-specified (truth = agent's model)"


def _inverse_cdf(probabilities: np.ndarray, u: float) -> int:
    idx = int(np.searchsorted(np.cumsum(probabilities), u, side="right"))
    return min(idx, len(probabilities) - 1)


class InventoryEnvironment:
    """Hidden-state inventory simulator (ground truth lives here and only here)."""

    def __init__(self, config: SentinelConfig, seed: Optional[int] = None,
                 options: Optional[EnvironmentOptions] = None):
        self.options = options or EnvironmentOptions()
        self.seed = config.seed if seed is None else int(seed)
        self.configure(config)
        self.reset()

    # ------------------------------------------------------------------ set-up
    def configure(self, config: SentinelConfig) -> None:
        """(Re)build the true dynamics from a configuration without touching the hidden state."""
        self.config = config
        opt = self.options
        kappa = config.demand_volatility if opt.true_demand_volatility is None else opt.true_demand_volatility
        rho = config.supplier_risk if opt.true_supplier_risk is None else opt.true_supplier_risk
        self._transitions = TransitionModel(config.transitions, kappa, rho).tensor()
        reliability = dict(config.reliability)
        if opt.true_reliability:
            reliability.update(opt.true_reliability)
        self._obs_model = ObservationModel(config.likelihoods, reliability, config.epsilon)
        self._nominal = {k: self._obs_model.nominal[k] for k in CHANNEL_KEYS}
        self._reliability = {k: self._obs_model.reliability[k] for k in CHANNEL_KEYS}
        self._initial_belief = np.asarray(config.initial_belief, dtype=float)

    def reset(self) -> None:
        streams = np.random.SeedSequence(self.seed).spawn(4)
        self._rng_init, self._rng_trans, self._rng_obs, self._rng_phys = (
            np.random.default_rng(s) for s in streams)
        self.t = 0
        if self.options.initial_state is None:
            self._state = _inverse_cdf(self._initial_belief, self._rng_init.random())
        else:
            self._state = int(self.options.initial_state)
        self.truth_log: List[dict] = []
        self._latest_observations = self._realise_period(action=None)

    # --------------------------------------------------------------- internals
    def _realise_period(self, action: Optional[int]) -> Dict[str, int]:
        """Draw this period's observations and its hidden, state-conditioned unit proxies."""
        s = self._state
        u = self._rng_obs.random((len(CHANNEL_KEYS), 3))
        echo_u = self._rng_obs.random()
        observations: Dict[str, int] = {}
        genuine: Dict[str, int] = {}
        for i, key in enumerate(CHANNEL_KEYS):
            n_obs = self._nominal[key].shape[1]
            genuine[key] = _inverse_cdf(self._nominal[key][s], u[i, 0])
            if u[i, 1] < self._reliability[key]:
                observations[key] = genuine[key]
            else:
                observations[key] = min(int(u[i, 2] * n_obs), n_obs - 1)
        if echo_u < self.options.system_echo_probability:
            observations["system"] = _SENSOR_TO_SYSTEM_ECHO[observations["sensor"]]

        # synthetic state-conditioned snapshot (evaluation only): sampled independently in the
        # current state's band each period; it is NOT rolled forward from the previous period
        lo, hi = INVENTORY_BANDS[s]
        units = float(lo + (hi - lo) * self._rng_phys.random())
        level = genuine["demand"]
        demand = float(max(1.0, DEMAND_MEAN_UNITS[level] + DEMAND_SD_UNITS[level] * self._rng_phys.standard_normal()))
        served = min(units, demand)
        self.truth_log.append({
            "step": self.t, "hidden_state": s, "hidden_state_name": STATES[s],
            "action_into_period": None if action is None else int(action),
            # logged to 0.1 unit, rounded DOWN so the value stays inside the half-open band [lower, upper)
            "inventory_units": float(np.floor(units * 10.0) / 10.0), "demand_units": round(demand, 1),
            "units_served": round(served, 1), "units_short": round(demand - served, 1),
            "fill_rate": served / demand, "stockout": bool(demand > units),
        })
        return observations

    # ------------------------------------------------------------------ public
    def step(self, action) -> Dict[str, int]:
        """Execute an action; the hidden state transitions; return the new observations."""
        a = action_index(action)
        self._state = _inverse_cdf(self._transitions[a, self._state], self._rng_trans.random())
        self.t += 1
        self._latest_observations = self._realise_period(a)
        return dict(self._latest_observations)

    # Evaluation-only accessors -------------------------------------------------
    def evaluation_state(self) -> int:
        """TRUE hidden state. For evaluation / debug views only - never for decisions."""
        return int(self._state)

    def evaluation_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.truth_log)
        if not frame.empty:
            frame["date"] = pd.Timestamp(SIM_START).normalize() + pd.to_timedelta(frame["step"], unit="D")
        return frame


# ----------------------------------------------------------------------------
# Model-implied service profile (available to the agent: it is model knowledge)
# ----------------------------------------------------------------------------
def service_profile(config: SentinelConfig, grid: int = 400) -> Dict[str, np.ndarray]:
    """P(stock-out | state) and expected fill rate per state implied by the unit proxies.

    Computed by deterministic numerical integration over the uniform within-band
    inventory snapshot and the demand distribution (the same assumptions the
    simulator uses for its per-period snapshot). This is knowledge of the
    MODEL (thresholds and demand assumptions), not of the hidden state.
    """
    nominal = ObservationModel.from_config(config).nominal["demand"]
    q = (np.arange(200) + 0.5) / 200
    z = norm.ppf(q)
    p_stockout = np.zeros(N_STATES)
    fill = np.zeros(N_STATES)
    for s in range(N_STATES):
        lo, hi = INVENTORY_BANDS[s]
        units = lo + (hi - lo) * (np.arange(grid) + 0.5) / grid
        for level, p_level in enumerate(nominal[s]):
            demand = np.maximum(1.0, DEMAND_MEAN_UNITS[level] + DEMAND_SD_UNITS[level] * z)
            short = demand[None, :] > units[:, None]
            p_stockout[s] += p_level * short.mean()
            fill[s] += p_level * (np.minimum(units[:, None], demand[None, :]) / demand[None, :]).mean()
    return {"stockout_probability": p_stockout, "fill_rate": fill}


# ----------------------------------------------------------------------------
# Episode runner (one prediction + one joint correction per period)
# ----------------------------------------------------------------------------
@dataclass
class AgentModels:
    """The model the AGENT assumes (may differ from the environment's truth)."""
    transition: TransitionModel
    observation: ObservationModel
    cost: CostModel

    @classmethod
    def from_config(cls, config: SentinelConfig) -> "AgentModels":
        return cls(TransitionModel.from_config(config), ObservationModel.from_config(config),
                   CostModel.from_config(config))


def run_episode(config: SentinelConfig, policy: Policy, horizon: Optional[int] = None,
                seed: Optional[int] = None, options: Optional[EnvironmentOptions] = None,
                models: Optional[AgentModels] = None) -> pd.DataFrame:
    """Simulate one episode and return one row per period.

    Row t holds the hidden state s_t, the belief b_t held when the action was
    chosen, the executed action a_t, the realised cost C(s_t, a_t, s_{t+1}) and
    the hidden service figures of period t. The belief filter is run for every
    policy (with that policy's executed actions) so belief metrics are always
    available; only policies flagged `requires_oracle` are handed the true state.
    """
    horizon = int(config.horizon if horizon is None else horizon)
    models = models or AgentModels.from_config(config)
    env = InventoryEnvironment(config, seed=seed, options=options)
    transitions = models.transition.tensor()
    likelihood = {k: models.observation.effective_matrix(k) for k in CHANNEL_KEYS}

    policy.reset()
    belief = np.asarray(config.initial_belief, dtype=float).copy()
    observations: Dict[str, int] = {}
    rows: List[dict] = []
    for t in range(horizon):
        state = env.evaluation_state()                     # evaluation bookkeeping only
        ctx = PolicyContext(step=t, belief=belief.copy(), observations=dict(observations),
                            true_state=state if policy.requires_oracle else None)
        action = int(policy.act(ctx))
        truth = env.truth_log[-1]
        new_observations = env.step(action)
        next_state = env.evaluation_state()
        rows.append({
            "step": t, "hidden_state": state, "action": action, "next_state": next_state,
            "cost": models.cost.transition_cost(state, action, next_state),
            "b_healthy": belief[0], "b_low": belief[1], "b_critical": belief[2],
            **{f"obs_{k}": observations.get(k, -1) for k in CHANNEL_KEYS},
            "inventory_units": truth["inventory_units"], "demand_units": truth["demand_units"],
            "units_served": truth["units_served"], "fill_rate": truth["fill_rate"],
            "stockout": truth["stockout"],
        })
        # ONE prediction with the executed action, then ONE joint correction
        predicted = belief @ transitions[action]
        vectors = [likelihood[k][:, new_observations[k]] for k in CHANNEL_KEYS]
        belief = correct(predicted / predicted.sum(), vectors).posterior
        observations = new_observations
    frame = pd.DataFrame(rows)
    frame["hidden_state_name"] = frame["hidden_state"].map(lambda i: STATES[i])
    frame["action_name"] = frame["action"].map(lambda i: ACTIONS[i])
    return frame


def generate_event_dataset(config: SentinelConfig, policy: Policy, horizon: int = 120,
                           seed: Optional[int] = None) -> pd.DataFrame:
    """Readable synthetic event table (what the notebook saves as data/sample_events.csv).

    Columns starting with `eval_` are hidden ground truth kept for evaluation
    only; the remaining columns are what an operational agent could observe.
    """
    frame = run_episode(config, policy, horizon=horizon, seed=seed)
    out = pd.DataFrame({
        "period": frame["step"],
        "date": (pd.Timestamp(SIM_START).normalize() + pd.to_timedelta(frame["step"], unit="D")).dt.date,
    })
    for key in CHANNEL_KEYS:
        labels = CHANNELS[key]["observations"]
        out[f"{key}_observation"] = frame[f"obs_{key}"].map(
            lambda i, labels=labels: labels[i] if i >= 0 else "(none - initial period)")
        out[f"{key}_assumed_reliability"] = config.reliability[key]
    out["executed_action"] = frame["action_name"]
    out["belief_healthy"] = frame["b_healthy"].round(4)
    out["belief_low"] = frame["b_low"].round(4)
    out["belief_critical"] = frame["b_critical"].round(4)
    out["eval_hidden_state"] = frame["hidden_state_name"]
    out["eval_inventory_units"] = frame["inventory_units"]
    out["eval_demand_units"] = frame["demand_units"]
    out["eval_fill_rate"] = frame["fill_rate"].round(4)
    out["eval_stockout"] = frame["stockout"]
    out["eval_realised_cost_scu"] = frame["cost"]
    return out


PolicyFactory = Callable[[AgentModels], Policy]


def evaluate_policies(config: SentinelConfig, policy_factories: Dict[str, PolicyFactory],
                      n_replications: int = 200, horizon: Optional[int] = None,
                      seed: Optional[int] = None, options: Optional[EnvironmentOptions] = None,
                      keep_trajectories: bool = False):
    """Run every policy on the SAME seeds (common random numbers).

    Returns a long DataFrame with one row per (policy, replication). With
    `keep_trajectories=True` also returns {policy: concatenated episode frames}.
    """
    from .metrics import summarise_episode   # local import avoids a circular dependency
    base_seed = config.seed if seed is None else int(seed)
    models = AgentModels.from_config(config)
    summaries: List[dict] = []
    trajectories: Dict[str, List[pd.DataFrame]] = {name: [] for name in policy_factories}
    for name, factory in policy_factories.items():
        policy = factory(models)
        for rep in range(int(n_replications)):
            frame = run_episode(config, policy, horizon=horizon, seed=base_seed + rep,
                                options=options, models=models)
            summary = summarise_episode(frame)
            summary.update({"policy": name, "replication": rep})
            summaries.append(summary)
            if keep_trajectories:
                frame = frame.assign(replication=rep)
                trajectories[name].append(frame)
    result = pd.DataFrame(summaries)
    if keep_trajectories:
        return result, {k: pd.concat(v, ignore_index=True) for k, v in trajectories.items()}
    return result
