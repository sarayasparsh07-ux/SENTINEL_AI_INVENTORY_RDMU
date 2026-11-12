"""SentinelEngine - the single decision engine shared by the notebook and the dashboard.

The engine keeps an interactive session:

* the agent side: predicted belief for the current period, the evidence
  received in this period, the posterior belief and the recommendation;
* the environment side: a hidden-state simulator. The engine passes exactly
  one thing from the environment to the agent logic - the observations.

Key rules (each is unit-tested):

1. RECOMMENDED != EXECUTED. `inject_alert` only corrects the belief and
   recomputes the recommendation. A transition model is applied only inside
   `run_next_step`, i.e. when an action is actually executed.
2. One prediction per period, then one JOINT correction with every
   observation of that period. The posterior is always recomputed from the
   period's predicted belief and its full evidence set.
3. Duplicate protection: by default each channel contributes at most one
   observation per period (a repeated report is ignored, a revised report
   replaces the earlier one). Switching the guard off treats repeats as
   independent evidence and demonstrates double counting.
4. The hidden state is never read by `inject_alert`, `run_next_step`'s belief
   maths, `evaluate_actions` or the explanation generator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .belief_updater import (BeliefUpdate, belief_concentration, correct, correct_sequential,
                             total_variation, validate_belief)
from .config import (ACTIONS, CHANNEL_KEYS, CHANNELS, COST_UNIT, SIM_START, STATES,
                     SentinelConfig, action_index, default_config, observation_index)
from .cost_model import CostModel
from .decision_policy import Decision, evaluate_actions, explain_decision
from .inventory_simulator import EnvironmentOptions, InventoryEnvironment, service_profile
from .metrics import brier_score, log_loss, map_accuracy
from .observation_model import ObservationModel
from .transition_model import TransitionModel


def format_belief(belief: Sequence[float]) -> str:
    return " / ".join(f"{100 * float(p):.1f}%" for p in belief)


@dataclass
class Evidence:
    """One counted observation of the current period."""
    event_id: int
    channel: str
    obs_index: int
    reliability: float
    uses_default_reliability: bool
    origin: str                       # "manual" | "environment" | "scenario"
    likelihood: np.ndarray            # [Z_r(o|Healthy), Z_r(o|Low), Z_r(o|Critical)]

    @property
    def channel_label(self) -> str:
        return CHANNELS[self.channel]["label"]

    @property
    def observation(self) -> str:
        return CHANNELS[self.channel]["observations"][self.obs_index]

    @property
    def severity(self) -> str:
        return CHANNELS[self.channel]["severity"][self.obs_index]


class SentinelEngine:
    """Interactive POMDP belief-state session (agent + hidden simulator)."""

    def __init__(self, config: Optional[SentinelConfig] = None, seed: Optional[int] = None,
                 environment_options: Optional[EnvironmentOptions] = None, deduplicate: bool = True):
        self.config = (config or default_config()).copy()
        problems = self.config.validate()
        if problems:
            raise ValueError("Invalid configuration: " + " ".join(problems))
        self.seed = self.config.seed if seed is None else int(seed)
        self.deduplicate = bool(deduplicate)
        self._environment_options = environment_options
        self._build_models()
        self.env = InventoryEnvironment(self.config, seed=self.seed, options=environment_options)
        self._start_session()

    # ------------------------------------------------------------------ set-up
    def _build_models(self) -> None:
        self.transition_model = TransitionModel.from_config(self.config)
        self.observation_model = ObservationModel.from_config(self.config)
        self.cost_model = CostModel.from_config(self.config)
        profile = service_profile(self.config)
        self.stockout_probability = profile["stockout_probability"]
        self.fill_rate = profile["fill_rate"]

    def _start_session(self) -> None:
        self.step = 0
        self._event_counter = 0
        self._events_in_period = 0
        b0 = validate_belief(self.config.initial_belief, "initial belief").copy()
        self.prior = b0.copy()            # posterior at the end of the previous period
        self.predicted = b0.copy()        # predicted belief of the current period (b0 at period 0)
        self.belief = b0.copy()           # posterior of the current period
        self.evidence: List[Evidence] = []
        self.last_executed_action: Optional[int] = None
        self.last_correction = correct(self.predicted, [])
        self.decision: Decision = evaluate_actions(self.belief, self.transition_model, self.cost_model)
        self.timeline: List[dict] = []
        self.event_log: List[dict] = []
        self.action_log: List[dict] = []
        self._evaluation_records: List[dict] = []
        self.alerts_processed = 0
        self.policy_changes = 0
        self.notices: List[str] = []
        self._snapshot("init", "Session initialised with the prior belief")

    def reset(self, seed: Optional[int] = None) -> None:
        """Restore the initial configuration: prior belief, empty logs, fresh hidden simulator."""
        if seed is not None:
            self.seed = int(seed)
        self._build_models()
        self.env = InventoryEnvironment(self.config, seed=self.seed, options=self._environment_options)
        self._start_session()

    def apply_config(self, config: SentinelConfig) -> None:
        """Change model parameters mid-session and recompute everything that depends on them.

        The current period's posterior is rebuilt from its predicted belief and
        evidence with the new likelihoods; the recommendation is re-evaluated
        with the new transitions and costs. History rows are kept as recorded.
        A changed initial belief takes effect immediately at period 0, otherwise
        at the next reset.
        """
        problems = config.validate()
        if problems:
            raise ValueError(" ".join(problems))
        self.config = config.copy()
        self._build_models()
        self.env.configure(self.config)
        if self.step == 0:
            b0 = validate_belief(self.config.initial_belief, "initial belief").copy()
            self.prior, self.predicted = b0.copy(), b0.copy()
        for item in self.evidence:
            reliability = None if item.uses_default_reliability else item.reliability
            item.likelihood = self.observation_model.likelihood_vector(item.channel, item.obs_index, reliability)
            if item.uses_default_reliability:
                item.reliability = self.observation_model.reliability[item.channel]
        self._recompute()
        self._snapshot("config", "Model configuration changed; outputs recomputed")

    # ---------------------------------------------------------------- clock
    def timestamp(self) -> pd.Timestamp:
        return (pd.Timestamp(SIM_START) + pd.Timedelta(days=self.step)
                + pd.Timedelta(minutes=20 * self._events_in_period))

    # ------------------------------------------------------------- core maths
    def _recompute(self) -> None:
        """Posterior = joint correction of the predicted belief with ALL current evidence."""
        self.last_correction = correct(self.predicted, [e.likelihood for e in self.evidence])
        self.belief = self.last_correction.posterior
        self.decision = evaluate_actions(self.belief, self.transition_model, self.cost_model)

    def _snapshot(self, kind: str, label: str) -> None:
        previous = self.timeline[-1]["recommended"] if self.timeline else None
        if previous is not None and previous != self.decision.action:
            self.policy_changes += 1
        row = {
            "event_index": len(self.timeline), "period": self.step, "timestamp": self.timestamp(),
            "kind": kind, "label": label,
            "b_healthy": float(self.belief[0]), "b_low": float(self.belief[1]),
            "b_critical": float(self.belief[2]),
            "recommended": int(self.decision.action), "recommended_name": self.decision.action_name,
            "expected_cost": self.decision.expected_cost,
            "concentration": belief_concentration(self.belief),
            "stockout_risk": float(self.belief @ self.stockout_probability),
            "expected_service_level": float(self.belief @ self.fill_rate),
            "executed_into_period": self.last_executed_action,
        }
        for i, q in enumerate(self.decision.q_values):
            row[f"q_{i}"] = float(q)
        self.timeline.append(row)
        self._events_in_period += 1

    # ------------------------------------------------------------- alerts
    def inject_alert(self, channel: str, observation, reliability: Optional[float] = None,
                     origin: str = "manual", deduplicate: Optional[bool] = None,
                     record_snapshot: bool = True, note: str = "") -> dict:
        """Add one observation to the current period and re-run the joint correction.

        No transition model is applied: an alert is evidence, not an executed action.
        Returns a dict describing what happened (status, beliefs, recommendations).
        """
        if channel not in CHANNEL_KEYS:
            raise ValueError(f"Unknown observation channel: {channel!r}. Valid: {CHANNEL_KEYS}.")
        obs_idx = observation_index(channel, observation)
        uses_default = reliability is None
        r = self.observation_model.reliability[channel] if uses_default else float(reliability)
        if not np.isfinite(r) or not (0.0 <= r <= 1.0):
            raise ValueError("Observation reliability must lie between 0 and 1.")
        dedupe = self.deduplicate if deduplicate is None else bool(deduplicate)

        before = self.belief.copy()
        previous_action = int(self.decision.action)
        same_channel = [e for e in self.evidence if e.channel == channel]
        operator_note = note
        status, note = "applied", ""

        if same_channel and dedupe:
            last = same_channel[-1]
            if last.obs_index == obs_idx and abs(last.reliability - r) < 1e-12:
                status = "duplicate ignored"
                note = (f"{CHANNELS[channel]['label']} already reported this observation in period "
                        f"{self.step}; a repeat adds no independent information.")
            else:
                self.evidence = [e for e in self.evidence if e.channel != channel]
                status = "applied (replaced earlier reading)"
                note = (f"Revised {CHANNELS[channel]['label'].lower()} report replaces the earlier one "
                        f"for period {self.step}.")
        elif same_channel and not dedupe:
            status = "applied as independent (guard off)"
            note = ("Duplicate guard is off: this repeat is counted as if it were independent evidence, "
                    "which can make the belief overconfident.")

        self._event_counter += 1
        event_id = self._event_counter
        if status != "duplicate ignored":
            vector = self.observation_model.likelihood_vector(channel, obs_idx, r)
            candidate = Evidence(event_id, channel, obs_idx, r, uses_default, origin, vector)
            trial = correct(self.predicted, [e.likelihood for e in self.evidence] + [vector])
            if trial.impossible:
                status = "rejected (impossible under the model)"
                note = "The model assigns zero probability to this evidence; the belief was left unchanged."
            else:
                self.evidence.append(candidate)
                self.alerts_processed += 1
            self._recompute()

        row = self._log_event(event_id, channel, obs_idx, r, origin, status, before, self.belief,
                              previous_action, int(self.decision.action),
                              operator_note=str(operator_note or "").strip()[:200])
        if note:
            self.notices.append(note)
        if record_snapshot and status != "duplicate ignored":
            self._snapshot("alert", f"{CHANNELS[channel]['label']}: {CHANNELS[channel]['observations'][obs_idx]}")
        return {"status": status, "note": note, "belief_before": before, "belief_after": self.belief.copy(),
                "previous_action": previous_action, "new_action": int(self.decision.action),
                "policy_changed": previous_action != int(self.decision.action), "event": row}

    def _log_event(self, event_id, channel, obs_idx, reliability, origin, status, before, after,
                   previous_action, new_action, operator_note: str = "") -> dict:
        row = {
            "Event": event_id, "Timestamp": self.timestamp().strftime("%Y-%m-%d %H:%M"), "Period": self.step,
            "Source": CHANNELS[channel]["label"],
            "Observation": CHANNELS[channel]["observations"][obs_idx],
            "Severity": CHANNELS[channel]["severity"][obs_idx],
            "Reliability": f"{100 * reliability:.0f}%", "Origin": origin, "Status": status,
            "Belief before (H / L / C)": format_belief(before),
            "Belief after (H / L / C)": format_belief(after),
            "Δ P(Critical) (pts)": round(100 * float(after[2] - before[2]), 2),
            "Previous recommendation": ACTIONS[previous_action],
            "New recommendation": ACTIONS[new_action],
            "Recommendation changed": "yes" if previous_action != new_action else "no",
            "Operator note": operator_note,
        }
        self.event_log.append(row)
        return row

    # ---------------------------------------------------------- time advance
    def run_next_step(self, action=None, observe: bool = True) -> dict:
        """Execute an action and advance the simulation by one period.

        `action=None` executes the current recommendation; otherwise the given
        action is executed (manual override). The executed action's transition
        model drives ONE prediction. With `observe=True` the simulated
        environment then emits one observation per channel, applied as a single
        joint correction.
        """
        recommended = int(self.decision.action)
        executed = recommended if action is None else action_index(action)
        belief_at_decision = self.belief.copy()
        decision_at_exec = self.decision

        # evaluation bookkeeping (hidden; never used below for any decision)
        state_before = self.env.evaluation_state()
        observations = self.env.step(executed)
        state_after = self.env.evaluation_state()
        self._evaluation_records.append({
            "period": self.step, "hidden_state": state_before, "next_hidden_state": state_after,
            "executed": executed,
            "realised_cost": self.cost_model.transition_cost(state_before, executed, state_after),
            "b_healthy": belief_at_decision[0], "b_low": belief_at_decision[1],
            "b_critical": belief_at_decision[2],
        })

        # ---- agent side: ONE prediction with the EXECUTED action ----
        predicted = self.transition_model.predict(belief_at_decision, executed)
        predicted = predicted / predicted.sum()
        action_row = {
            "Period": self.step, "Timestamp": self.timestamp().strftime("%Y-%m-%d %H:%M"),
            "Belief at decision (H / L / C)": format_belief(belief_at_decision),
            "Recommended action": ACTIONS[recommended], "Executed action": ACTIONS[executed],
            "Followed recommendation": "yes" if executed == recommended else "no (manual override)",
            f"Expected cost of executed action ({COST_UNIT})": round(float(decision_at_exec.q_values[executed]), 2),
            f"Expected cost of recommendation ({COST_UNIT})": round(decision_at_exec.expected_cost, 2),
            "Predicted outcome (H / L / C)": format_belief(predicted),
            "Predicted P(Critical)": f"{100 * predicted[2]:.1f}%",
            "Observed subsequent evidence": "(none yet)",
            "Posterior after evidence (H / L / C)": "(pending)",
            "_executed": executed, "_recommended": recommended,
            "_expected_cost_executed": float(decision_at_exec.q_values[executed]),
        }
        self.action_log.append(action_row)

        self.step += 1
        self._events_in_period = 0
        self.prior = belief_at_decision
        self.predicted = predicted
        self.evidence = []
        self.last_executed_action = executed
        self._recompute()
        self._snapshot("predict", f"Executed {ACTIONS[executed]}; belief predicted for period {self.step}")

        if observe:
            self._apply_environment_observations(observations)
        return {"executed": executed, "recommended": recommended, "predicted": predicted.copy(),
                "posterior": self.belief.copy(), "new_recommendation": int(self.decision.action),
                "observations": {k: CHANNELS[k]["observations"][v] for k, v in observations.items()}
                if observe else {}}

    def _apply_environment_observations(self, observations: Dict[str, int]) -> None:
        """Joint correction with the simulator's four channel readings (one batch)."""
        vectors, items = [], []
        for channel in CHANNEL_KEYS:
            idx = observations[channel]
            self._event_counter += 1
            vec = self.observation_model.likelihood_vector(channel, idx)
            items.append(Evidence(self._event_counter, channel, idx,
                                  self.observation_model.reliability[channel], True, "environment", vec))
            vectors.append(vec)
        intermediate = correct_sequential(self.predicted, vectors)   # for per-row display only
        before = self.predicted.copy()
        previous_action = int(self.decision.action)
        for item, after in zip(items, intermediate):
            new_action = int(evaluate_actions(after, self.transition_model, self.cost_model).action)
            self._log_event(item.event_id, item.channel, item.obs_index, item.reliability, "environment",
                            "applied", before, after, previous_action, new_action)
            before, previous_action = after, new_action
        self.evidence = items
        self.alerts_processed += len(items)
        self._recompute()                                            # the actual JOINT correction
        self._snapshot("correct", f"Period {self.step}: joint correction with {len(items)} observations")
        self._refresh_action_log()

    def _refresh_action_log(self) -> None:
        if self.action_log and self.evidence:
            self.action_log[-1]["Observed subsequent evidence"] = "; ".join(
                f"{e.channel_label}: {e.observation}" for e in self.evidence)
            self.action_log[-1]["Posterior after evidence (H / L / C)"] = format_belief(self.belief)

    # --------------------------------------------------------- scenario play
    def play_scenario(self, scenario, reset: bool = True) -> None:
        """Replay a scripted scenario (deterministic: scripted alerts, no sampled observations)."""
        if reset:
            self.reset()
        for stage in scenario.stages:
            if stage.advance:
                self.run_next_step(action=stage.execute, observe=False)
            for alert in stage.alerts:
                self.inject_alert(alert[0], alert[1], alert[2] if len(alert) > 2 else None, origin="scenario")
            self._refresh_action_log()

    # ------------------------------------------------------------- read-outs
    def evidence_summary(self) -> List[dict]:
        """Current period's evidence with each item's leave-one-out impact on the belief."""
        out = []
        for i, item in enumerate(self.evidence):
            others = [e.likelihood for j, e in enumerate(self.evidence) if j != i]
            without = correct(self.predicted, others).posterior
            out.append({"channel": item.channel, "channel_label": item.channel_label,
                        "observation": item.observation, "severity": item.severity,
                        "reliability": item.reliability, "origin": item.origin,
                        "likelihood": item.likelihood.copy(),
                        "impact": total_variation(self.belief, without)})
        return out

    def explanation(self) -> Dict[str, str]:
        previous = None
        if len(self.timeline) >= 2:
            previous = int(self.timeline[-2]["recommended"])
        return explain_decision(self.decision, self.predicted, self.evidence_summary(), previous)

    def bayes_breakdown(self) -> BeliefUpdate:
        """Prior, predicted, joint likelihood and posterior of the current period."""
        return BeliefUpdate(self.prior.copy(), self.predicted.copy(),
                            self.last_correction.joint_likelihood.copy(), self.belief.copy(),
                            self.last_correction.evidence_probability, self.last_correction.impossible)

    def kpis(self) -> dict:
        return {
            "p_healthy": float(self.belief[0]), "p_low": float(self.belief[1]),
            "p_critical": float(self.belief[2]),
            "recommended_action": self.decision.action_name, "recommended_index": int(self.decision.action),
            "expected_cost": self.decision.expected_cost,
            "alerts_processed": int(self.alerts_processed), "policy_changes": int(self.policy_changes),
            "concentration": belief_concentration(self.belief),
            "stockout_risk": float(self.belief @ self.stockout_probability),
            "expected_service_level": float(self.belief @ self.fill_rate),
            "period": int(self.step), "timestamp": self.timestamp(),
        }

    def timeline_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.timeline)

    def event_log_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.event_log)

    def action_log_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.action_log)
        return frame[[c for c in frame.columns if not c.startswith("_")]] if not frame.empty else frame

    def action_history(self) -> pd.DataFrame:
        """Numeric action history (period, recommended, executed, expected cost)."""
        return pd.DataFrame([{"period": r["Period"], "recommended": r["_recommended"],
                              "executed": r["_executed"],
                              "expected_cost": r["_expected_cost_executed"]} for r in self.action_log])

    # ---------------------------------------------- evaluation-only (hidden truth)
    def evaluation_frame(self) -> pd.DataFrame:
        """Hidden ground truth joined with the agent's beliefs. EVALUATION / DEBUG ONLY."""
        records = pd.DataFrame(self._evaluation_records)
        truth = self.env.evaluation_frame()
        if records.empty:
            return truth
        merged = truth.merge(records.drop(columns=["hidden_state"]), left_on="step", right_on="period", how="left")
        return merged

    def evaluation_metrics(self) -> Optional[dict]:
        """Belief accuracy against the simulated truth for the periods completed so far."""
        records = pd.DataFrame(self._evaluation_records)
        if records.empty:
            return None
        beliefs = records[["b_healthy", "b_low", "b_critical"]].to_numpy()
        truth_states = records["hidden_state"].to_numpy()
        truth = self.env.evaluation_frame().iloc[: len(records)]
        return {
            "periods": int(len(records)), "brier": brier_score(beliefs, truth_states),
            "log_loss": log_loss(beliefs, truth_states), "accuracy": map_accuracy(beliefs, truth_states),
            "realised_cost": float(records["realised_cost"].sum()),
            "stockout_rate": float(truth["stockout"].mean()),
            "service_level": float(truth["units_served"].sum() / truth["demand_units"].sum()),
        }
