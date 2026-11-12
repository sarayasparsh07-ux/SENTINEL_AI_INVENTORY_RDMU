"""Transparent one-step expected-cost decision policy, baselines and explanations.

Primary policy (myopic, gamma effectively 0):

    Q(b, a) = c_action(a) + sum_s b(s) c_current(s, a) + sum_s' b^-_a(s') c_post(s')
    a*      = argmin_a Q(b, a)                R(b, a) = -Q(b, a)

This is the exact optimum for a ONE-period problem under the stated model.
It is NOT an optimal infinite-horizon POMDP policy: it ignores costs beyond
the next period and places no value on gathering information.

The recommendation is a pure function of (belief, model). Nothing in this
module can read the simulator's hidden state except `OraclePolicy`, which is
an evaluation-only benchmark and is labelled as such.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import (ACTION_KEYS, ACTION_SHORT, ACTIONS, CHANNELS, COST_UNIT,
                     N_ACTIONS, N_STATES, STATES, action_index)
from .cost_model import CostModel
from .transition_model import TransitionModel


# ----------------------------------------------------------------------------
# Decision evaluation
# ----------------------------------------------------------------------------
@dataclass
class Decision:
    """Complete, inspectable result of evaluating every action for one belief."""
    belief: np.ndarray
    q_values: np.ndarray                 # expected total cost per action
    direct: np.ndarray                   # c_action(a)
    current_state: np.ndarray            # sum_s b(s) c_current(s, a)
    post_transition: np.ndarray          # sum_s' b^-_a(s') c_post(s')
    predicted_beliefs: np.ndarray        # b^-_a for each action, shape (actions, states)
    action: int                          # argmin
    runner_up: int
    margin: float                        # Q(runner-up) - Q(best)
    ranking: List[int] = field(default_factory=list)

    @property
    def action_name(self) -> str:
        return ACTIONS[self.action]

    @property
    def expected_cost(self) -> float:
        return float(self.q_values[self.action])

    @property
    def expected_reward(self) -> float:
        return -self.expected_cost

    @property
    def margin_pct(self) -> float:
        best = self.q_values[self.action]
        return float(100.0 * self.margin / best) if best > 0 else float("inf")

    def as_frame(self):
        import pandas as pd
        frame = pd.DataFrame({
            "Action": ACTIONS,
            f"Direct action cost ({COST_UNIT})": self.direct,
            f"Expected current-state cost ({COST_UNIT})": self.current_state,
            f"Expected post-transition cost ({COST_UNIT})": self.post_transition,
            f"Expected total cost Q(b,a) ({COST_UNIT})": self.q_values,
            "Predicted P(Critical) next period": self.predicted_beliefs[:, 2],
        })
        frame["Rank"] = frame[f"Expected total cost Q(b,a) ({COST_UNIT})"].rank(method="first").astype(int)
        frame["Recommended"] = ["yes" if i == self.action else "" for i in range(N_ACTIONS)]
        return frame


def evaluate_actions(belief, transition_model: TransitionModel, cost_model: CostModel) -> Decision:
    """Evaluate ALL actions and return the lowest-expected-cost one.

    Ties are broken in favour of the lower action index (the less
    interventionist action), which makes the policy deterministic.
    """
    b = np.asarray(belief, dtype=float)
    q = np.zeros(N_ACTIONS)
    direct = np.zeros(N_ACTIONS)
    current = np.zeros(N_ACTIONS)
    post = np.zeros(N_ACTIONS)
    predicted = np.zeros((N_ACTIONS, N_STATES))
    for a in range(N_ACTIONS):
        parts = cost_model.expected_cost_components(b, a, transition_model.matrix(a))
        direct[a], current[a], post[a] = parts["direct"], parts["current_state"], parts["post_transition"]
        q[a] = parts["total"]
        predicted[a] = parts["predicted_belief"]
    ranking = [int(i) for i in np.argsort(q, kind="stable")]
    best, second = ranking[0], ranking[1]
    return Decision(b.copy(), q, direct, current, post, predicted, best, second,
                    float(q[second] - q[best]), ranking)


def select_action(belief, transition_model: TransitionModel, cost_model: CostModel) -> int:
    return evaluate_actions(belief, transition_model, cost_model).action


# ----------------------------------------------------------------------------
# Dynamic explanations
# ----------------------------------------------------------------------------
def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _evidence_phrase(evidence: Sequence[dict], limit: int = 3) -> str:
    ranked = sorted(evidence, key=lambda e: e.get("impact", 0.0), reverse=True)[:limit]
    parts = [f"{e['channel_label']}: “{e['observation']}” ({100 * e['reliability']:.0f}% reliable)"
             for e in ranked]
    if len(evidence) > limit:
        parts.append(f"{len(evidence) - limit} further observation(s)")
    return "; ".join(parts)


def explain_decision(decision: Decision, reference_belief: Optional[np.ndarray] = None,
                     evidence: Optional[Sequence[dict]] = None,
                     previous_action: Optional[int] = None) -> Dict[str, str]:
    """Build a human-readable explanation from the live numbers.

    `reference_belief` is the belief before the current period's evidence (the
    predicted belief), `evidence` the list of observations counted so far in
    this period, each a dict with channel_label, observation, reliability, impact.
    Returns a dict with a headline, a paragraph, a NOVA one-liner and the main evidence.
    """
    b = decision.belief
    a, r = decision.action, decision.runner_up
    evidence = list(evidence or [])
    sentences: List[str] = []

    # 1. what changed in the belief and which evidence did it
    if reference_belief is not None and evidence:
        delta = b - np.asarray(reference_belief, dtype=float)
        mover = int(np.argmax(np.abs(delta)))
        direction = "rose" if delta[mover] > 0 else "fell"
        if abs(delta[mover]) < 0.005:
            sentences.append(
                f"The latest evidence ({_evidence_phrase(evidence)}) left the belief almost unchanged, "
                f"because its likelihood is nearly the same in every inventory state.")
        else:
            sentences.append(
                f"P({STATES[mover]}) {direction} from {_pct(reference_belief[mover])} to {_pct(b[mover])} "
                f"({100 * delta[mover]:+.1f} pts) after {_evidence_phrase(evidence)}.")
    elif not evidence:
        sentences.append(
            f"No new evidence has been received this period, so the belief is the model's prediction: "
            f"Healthy {_pct(b[0])}, Low {_pct(b[1])}, Critical {_pct(b[2])}.")

    # 2. cost comparison
    sentences.append(
        f"{ACTIONS[a]} has the lowest expected total cost, {decision.q_values[a]:.1f} {COST_UNIT}, "
        f"against {decision.q_values[r]:.1f} {COST_UNIT} for {ACTIONS[r]} "
        f"(margin {decision.margin:.1f} {COST_UNIT}, {decision.margin_pct:.1f}%).")

    # 3. why: trade-off between direct cost and shortage exposure
    p_crit_best = decision.predicted_beliefs[a, 2]
    p_crit_monitor = decision.predicted_beliefs[0, 2]
    if a == 0:
        cheapest_other = min(range(1, N_ACTIONS), key=lambda i: decision.q_values[i])
        saved = decision.post_transition[0] - decision.post_transition[cheapest_other]
        paid = (decision.direct[cheapest_other] + decision.current_state[cheapest_other])
        sentences.append(
            f"Intervening with {ACTIONS[cheapest_other]} would cut the expected holding-and-shortage cost by "
            f"only {saved:.1f} {COST_UNIT} while adding {paid:.1f} {COST_UNIT} of ordering and surplus cost, "
            f"so waiting for more evidence is cheaper under the current belief.")
    else:
        extra_direct = decision.direct[a] + decision.current_state[a]
        saved = decision.post_transition[0] - decision.post_transition[a]
        sentences.append(
            f"It costs {extra_direct:.1f} {COST_UNIT} to execute, but lowers the predicted probability of "
            f"Critical inventory next period from {_pct(p_crit_monitor)} (if we only monitor) to "
            f"{_pct(p_crit_best)}, saving an expected {saved:.1f} {COST_UNIT} of holding-and-shortage cost.")
        if decision.direct[a] > decision.direct[r] and r != 0:
            sentences.append(
                f"Although {ACTIONS[a]} has a higher direct cost than {ACTIONS[r]}, the stronger recovery "
                f"more than pays for it at this level of Critical risk.")

    # 4. policy switch
    if previous_action is not None and previous_action != a:
        sentences.append(f"This replaces the previous recommendation, {ACTIONS[previous_action]}.")

    # NOVA one-liner
    if evidence:
        top = max(evidence, key=lambda e: e.get("impact", 0.0))
        lead = f"{top['channel_label']} report received: {top['observation'].lower()}. Beliefs updated."
        main = f"{top['channel_label']}: {top['observation']}"
    else:
        lead = "No new evidence this period. Holding the predicted belief."
        main = "None this period (prediction only)"
    if previous_action is not None and previous_action != a:
        tail = f" Recommendation changed to {ACTIONS[a]}."
    else:
        tail = f" {ACTIONS[a]} currently has the lowest expected cost."
    return {
        "headline": f"NOVA RECOMMENDATION: {ACTIONS[a].upper()}",
        "paragraph": " ".join(sentences),
        "nova": lead + tail,
        "main_evidence": main,
    }


# ----------------------------------------------------------------------------
# Policies (the belief policy plus evaluation baselines)
# ----------------------------------------------------------------------------
@dataclass
class PolicyContext:
    """Everything a policy is allowed to see when it chooses an action."""
    step: int
    belief: np.ndarray                      # the Bayes-filter belief b_t
    observations: Dict[str, int]            # most recent observation per channel ({} at t = 0)
    true_state: Optional[int] = None        # filled ONLY for policies with requires_oracle = True


class Policy:
    name = "policy"
    requires_oracle = False
    description = ""

    def reset(self) -> None:  # stateful policies override
        pass

    def act(self, ctx: PolicyContext) -> int:
        raise NotImplementedError


class BeliefMyopicPolicy(Policy):
    """SENTINEL's policy: argmin of the one-step expected cost under the belief."""
    name = "SENTINEL belief agent (myopic)"
    description = "Bayes-filter belief + one-step expected-cost minimisation."

    def __init__(self, transition_model: TransitionModel, cost_model: CostModel):
        self.transition_model, self.cost_model = transition_model, cost_model
        # Q(b, a) = (b @ M)[a]; identical to evaluate_actions (unit-tested), just faster.
        self._one_step = cost_model.one_step_cost_matrix(transition_model.tensor())

    def act(self, ctx: PolicyContext) -> int:
        return int(np.argmin(np.asarray(ctx.belief, dtype=float) @ self._one_step))


class FixedPolicy(Policy):
    """Always executes the same action, regardless of evidence."""

    def __init__(self, action):
        self.action = action_index(action)
        self.name = f"Fixed: always {ACTION_SHORT[self.action].lower()}"
        self.description = f"Executes {ACTIONS[self.action]} every period."

    def act(self, ctx: PolicyContext) -> int:
        return self.action


class PeriodicReorderPolicy(Policy):
    """Calendar rule: Standard Reorder every `every` periods, otherwise monitor."""

    def __init__(self, every: int = 3):
        self.every = int(every)
        self.name = f"Fixed: reorder every {self.every} periods"
        self.description = "Periodic-review rule that ignores all evidence."

    def act(self, ctx: PolicyContext) -> int:
        return 1 if ctx.step % self.every == 0 else 0


class RecordTrustingPolicy(Policy):
    """Treats the latest sensor reading as if it were the true state.

    Normal -> monitor, Low reading -> reorder, Critical warning -> expedite,
    malfunction / no reading -> monitor. No probabilities, no memory.
    """
    name = "Record-trusting heuristic"
    description = "Acts on the most recent sensor reading as if it were the truth."
    _MAP = {0: 0, 1: 1, 2: 2, 3: 0}

    def act(self, ctx: PolicyContext) -> int:
        reading = ctx.observations.get("sensor")
        return self._MAP.get(reading, 0)


class InfographicThresholdPolicy(Policy):
    """The fixed belief thresholds sketched on the lecture infographic.

    P(Critical) > 20% -> emergency replenishment (Expedite);
    else P(Low) > 40% -> Standard Reorder; else Continue Monitoring.
    It uses the same belief as SENTINEL but hard-coded cut-offs instead of costs.
    """
    name = "Infographic threshold rule"
    description = "Belief thresholds 20% / 40% from the lecture infographic (no cost model)."

    def act(self, ctx: PolicyContext) -> int:
        if ctx.belief[2] > 0.20:
            return 2
        if ctx.belief[1] > 0.40:
            return 1
        return 0


class OraclePolicy(Policy):
    """Full-information myopic benchmark: sees the TRUE state (simulation only).

    It applies the same one-step cost rule to a one-hot belief. It is not
    implementable under partial observability and it is not 'the optimal
    policy'; it only shows what perfect visibility would be worth to a myopic
    decision-maker.
    """
    name = "Full-information myopic oracle"
    description = "Evaluation-only benchmark with access to the hidden state."
    requires_oracle = True

    def __init__(self, transition_model: TransitionModel, cost_model: CostModel):
        self.transition_model, self.cost_model = transition_model, cost_model

    def act(self, ctx: PolicyContext) -> int:
        if ctx.true_state is None:
            raise RuntimeError("OraclePolicy needs the true state; the runner did not provide it.")
        one_hot = np.zeros(N_STATES)
        one_hot[int(ctx.true_state)] = 1.0
        return select_action(one_hot, self.transition_model, self.cost_model)


def solve_underlying_mdp(transition_model: TransitionModel, cost_model: CostModel,
                         discount: float, tol: float = 1e-10, max_iter: int = 10_000) -> np.ndarray:
    """Value iteration on the fully observable MDP; returns Q_MDP[s, a] (costs)."""
    t = transition_model.tensor()                 # (a, s, s')
    c = cost_model.cost_tensor()                  # (a, s, s')
    v = np.zeros(N_STATES)
    for _ in range(max_iter):
        q = np.einsum("ast,ast->sa", t, c + discount * v[None, None, :])
        v_new = q.min(axis=1)
        if np.max(np.abs(v_new - v)) < tol:
            v = v_new
            break
        v = v_new
    return np.einsum("ast,ast->sa", t, c + discount * v[None, None, :])


class QMDPPolicy(Policy):
    """Optional comparison: QMDP approximation (Littman, Cassandra & Kaelbling, 1995).

    Solves the underlying fully observable MDP by value iteration with discount
    gamma, then acts on  argmin_a sum_s b(s) Q_MDP(s, a). It looks beyond one
    period but assumes all uncertainty disappears after the next step, so, like
    the myopic rule, it never values information gathering.
    """

    def __init__(self, transition_model: TransitionModel, cost_model: CostModel, discount: float):
        self.q_mdp = solve_underlying_mdp(transition_model, cost_model, discount)
        self.name = f"QMDP approximation (γ = {discount:.2f})"
        self.description = "Infinite-horizon MDP values weighted by the belief (optional comparison)."

    def act(self, ctx: PolicyContext) -> int:
        return int(np.argmin(np.asarray(ctx.belief) @ self.q_mdp))


def policy_region_grid(transition_model: TransitionModel, cost_model: CostModel, resolution: int = 40):
    """Evaluate the myopic policy on a grid over the belief simplex.

    Returns a DataFrame with columns Healthy, Low, Critical, action, action_name,
    expected_cost - used to draw the policy map (which action wins where).
    Uses the linear form Q(b, .) = b @ M, which is identical to `evaluate_actions`.
    """
    import pandas as pd
    points = np.array([[i, j, resolution - i - j] for i in range(resolution + 1)
                       for j in range(resolution + 1 - i)], dtype=float) / resolution
    q = points @ cost_model.one_step_cost_matrix(transition_model.tensor())
    best = q.argmin(axis=1)
    return pd.DataFrame({"Healthy": points[:, 0], "Low": points[:, 1], "Critical": points[:, 2],
                         "action": best, "action_name": [ACTIONS[a] for a in best],
                         "expected_cost": q.min(axis=1)})


def channel_label(channel: str) -> str:
    return CHANNELS[channel]["label"]


__all__ = [
    "Decision", "evaluate_actions", "select_action", "explain_decision", "PolicyContext", "Policy",
    "BeliefMyopicPolicy", "FixedPolicy", "PeriodicReorderPolicy", "RecordTrustingPolicy",
    "InfographicThresholdPolicy", "OraclePolicy", "QMDPPolicy", "solve_underlying_mdp",
    "policy_region_grid", "channel_label", "ACTION_KEYS",
]
