"""Decision policy validation: expected costs, action selection, recommended vs executed, no leakage."""
import inspect

import numpy as np
import pytest

from src.belief_updater import correct
from src.config import ACTION_KEYS, ACTIONS, default_config
from src.cost_model import CostModel
from src.decision_policy import (BeliefMyopicPolicy, OraclePolicy, PolicyContext, QMDPPolicy,
                                 evaluate_actions, explain_decision, policy_region_grid)
from src.inventory_simulator import EnvironmentOptions
from src.pomdp_engine import SentinelEngine
from src.transition_model import TransitionModel


@pytest.fixture()
def models():
    cfg = default_config()
    return cfg, TransitionModel.from_config(cfg), CostModel.from_config(cfg)


def test_all_actions_are_evaluated(models):
    cfg, tm, cm = models
    d = evaluate_actions(cfg.initial_belief, tm, cm)
    assert d.q_values.shape == (len(ACTIONS),) and np.all(np.isfinite(d.q_values))
    assert sorted(d.ranking) == [0, 1, 2, 3]
    assert d.predicted_beliefs.shape == (4, 3)
    assert np.allclose(d.predicted_beliefs.sum(axis=1), 1.0)


def test_cost_decomposition_equals_bellman_expectation(models):
    """c_action + E_b[c_current] + E_{b^-}[c_post]  ==  sum_s b(s) sum_s' T(s'|s,a) C(s,a,s')."""
    cfg, tm, cm = models
    rng = np.random.default_rng(3)
    for _ in range(300):
        b = rng.dirichlet(np.ones(3))
        d = evaluate_actions(b, tm, cm)
        for a in range(4):
            assert d.q_values[a] == pytest.approx(cm.expected_cost_bellman(b, a, tm.matrix(a)), abs=1e-10)
            assert d.q_values[a] == pytest.approx(d.direct[a] + d.current_state[a] + d.post_transition[a])
        linear = b @ cm.one_step_cost_matrix(tm.tensor())
        assert np.allclose(linear, d.q_values, atol=1e-10)


def test_hand_calculated_expected_cost_at_initial_belief(models):
    cfg, tm, cm = models
    d = evaluate_actions(cfg.initial_belief, tm, cm)
    # Continue Monitoring: b^- = [0.526, 0.297, 0.177]; Q = 0 + 0 + 0.526*4 + 0.297*20 + 0.177*150
    assert np.allclose(d.predicted_beliefs[0], [0.526, 0.297, 0.177], atol=1e-12)
    assert d.q_values[0] == pytest.approx(0.526 * 4 + 0.297 * 20 + 0.177 * 150)
    assert d.q_values[0] == pytest.approx(34.594)
    # Standard Reorder: 15 + 0.6*6 + ([0.723, 0.207, 0.070] . [4, 20, 150])
    assert d.q_values[1] == pytest.approx(15 + 3.6 + 0.723 * 4 + 0.207 * 20 + 0.070 * 150)


def test_lowest_expected_cost_action_is_selected(models):
    cfg, tm, cm = models
    rng = np.random.default_rng(11)
    for _ in range(300):
        b = rng.dirichlet(np.ones(3))
        d = evaluate_actions(b, tm, cm)
        assert d.q_values[d.action] == d.q_values.min()
        assert d.margin >= 0 and d.q_values[d.runner_up] >= d.q_values[d.action]
        fast = BeliefMyopicPolicy(tm, cm).act(PolicyContext(0, b, {}))
        assert fast == d.action
        assert d.expected_reward == -d.expected_cost


def test_pure_states_map_to_sensible_actions(models):
    cfg, tm, cm = models
    assert evaluate_actions([1, 0, 0], tm, cm).action == ACTION_KEYS.index("monitor")
    assert evaluate_actions([0, 1, 0], tm, cm).action == ACTION_KEYS.index("reorder")
    assert evaluate_actions([0, 0, 1], tm, cm).action == ACTION_KEYS.index("expedite")
    regions = set(policy_region_grid(tm, cm, 30)["action"])
    assert regions == {0, 1, 2, 3}                      # every action is optimal somewhere


def test_an_alert_changes_the_belief_and_can_change_the_recommendation():
    engine = SentinelEngine()
    assert engine.decision.action == 0                  # Continue Monitoring at b0
    before = engine.belief.copy()
    out = engine.inject_alert("sensor", "Critical inventory warning")
    assert not np.allclose(before, engine.belief)
    assert engine.belief[2] > before[2]
    assert out["policy_changed"] and engine.decision.action != 0
    assert engine.policy_changes == 1 and engine.alerts_processed == 1


def test_recommendation_changes_only_through_the_belief(models):
    cfg, tm, cm = models
    a = evaluate_actions([0.9, 0.08, 0.02], tm, cm).action
    b = evaluate_actions([0.1, 0.3, 0.6], tm, cm).action
    assert a != b
    assert evaluate_actions([0.1, 0.3, 0.6], tm, cm).action == b      # deterministic


def test_different_actions_give_different_predictions(models):
    cfg, tm, cm = models
    preds = [tm.predict(cfg.initial_belief, k) for k in ACTION_KEYS]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.allclose(preds[i], preds[j])
    assert preds[2][2] < preds[1][2] < preds[0][2]      # expedite < reorder < monitor on P(Critical)


def test_injecting_an_alert_never_applies_a_transition():
    """Recommended != executed: an alert is a pure correction of the current period's belief."""
    engine = SentinelEngine()
    obs = engine.observation_model
    engine.inject_alert("supplier", "Significant delivery delay")
    engine.inject_alert("demand", "High demand")
    expected = correct(engine.config.initial_belief,
                       [obs.likelihood_vector("supplier", "Significant delivery delay"),
                        obs.likelihood_vector("demand", "High demand")]).posterior
    assert np.allclose(engine.belief, expected, atol=1e-12)
    assert engine.step == 0 and engine.last_executed_action is None and engine.action_log == []
    # an uninformative alert must leave the belief exactly where it was (no hidden prediction step)
    snapshot = engine.belief.copy()
    engine.inject_alert("sensor", "Sensor malfunction / unreliable reading")
    assert np.allclose(engine.belief, snapshot, atol=1e-12)


def test_only_the_executed_action_drives_the_prediction():
    engine = SentinelEngine()
    engine.inject_alert("sensor", "Critical inventory warning")
    recommended = engine.decision.action
    assert recommended != 0
    belief = engine.belief.copy()
    out = engine.run_next_step(action="monitor", observe=False)       # operator overrides NOVA
    assert out["executed"] == 0 and out["recommended"] == recommended
    assert np.allclose(engine.predicted, engine.transition_model.predict(belief, "monitor"))
    assert not np.allclose(engine.predicted, engine.transition_model.predict(belief, recommended))
    row = engine.action_log[-1]
    assert row["Recommended action"] == ACTIONS[recommended]
    assert row["Executed action"] == ACTIONS[0]
    assert row["Followed recommendation"].startswith("no")


def test_executing_the_recommendation_uses_its_transition_model():
    engine = SentinelEngine()
    engine.inject_alert("sensor", "Critical inventory warning")
    engine.inject_alert("system", "Critical stock warning")
    recommended, belief = engine.decision.action, engine.belief.copy()
    engine.run_next_step(observe=False)
    assert engine.last_executed_action == recommended
    assert np.allclose(engine.predicted, engine.transition_model.predict(belief, recommended))
    assert engine.action_log[-1]["Followed recommendation"] == "yes"


def test_hidden_state_never_influences_beliefs_or_decisions():
    """Same evidence + different hidden truths => identical beliefs and recommendations."""
    script = [("sensor", "Low inventory reading"), ("supplier", "Minor delivery delay"), None,
              ("demand", "High demand"), ("system", "Critical stock warning"), None,
              ("sensor", "Normal inventory reading"), None]
    traces = []
    for seed, initial_state in ((1, 0), (2, 2), (99, 1)):
        engine = SentinelEngine(seed=seed, environment_options=EnvironmentOptions(initial_state=initial_state))
        trace = []
        for item in script:
            if item is None:
                engine.run_next_step(observe=False)
            else:
                engine.inject_alert(*item)
            trace.append((engine.belief.copy(), engine.decision.action, engine.decision.q_values.copy()))
        traces.append(trace)
    hidden = {SentinelEngine(seed=s, environment_options=EnvironmentOptions(initial_state=i)).env.evaluation_state()
              for s, i in ((1, 0), (2, 2), (99, 1))}
    assert len(hidden) == 3                                            # the truths really differ
    for other in traces[1:]:
        for (b0, a0, q0), (b1, a1, q1) in zip(traces[0], other):
            assert np.array_equal(b0, b1) and a0 == a1 and np.array_equal(q0, q1)


def test_decision_functions_have_no_access_to_the_true_state(models):
    cfg, tm, cm = models
    assert list(inspect.signature(evaluate_actions).parameters) == ["belief", "transition_model", "cost_model"]
    assert BeliefMyopicPolicy.requires_oracle is False and OraclePolicy.requires_oracle is True
    with pytest.raises(RuntimeError):
        OraclePolicy(tm, cm).act(PolicyContext(0, cfg.initial_belief, {}, true_state=None))


def test_explanations_are_generated_from_the_live_numbers():
    engine = SentinelEngine()
    first = engine.explanation()
    engine.inject_alert("supplier", "Significant delivery delay")
    second = engine.explanation()
    engine.inject_alert("sensor", "Critical inventory warning")
    third = engine.explanation()
    assert len({first["paragraph"], second["paragraph"], third["paragraph"]}) == 3
    assert "Supplier updates" in second["paragraph"] and "Sensor readings" in third["main_evidence"]
    assert ACTIONS[engine.decision.action].upper() in third["headline"]
    assert f"{engine.decision.expected_cost:.1f}" in third["paragraph"]


def test_qmdp_policy_runs_and_differs_from_myopic_somewhere(models):
    cfg, tm, cm = models
    qmdp, myopic = QMDPPolicy(tm, cm, cfg.discount), BeliefMyopicPolicy(tm, cm)
    grid = policy_region_grid(tm, cm, 20)
    differs = sum(qmdp.act(PolicyContext(0, r[["Healthy", "Low", "Critical"]].to_numpy(float), {})) != r["action"]
                  for _, r in grid.iterrows())
    assert differs > 0
    assert np.all(np.isfinite(qmdp.q_mdp))


def test_explain_decision_without_evidence(models):
    cfg, tm, cm = models
    text = explain_decision(evaluate_actions(cfg.initial_belief, tm, cm), cfg.initial_belief, [], None)
    assert "No new evidence" in text["paragraph"] and "Continue Monitoring" in text["paragraph"]
