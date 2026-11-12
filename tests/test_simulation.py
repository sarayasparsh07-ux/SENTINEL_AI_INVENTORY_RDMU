"""Simulation engine, session logic, scenarios, mismatch experiments and metrics."""
import numpy as np
import pandas as pd
import pytest

from src.belief_updater import correct
from src.config import ACTIONS, CHANNEL_KEYS, default_config
from src.decision_policy import (BeliefMyopicPolicy, FixedPolicy, OraclePolicy, RecordTrustingPolicy)
from src.inventory_simulator import (AgentModels, EnvironmentOptions, InventoryEnvironment,
                                     evaluate_policies, generate_event_dataset, run_episode, service_profile)
from src.metrics import (BELIEF_COLUMNS, calibration_table, expected_calibration_error, gap_closed,
                         mean_confidence_interval, recovery_times, summarise_replications)
from src.pomdp_engine import SentinelEngine
from src.scenarios import CROSS_CHECK_SCENARIOS, DASHBOARD_PRESETS, NOTEBOOK_SCENARIOS, SCENARIOS


def _belief_policy(cfg):
    m = AgentModels.from_config(cfg)
    return BeliefMyopicPolicy(m.transition, m.cost)


def test_same_seed_reproduces_the_episode_exactly():
    cfg = default_config()
    a = run_episode(cfg, _belief_policy(cfg), horizon=40, seed=123)
    b = run_episode(cfg, _belief_policy(cfg), horizon=40, seed=123)
    pd.testing.assert_frame_equal(a, b)
    c = run_episode(cfg, _belief_policy(cfg), horizon=40, seed=124)
    assert not a["hidden_state"].equals(c["hidden_state"]) or not a["obs_sensor"].equals(c["obs_sensor"])


def test_episode_beliefs_are_always_valid_distributions():
    cfg = default_config()
    frame = run_episode(cfg, _belief_policy(cfg), horizon=200, seed=5)
    b = frame[BELIEF_COLUMNS].to_numpy()
    assert np.allclose(b.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(b >= 0) and np.all(b <= 1)
    assert np.allclose(b[0], cfg.initial_belief)
    assert frame["fill_rate"].between(0, 1).all()


def test_common_random_numbers_for_identical_action_sequences():
    cfg = default_config()
    a = run_episode(cfg, FixedPolicy("reorder"), horizon=50, seed=9)
    b = run_episode(cfg, FixedPolicy("reorder"), horizon=50, seed=9)
    assert a["hidden_state"].equals(b["hidden_state"])
    env1, env2 = InventoryEnvironment(cfg, seed=9), InventoryEnvironment(cfg, seed=9)
    assert env1.evaluation_state() == env2.evaluation_state()          # same initial truth for every policy


def test_engine_and_batch_runner_produce_identical_beliefs():
    """The interactive engine and run_episode share the same maths: same seed -> same belief path."""
    cfg = default_config()
    frame = run_episode(cfg, _belief_policy(cfg), horizon=30, seed=77)
    engine = SentinelEngine(cfg, seed=77)
    for _ in range(30):
        engine.run_next_step()                                         # execute recommendation, observe
    records = pd.DataFrame(engine._evaluation_records)
    assert np.allclose(records[BELIEF_COLUMNS].to_numpy(), frame[BELIEF_COLUMNS].to_numpy(), atol=1e-12)
    assert records["executed"].tolist() == frame["action"].tolist()
    assert records["hidden_state"].tolist() == frame["hidden_state"].tolist()
    assert records["realised_cost"].sum() == pytest.approx(frame["cost"].sum())


def test_environment_step_returns_observations_only():
    env = InventoryEnvironment(default_config(), seed=3)
    out = env.step("monitor")
    assert set(out) == set(CHANNEL_KEYS) and all(isinstance(v, int) for v in out.values())


def test_duplicate_guard_ignores_repeats_and_replaces_revisions():
    engine = SentinelEngine()
    first = engine.inject_alert("sensor", "Low inventory reading")
    after_first = engine.belief.copy()
    repeat = engine.inject_alert("sensor", "Low inventory reading")
    assert repeat["status"] == "duplicate ignored"
    assert np.allclose(engine.belief, after_first) and engine.alerts_processed == 1
    revised = engine.inject_alert("sensor", "Critical inventory warning")
    assert "replaced" in revised["status"]
    expected = correct(engine.config.initial_belief,
                       [engine.observation_model.likelihood_vector("sensor", "Critical inventory warning")]).posterior
    assert np.allclose(engine.belief, expected, atol=1e-12)            # old reading no longer counted
    assert len(engine.evidence) == 1 and len(engine.event_log) == 3
    assert first["status"] == "applied"


def test_switching_the_guard_off_double_counts_evidence():
    guarded, unguarded = SentinelEngine(), SentinelEngine(deduplicate=False)
    for engine in (guarded, unguarded):
        for _ in range(3):
            engine.inject_alert("system", "Critical stock warning")
    assert len(guarded.evidence) == 1 and len(unguarded.evidence) == 3
    assert unguarded.belief[2] > guarded.belief[2] + 0.2               # overconfident in Critical
    assert "independent" in unguarded.event_log[-1]["Status"]


def test_reset_restores_the_initial_configuration():
    engine = SentinelEngine()
    engine.inject_alert("sensor", "Critical inventory warning")
    engine.run_next_step()
    engine.run_next_step(action="expedite")
    assert engine.step == 2 and engine.event_log and engine.action_log
    engine.reset()
    assert engine.step == 0 and engine.alerts_processed == 0 and engine.policy_changes == 0
    assert np.allclose(engine.belief, [0.6, 0.3, 0.1])
    assert engine.event_log == [] and engine.action_log == [] and engine.evidence == []
    assert len(engine.timeline) == 1 and engine.decision.action == 0


def test_alerts_do_not_reset_the_model():
    engine = SentinelEngine()
    engine.run_next_step()
    engine.run_next_step()
    step, n_events = engine.step, len(engine.timeline)
    engine.inject_alert("demand", "Unexpected demand surge")
    assert engine.step == step and len(engine.timeline) == n_events + 1 and len(engine.action_log) == 2


def test_apply_config_recomputes_dependent_outputs_and_validates():
    engine = SentinelEngine()
    engine.inject_alert("sensor", "Low inventory reading")
    belief_before, q_before = engine.belief.copy(), engine.decision.q_values.copy()
    cfg = engine.config.copy()
    cfg.reliability["sensor"] = 0.4
    cfg.post_state_cost = np.array([4.0, 20.0, 400.0])
    engine.apply_config(cfg)
    assert not np.allclose(engine.belief, belief_before)               # likelihood recomputed
    assert not np.allclose(engine.decision.q_values, q_before)         # costs recomputed
    expected = correct([0.6, 0.3, 0.1], [engine.observation_model.likelihood_vector("sensor", 1, 0.4)]).posterior
    assert np.allclose(engine.belief, expected, atol=1e-12)
    bad = engine.config.copy()
    bad.transitions["monitor"][0] = [0.7, 0.7, 0.1]
    with pytest.raises(ValueError):
        engine.apply_config(bad)
    assert np.allclose(engine.belief, expected, atol=1e-12)            # failed update leaves the session intact


def test_initial_belief_is_configurable():
    cfg = default_config()
    cfg.initial_belief = np.array([0.2, 0.3, 0.5])
    engine = SentinelEngine(cfg)
    assert np.allclose(engine.belief, [0.2, 0.3, 0.5]) and engine.decision.action != 0
    cfg.initial_belief = np.array([0.5, 0.6, 0.1])
    with pytest.raises(ValueError):
        SentinelEngine(cfg)


def test_invalid_alert_inputs_raise_clear_errors():
    engine = SentinelEngine()
    with pytest.raises(ValueError):
        engine.inject_alert("radar", "Normal inventory reading")
    with pytest.raises(ValueError):
        engine.inject_alert("sensor", "Totally unknown")
    with pytest.raises(ValueError):
        engine.inject_alert("sensor", "Normal inventory reading", reliability=2.0)
    with pytest.raises(ValueError):
        engine.run_next_step(action="teleport stock")
    assert engine.step == 0 and engine.alerts_processed == 0


@pytest.mark.parametrize("key", sorted(SCENARIOS))
def test_scenarios_are_deterministic_and_valid(key):
    a, b = SentinelEngine(seed=1), SentinelEngine(seed=2)
    a.play_scenario(SCENARIOS[key])
    b.play_scenario(SCENARIOS[key])
    ta, tb = a.timeline_frame(), b.timeline_frame()
    assert np.allclose(ta[BELIEF_COLUMNS].to_numpy(), tb[BELIEF_COLUMNS].to_numpy(), atol=0)
    assert ta["recommended"].tolist() == tb["recommended"].tolist()
    assert np.allclose(ta[BELIEF_COLUMNS].sum(axis=1), 1.0)


def test_scenario_catalogue_is_complete():
    assert len(DASHBOARD_PRESETS) >= 7 and len(NOTEBOOK_SCENARIOS) == 8 and len(CROSS_CHECK_SCENARIOS) == 3
    assert all(k in SCENARIOS for k in DASHBOARD_PRESETS + NOTEBOOK_SCENARIOS + CROSS_CHECK_SCENARIOS)


def test_scenario_outcomes_tell_the_expected_story():
    e = SentinelEngine()
    e.play_scenario(SCENARIOS["normal_ops"])
    assert e.belief[0] > 0.9 and e.decision.action == 0
    e.play_scenario(SCENARIOS["critical_warning"])
    assert e.belief[2] > 0.6 and e.decision.action == 2               # Expedite
    e.play_scenario(SCENARIOS["recovery"])
    t = e.timeline_frame()
    assert t["b_critical"].max() > 0.6 and e.belief[0] > 0.8 and e.decision.action == 0
    e.play_scenario(SCENARIOS["consecutive"])
    assert len(set(e.timeline_frame()["recommended"])) >= 3           # recommendation escalates


def test_unreliable_sensor_moves_belief_less_than_reliable_sensor():
    weak, strong = SentinelEngine(), SentinelEngine()
    weak.inject_alert("sensor", "Low inventory reading", reliability=0.30)
    strong.inject_alert("sensor", "Low inventory reading", reliability=0.95)
    assert 0.30 < weak.belief[1] < strong.belief[1]


def test_service_profile_is_monotone_in_state():
    p = service_profile(default_config())
    assert p["stockout_probability"][0] < p["stockout_probability"][1] < p["stockout_probability"][2]
    assert p["fill_rate"][0] > p["fill_rate"][1] > p["fill_rate"][2]
    assert np.all((p["fill_rate"] >= 0) & (p["fill_rate"] <= 1))


def test_event_dataset_separates_observations_from_hidden_labels():
    cfg = default_config()
    data = generate_event_dataset(cfg, _belief_policy(cfg), horizon=30, seed=1)
    assert len(data) == 30
    hidden = [c for c in data.columns if c.startswith("eval_")]
    assert "eval_hidden_state" in hidden and "sensor_observation" in data.columns
    assert set(data["executed_action"]).issubset(set(ACTIONS))


def test_belief_agent_beats_naive_fixed_policies_in_simulation():
    cfg = default_config()
    factories = {
        "belief": lambda m: BeliefMyopicPolicy(m.transition, m.cost),
        "monitor": lambda m: FixedPolicy("monitor"),
        "reorder": lambda m: FixedPolicy("reorder"),
        "record": lambda m: RecordTrustingPolicy(),
        "oracle": lambda m: OraclePolicy(m.transition, m.cost),
    }
    results = evaluate_policies(cfg, factories, n_replications=60, horizon=50, seed=2026)
    cost = summarise_replications(results).set_index("policy")["total_cost"]
    assert cost["oracle"] < cost["belief"] < cost["reorder"] < cost["monitor"]
    assert 0 < gap_closed(cost["reorder"], cost["belief"], cost["oracle"]) < 1
    assert len(results) == 5 * 60


def test_well_specified_beliefs_are_calibrated_and_mismatch_degrades_them():
    """Model-mismatch experiment: a system alert that echoes the sensor is double counted."""
    cfg = default_config()
    factory = {"belief": lambda m: BeliefMyopicPolicy(m.transition, m.cost)}
    good, traj = evaluate_policies(cfg, factory, 150, 50, seed=11, keep_trajectories=True)
    table = calibration_table(traj["belief"][BELIEF_COLUMNS].to_numpy(), traj["belief"]["hidden_state"].to_numpy())
    assert expected_calibration_error(table) < 0.03
    mismatch = EnvironmentOptions(system_echo_probability=1.0, true_reliability={"sensor": 0.6})
    bad = evaluate_policies(cfg, factory, 150, 50, seed=11, options=mismatch)
    assert bad["log_loss"].mean() > good["log_loss"].mean()
    assert bad["brier"].mean() > good["brier"].mean()


def test_metric_helpers():
    ci = mean_confidence_interval([1.0, 2.0, 3.0, 4.0, 5.0])
    assert ci["mean"] == 3.0 and ci["low"] < 3.0 < ci["high"] and ci["n"] == 5
    assert recovery_times([0, 2, 2, 1, 0, 2, 0]) == [3, 1]
    assert gap_closed(100, 70, 40) == pytest.approx(0.5)


def test_live_evaluation_metrics_use_hidden_truth_only_for_reporting():
    engine = SentinelEngine(seed=4)
    assert engine.evaluation_metrics() is None
    for _ in range(12):
        engine.run_next_step()
    m = engine.evaluation_metrics()
    assert m["periods"] == 12 and 0 <= m["brier"] <= 2 and 0 <= m["accuracy"] <= 1
    frame = engine.evaluation_frame()
    assert {"hidden_state", "inventory_units", "demand_units", "fill_rate", "stockout"}.issubset(frame.columns)


def test_inventory_bands_partition_the_capacity_range():
    """Healthy [600, 1,000], Low [250, 600), Critical [0, 250): non-overlapping and gap-free."""
    from src.config import check_inventory_bands, inventory_band_label
    assert check_inventory_bands() == []
    assert [inventory_band_label(i) for i in range(3)] == ["[600, 1,000]", "[250, 600)", "[0, 250)"]
    assert check_inventory_bands([[600, 1000], [250, 599], [0, 250]]) != []      # a gap is detected
    assert check_inventory_bands([[590, 1000], [250, 600], [0, 250]]) != []      # an overlap is detected


def test_every_inventory_snapshot_lies_in_its_states_half_open_band():
    from src.config import INVENTORY_BANDS
    cfg = default_config()
    frame = pd.concat([run_episode(cfg, FixedPolicy("monitor"), horizon=200, seed=s) for s in range(300)])
    lows, highs = INVENTORY_BANDS[frame["hidden_state"], 0], INVENTORY_BANDS[frame["hidden_state"], 1]
    assert ((frame["inventory_units"] >= lows) & (frame["inventory_units"] < highs)).all()   # logged values too
    assert set(frame["hidden_state"]) == {0, 1, 2}                                # all three bands exercised
