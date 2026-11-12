"""Functional dashboard tests (headless, via streamlit.testing.v1.AppTest).

These drive the real dashboard script: every button and control is exercised
and the shared engine held in session state is inspected afterwards.
The timed auto-simulation fragment itself is exercised in a real browser
(see README, 'Testing and validation'); here we test its start/pause/stop logic.
"""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.config import ACTIONS, default_config  # noqa: E402
from src.pomdp_engine import SentinelEngine  # noqa: E402
from src.scenarios import DASHBOARD_PRESETS, SCENARIOS  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
APPLY = "FormSubmitter:config_form_0-APPLY CONFIGURATION"
RESTORE = "FormSubmitter:config_form_0-RESTORE DEFAULTS"


@pytest.fixture()
def app():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def engine(at) -> SentinelEngine:
    return at.session_state["engine"]


def test_application_launches_with_all_sections_and_controls(app):
    keys = {b.key for b in app.button}
    assert {"btn_inject", "btn_step", "btn_start", "btn_pause", "btn_reset", "btn_scenario", "btn_batch",
            APPLY, RESTORE} <= keys
    assert len(app.tabs) >= 11                         # ops (3) + model explorer (5) + lab (3)
    assert np.allclose(engine(app).belief, [0.6, 0.3, 0.1])
    assert engine(app).decision.action_name == "Continue Monitoring"
    assert len(app.get("plotly_chart")) >= 6


def test_inject_alert_updates_belief_kpis_recommendation_and_log(app):
    app.radio(key="alert_channel").set_value("sensor").run()
    app.selectbox(key="alert_obs_sensor").set_value(2).run()          # Critical inventory warning
    app.button(key="btn_inject").click().run()
    assert not app.exception
    eng = engine(app)
    assert eng.alerts_processed == 1 and eng.step == 0                # no time passed, nothing executed
    assert eng.belief[2] > 0.3 and eng.decision.action != 0
    assert eng.policy_changes == 1 and len(eng.event_log) == 1
    assert eng.event_log[0]["Observation"] == "Critical inventory warning"
    assert eng.event_log[0]["Recommendation changed"] == "yes"
    assert any("Alert injected" in s.value for s in app.success)
    assert len(app.dataframe) >= 1                                    # event log table is rendered


def test_reliability_slider_and_operator_note_reach_the_engine(app):
    app.slider(key="alert_rel_sensor").set_value(0).run()
    app.text_input(key="alert_note").set_value("sensor under maintenance").run()
    app.selectbox(key="alert_obs_sensor").set_value(2).run()
    app.button(key="btn_inject").click().run()
    eng = engine(app)
    assert np.allclose(eng.belief, [0.6, 0.3, 0.1], atol=1e-9)        # 0% reliable -> uninformative
    assert eng.event_log[0]["Reliability"] == "0%"
    assert eng.event_log[0]["Operator note"] == "sensor under maintenance"


def test_duplicate_guard_toggle(app):
    app.selectbox(key="alert_obs_sensor").set_value(1).run()
    app.button(key="btn_inject").click().run()
    first = engine(app).belief.copy()
    app.button(key="btn_inject").click().run()
    assert np.allclose(engine(app).belief, first) and any("Duplicate ignored" in w.value for w in app.warning)
    app.toggle(key="dedupe_guard").set_value(False).run()
    app.button(key="btn_inject").click().run()
    assert engine(app).belief[1] > first[1]                            # counted again: double counting
    assert len(engine(app).evidence) == 2


def test_run_next_step_executes_recommendation_and_records_history(app):
    app.button(key="btn_step").click().run()
    eng = engine(app)
    assert eng.step == 1 and len(eng.action_log) == 1 and eng.alerts_processed == 4
    assert eng.action_log[0]["Followed recommendation"] == "yes"
    assert eng.action_log[0]["Observed subsequent evidence"] != "(none yet)"
    assert len(eng.event_log) == 4 and len(eng.timeline) == 3          # init, predict, correct


def test_manual_override_is_distinguished_from_the_recommendation(app):
    app.selectbox(key="exec_choice").set_value("expedite").run()
    app.toggle(key="observe_env").set_value(False).run()
    app.button(key="btn_step").click().run()
    eng = engine(app)
    row = eng.action_log[0]
    assert row["Recommended action"] == "Continue Monitoring"
    assert row["Executed action"] == "Expedite Supplier Delivery"
    assert row["Followed recommendation"].startswith("no")
    assert np.allclose(eng.predicted, eng.transition_model.predict([0.6, 0.3, 0.1], "expedite"))
    assert eng.alerts_processed == 0                                   # observations switched off


def test_start_pause_and_duration_limit(app):
    app.button(key="btn_start").click().run()
    assert app.session_state["auto_running"] is True
    app.button(key="btn_pause").click().run()
    assert app.session_state["auto_running"] is False
    eng = engine(app)
    cfg = eng.config.copy()
    cfg.horizon = 5
    eng.apply_config(cfg)
    for _ in range(5):
        eng.run_next_step()
    app.run()
    app.button(key="btn_start").click().run()
    assert app.session_state["auto_running"] is False                  # refuses to run past the duration
    assert any("duration already reached" in w.value for w in app.warning)


def test_reset_restores_initial_configuration(app):
    app.selectbox(key="alert_obs_sensor").set_value(2).run()
    app.button(key="btn_inject").click().run()
    app.button(key="btn_step").click().run()
    assert engine(app).step == 1
    app.button(key="btn_reset").click().run()
    eng = engine(app)
    assert eng.step == 0 and eng.alerts_processed == 0 and eng.policy_changes == 0
    assert np.allclose(eng.belief, [0.6, 0.3, 0.1]) and eng.event_log == [] and eng.action_log == []


@pytest.mark.parametrize("key", DASHBOARD_PRESETS)
def test_every_preset_scenario_loads_and_matches_the_engine(app, key):
    app.selectbox(key="scenario_key").set_value(key).run()
    app.button(key="btn_scenario").click().run()
    assert not app.exception
    reference = SentinelEngine(default_config())
    reference.play_scenario(SCENARIOS[key])
    eng = engine(app)
    assert np.allclose(eng.belief, reference.belief, atol=1e-12)
    assert eng.decision.action == reference.decision.action
    assert np.allclose(eng.decision.q_values, reference.decision.q_values, atol=1e-12)


def test_invalid_configuration_is_rejected_with_a_meaningful_message(app):
    before = engine(app).config.initial_belief.copy()
    app.number_input(key="cfg_b0_0").set_value(80.0)
    app.button(key=APPLY).click().run()
    assert not app.exception
    assert any("Initial belief must sum to 100%" in e.value for e in app.error)
    assert np.allclose(engine(app).config.initial_belief, before)


def test_valid_configuration_recomputes_outputs_and_defaults_can_be_restored(app):
    q_before = engine(app).decision.q_values.copy()
    app.number_input(key="cfg_pc_2_0").set_value(400.0)
    app.slider(key="cfg_rho_0").set_value(0.5)
    app.button(key=APPLY).click().run()
    assert not app.exception and not app.error
    eng = engine(app)
    assert eng.config.post_state_cost[2] == 400.0 and eng.config.supplier_risk == 0.5
    assert not np.allclose(eng.decision.q_values, q_before)
    assert eng.decision.action != 0                                    # dearer stock-outs -> intervene
    app.button(key=RESTORE).click().run()
    eng = engine(app)
    assert eng.config.post_state_cost[2] == 150.0 and eng.config.supplier_risk == 0.0
    assert np.allclose(eng.decision.q_values, q_before)


def test_initial_belief_can_be_changed_from_the_dashboard(app):
    app.number_input(key="cfg_b0_0").set_value(20.0)
    app.number_input(key="cfg_b1_0").set_value(30.0)
    app.number_input(key="cfg_b2_0").set_value(50.0)
    app.button(key=APPLY).click().run()
    assert not app.error
    assert np.allclose(engine(app).belief, [0.2, 0.3, 0.5])
    assert engine(app).decision.action_name in ACTIONS[1:]


def test_batch_evaluation_runs_and_reports_all_policies(app):
    app.slider(key="batch_reps").set_value(20).run()
    app.button(key="btn_batch").click().run()
    assert not app.exception
    batch = app.session_state["batch"]
    assert batch is not None and batch["reps"] == 20
    summary = batch["summary"].set_index("policy")
    assert len(summary) == 6
    assert summary.loc["Full-information myopic oracle", "total_cost"] < summary.loc["SENTINEL belief agent", "total_cost"]
    assert 0 <= summary.loc["SENTINEL belief agent", "brier"] <= 2


def test_debug_view_reveals_ground_truth_only_on_request(app):
    app.button(key="btn_step").click().run()
    n_frames_hidden = len(app.dataframe)
    app.toggle(key="reveal_truth").set_value(True).run()
    assert not app.exception
    assert len(app.dataframe) == n_frames_hidden + 1                   # hidden-truth table appears only now
