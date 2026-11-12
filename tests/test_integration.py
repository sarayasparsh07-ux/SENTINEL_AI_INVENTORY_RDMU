"""Notebook <-> dashboard numerical consistency.

Three identical alert scenarios are run through
  (a) the low-level maths the notebook uses (no engine),
  (b) the shared SentinelEngine, and
  (c) the real dashboard script driven headlessly,
and, when the executed notebook has written its results to
outputs/cross_check_notebook.json, (d) the notebook's own saved numbers.
Prior, predicted belief, joint likelihood, posterior, expected cost per action
and the recommended action must agree to floating-point tolerance.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from src.config import default_config
from src.pomdp_engine import SentinelEngine
from src.scenarios import CROSS_CHECK_SCENARIOS, SCENARIOS, replay_with_primitives

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-10
FIELDS = ("prior", "predicted", "joint_likelihood", "posterior", "q_values")


def engine_result(engine: SentinelEngine) -> dict:
    update = engine.bayes_breakdown()
    return {"prior": update.prior, "predicted": update.predicted, "joint_likelihood": update.joint_likelihood,
            "posterior": update.posterior, "q_values": engine.decision.q_values, "action": int(engine.decision.action)}


def assert_same(a: dict, b: dict) -> None:
    for name in FIELDS:
        assert np.allclose(np.asarray(a[name], dtype=float), np.asarray(b[name], dtype=float),
                           atol=TOLERANCE, rtol=0), name
    assert int(a["action"]) == int(b["action"])


@pytest.mark.parametrize("key", sorted(SCENARIOS))
def test_engine_matches_low_level_maths_for_every_scenario(key):
    engine = SentinelEngine(default_config())
    engine.play_scenario(SCENARIOS[key])
    assert_same(replay_with_primitives(SCENARIOS[key]), engine_result(engine))


@pytest.mark.parametrize("key", CROSS_CHECK_SCENARIOS)
def test_dashboard_matches_notebook_maths(key):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120).run()
    at.selectbox(key="scenario_key").set_value(key).run()
    at.button(key="btn_scenario").click().run()
    assert not at.exception
    dashboard = engine_result(at.session_state["engine"])
    assert_same(replay_with_primitives(SCENARIOS[key]), dashboard)
    saved = ROOT / "outputs" / "cross_check_notebook.json"
    if saved.exists():
        notebook = json.loads(saved.read_text())[key]
        assert_same(notebook, dashboard)
