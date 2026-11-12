"""Validation of transition and observation matrices."""
import numpy as np
import pytest

from src.config import ACTION_KEYS, CHANNEL_KEYS, default_config
from src.observation_model import ObservationModel
from src.transition_model import TransitionModel, validate_stochastic_matrix


def test_every_transition_row_sums_to_one():
    tm = TransitionModel.from_config(default_config())
    for key in ACTION_KEYS:
        m = tm.matrix(key)
        assert m.shape == (3, 3)
        assert np.allclose(m.sum(axis=1), 1.0, atol=1e-12)
        assert np.all(m >= 0) and np.all(m <= 1)


@pytest.mark.parametrize("kappa", [0.0, 0.1, 0.3, 0.5])
@pytest.mark.parametrize("rho", [0.0, 0.25, 0.6, 1.0])
def test_modifiers_preserve_stochasticity(kappa, rho):
    cfg = default_config()
    tm = TransitionModel(cfg.transitions, kappa, rho)
    assert np.allclose(tm.tensor().sum(axis=2), 1.0, atol=1e-12)
    assert np.all(tm.tensor() >= 0)


def test_supplier_risk_one_makes_supplier_actions_equal_to_monitoring():
    cfg = default_config()
    tm = TransitionModel(cfg.transitions, 0.0, 1.0)
    assert np.allclose(tm.matrix("reorder"), tm.matrix("monitor"))
    assert np.allclose(tm.matrix("expedite"), tm.matrix("monitor"))
    assert np.allclose(tm.matrix("reallocate"), cfg.transitions["reallocate"])     # internal action unaffected


def test_demand_volatility_increases_depletion():
    cfg = default_config()
    base = TransitionModel(cfg.transitions, 0.0, 0.0)
    volatile = TransitionModel(cfg.transitions, 0.2, 0.0)
    for key in ACTION_KEYS:
        assert np.all(volatile.matrix(key)[:, 2] >= base.matrix(key)[:, 2] - 1e-12)
        assert np.all(volatile.matrix(key)[:, 0] <= base.matrix(key)[:, 0] + 1e-12)


def test_action_effects_are_ordered_sensibly_from_critical():
    tm = TransitionModel.from_config(default_config())
    stay_critical = {k: tm.matrix(k)[2, 2] for k in ACTION_KEYS}
    assert stay_critical["expedite"] < stay_critical["reallocate"] < stay_critical["reorder"] < stay_critical["monitor"]
    recover = {k: tm.matrix(k)[2, 0] for k in ACTION_KEYS}
    assert recover["expedite"] > recover["reorder"] > recover["monitor"]


def test_invalid_transition_matrices_are_rejected():
    cfg = default_config()
    bad = {k: v.copy() for k, v in cfg.transitions.items()}
    bad["reorder"][0] = [0.5, 0.3, 0.3]
    with pytest.raises(ValueError):
        TransitionModel(bad)
    bad = {k: v.copy() for k, v in cfg.transitions.items()}
    bad["monitor"][1] = [1.2, -0.1, -0.1]
    with pytest.raises(ValueError):
        TransitionModel(bad)
    with pytest.raises(ValueError):
        TransitionModel(cfg.transitions, demand_volatility=0.9)
    with pytest.raises(ValueError):
        validate_stochastic_matrix(np.array([[0.5, 0.6], [0.5, 0.5]]))


def test_config_validation_reports_readable_problems():
    cfg = default_config()
    assert cfg.validate() == []
    cfg.initial_belief = np.array([0.5, 0.3, 0.3])
    cfg.transitions["monitor"][0] = [0.9, 0.2, 0.0]
    cfg.reliability["sensor"] = 1.5
    problems = cfg.validate()
    assert len(problems) == 3
    assert any("Initial belief" in p for p in problems)
    assert any("Transition matrix 'monitor'" in p for p in problems)
    assert any("Reliability" in p for p in problems)


def test_observation_likelihoods_are_valid():
    om = ObservationModel.from_config(default_config())
    om.validate()
    for key in CHANNEL_KEYS:
        assert np.allclose(om.nominal[key].sum(axis=1), 1.0, atol=1e-12)
        assert np.all(om.nominal[key] > 0)             # epsilon floor: nothing is impossible


def test_observations_are_informative_in_the_expected_direction():
    om = ObservationModel.from_config(default_config())
    crit = om.likelihood_vector("sensor", "Critical inventory warning")
    assert crit[2] > crit[1] > crit[0]
    normal = om.likelihood_vector("sensor", "Normal inventory reading")
    assert normal[0] > normal[1] > normal[2]
    surge = om.likelihood_vector("demand", "Unexpected demand surge")
    assert surge[2] > surge[0]
