"""Mathematical validation of the Bayes filter, reliability model and scoring rules."""
import itertools

import numpy as np
import pytest

from src.belief_updater import (BeliefError, bayes_update, belief_concentration, correct,
                                correct_sequential, predict, validate_belief)
from src.config import CHANNEL_KEYS, CHANNELS, default_config
from src.metrics import brier_score, brier_scores, log_loss
from src.observation_model import ObservationModel
from src.transition_model import TransitionModel


@pytest.fixture()
def cfg():
    return default_config()


@pytest.fixture()
def obs(cfg):
    return ObservationModel.from_config(cfg)


def test_initial_belief_is_a_distribution(cfg):
    b = validate_belief(cfg.initial_belief)
    assert np.isclose(b.sum(), 1.0) and np.all(b >= 0) and np.all(b <= 1)
    assert np.allclose(b, [0.60, 0.30, 0.10])


def test_invalid_beliefs_are_rejected():
    for bad in ([0.5, 0.5], [0.7, 0.2, 0.2], [-0.1, 0.6, 0.5], [np.nan, 0.5, 0.5], [1.2, -0.1, -0.1]):
        with pytest.raises(BeliefError):
            validate_belief(bad)


def test_worked_example_matches_hand_calculation(cfg, obs):
    """b0 = [.6,.3,.1], sensor 'Low inventory reading' at reliability 1.0 (nominal likelihoods)."""
    z = obs.likelihood_vector("sensor", "Low inventory reading", reliability=1.0)
    assert np.allclose(z, [0.12, 0.68, 0.21], atol=1e-8)
    unnormalised = np.array([0.6 * 0.12, 0.3 * 0.68, 0.1 * 0.21])      # 0.072, 0.204, 0.021
    expected = unnormalised / unnormalised.sum()                        # eta^-1 = 0.297
    result = correct(cfg.initial_belief, [z])
    assert np.allclose(result.posterior, expected, atol=1e-9)
    assert np.isclose(result.evidence_probability, 0.297, atol=1e-9)
    assert np.allclose(result.posterior, [0.242424, 0.686869, 0.070707], atol=1e-6)


def test_posteriors_are_valid_for_every_single_observation(cfg, obs):
    for channel in CHANNEL_KEYS:
        for idx in range(obs.n_observations(channel)):
            for r in (0.0, 0.3, 0.9, 1.0):
                post = correct(cfg.initial_belief, [obs.likelihood_vector(channel, idx, r)]).posterior
                assert abs(post.sum() - 1.0) < 1e-12
                assert np.all(post >= 0) and np.all(post <= 1)


def test_joint_correction_equals_sequential_correction_in_any_order(cfg, obs):
    rng = np.random.default_rng(7)
    for _ in range(200):
        prior = rng.dirichlet(np.ones(3))
        vectors = [obs.likelihood_vector(k, int(rng.integers(obs.n_observations(k))), float(rng.uniform(0.2, 1)))
                   for k in CHANNEL_KEYS]
        joint = correct(prior, vectors).posterior
        for order in itertools.permutations(range(len(vectors))):
            sequential = correct_sequential(prior, [vectors[i] for i in order])[-1]
            assert np.allclose(joint, sequential, atol=1e-12)


def test_prediction_is_applied_once_per_period_not_once_per_observation(cfg, obs):
    t = TransitionModel.from_config(cfg).matrix("monitor")
    vectors = [obs.likelihood_vector("sensor", 1), obs.likelihood_vector("demand", 2)]
    once = bayes_update(cfg.initial_belief, t, vectors).posterior
    wrong = cfg.initial_belief
    for v in vectors:                       # the bug this test guards against: predict before EVERY observation
        wrong = bayes_update(wrong, t, [v]).posterior
    assert not np.allclose(once, wrong, atol=1e-4)
    assert np.allclose(once, correct(predict(cfg.initial_belief, t), vectors).posterior, atol=1e-12)


def test_zero_reliability_is_uninformative(cfg, obs):
    for channel in CHANNEL_KEYS:
        for idx in range(obs.n_observations(channel)):
            v = obs.likelihood_vector(channel, idx, reliability=0.0)
            assert np.allclose(v, v[0])                      # identical across states
            assert np.allclose(correct(cfg.initial_belief, [v]).posterior, cfg.initial_belief, atol=1e-12)


def test_malfunction_symbols_do_not_move_the_belief(cfg, obs):
    for channel, label in (("sensor", "Sensor malfunction / unreliable reading"),
                           ("system", "Monitoring system malfunction")):
        post = correct(cfg.initial_belief, [obs.likelihood_vector(channel, label)]).posterior
        assert np.allclose(post, cfg.initial_belief, atol=1e-9)


def test_reliability_shrinks_the_update_monotonically(cfg, obs):
    shifts = []
    for r in (0.0, 0.25, 0.5, 0.75, 1.0):
        post = correct(cfg.initial_belief, [obs.likelihood_vector("sensor", "Critical inventory warning", r)]).posterior
        shifts.append(post[2] - cfg.initial_belief[2])
    assert shifts[0] == pytest.approx(0.0, abs=1e-12)
    assert all(a < b for a, b in zip(shifts, shifts[1:]))


def test_reliability_adjusted_matrices_are_valid_distributions(obs):
    for channel in CHANNEL_KEYS:
        for r in np.linspace(0, 1, 11):
            m = obs.effective_matrix(channel, float(r))
            assert np.allclose(m.sum(axis=1), 1.0, atol=1e-12)
            assert np.all(m >= 0) and np.all(m <= 1)
            assert m.shape[1] == len(CHANNELS[channel]["observations"])


def test_invalid_reliability_and_observation_are_rejected(obs):
    with pytest.raises(ValueError):
        obs.likelihood_vector("sensor", 0, reliability=1.4)
    with pytest.raises(ValueError):
        obs.likelihood_vector("sensor", "No such reading")
    with pytest.raises(ValueError):
        obs.likelihood_vector("satellite", 0)


def test_impossible_evidence_is_flagged_and_belief_kept():
    prior = np.array([0.5, 0.5, 0.0])
    result = correct(prior, [np.array([0.0, 0.0, 0.9])])     # only the zero-probability state explains it
    assert result.impossible and result.evidence_probability == 0.0
    assert np.allclose(result.posterior, prior)
    result = correct([0.2, 0.3, 0.5], [np.zeros(3)])
    assert result.impossible and np.allclose(result.posterior, [0.2, 0.3, 0.5])


def test_partial_zero_likelihood_is_handled():
    result = correct([0.6, 0.3, 0.1], [np.array([0.0, 0.5, 0.5])])
    assert not result.impossible
    assert np.allclose(result.posterior, [0.0, 0.75, 0.25])


def test_log_space_normalisation_survives_extreme_evidence():
    tiny = [np.array([1e-9, 3e-9, 2e-9]) for _ in range(400)]          # naive product underflows to 0
    assert np.prod([v[1] for v in tiny]) == 0.0
    result = correct([0.6, 0.3, 0.1], tiny)
    assert not result.impossible
    assert np.all(np.isfinite(result.posterior)) and abs(result.posterior.sum() - 1) < 1e-12
    assert result.posterior[1] > 0.999999


def test_likelihood_vectors_are_validated():
    for bad in ([0.5, 0.5], [0.1, -0.2, 0.3], [0.1, np.nan, 0.3], [0.1, 1.5, 0.3]):
        with pytest.raises(BeliefError):
            correct([0.6, 0.3, 0.1], [np.array(bad)])


def test_belief_concentration_bounds():
    assert belief_concentration([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0, abs=1e-12)
    assert belief_concentration([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert 0 < belief_concentration([0.6, 0.3, 0.1]) < 1


def test_multiclass_brier_score_reference_values():
    """Mandatory correction: for [0.60, 0.30, 0.10] the scores are 0.26, 0.86 and 1.26."""
    b = np.array([[0.60, 0.30, 0.10]] * 3)
    assert np.allclose(brier_scores(b, [0, 1, 2]), [0.26, 0.86, 1.26], atol=1e-12)
    assert brier_score([[1, 0, 0]], [0]) == 0.0
    assert brier_score([[1, 0, 0]], [2]) == 2.0                        # worst case of the 0-2 form
    assert brier_score([[1 / 3, 1 / 3, 1 / 3]], [1]) == pytest.approx(2 / 3)


def test_log_loss_reference_values():
    assert log_loss([[0.6, 0.3, 0.1]], [0]) == pytest.approx(-np.log(0.6))
    assert np.isfinite(log_loss([[1.0, 0.0, 0.0]], [2]))               # clipped, not infinite
