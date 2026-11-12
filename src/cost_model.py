"""Transition-dependent cost model and its expectation under a belief.

    C(s, a, s') = c_action(a) + c_current(s, a) + c_post(s')        R = -C

    Q(b, a) = sum_s b(s) sum_s' T(s'|s,a) C(s, a, s')
            = c_action(a) + sum_s b(s) c_current(s, a) + sum_s' b^-_a(s') c_post(s')

The second line is what the engine evaluates; the first line is the textbook
Bellman-style one-step expectation. A unit test asserts they are identical.
The formula in the assignment brief,  Q = C_action(a) + sum b^-_a(s) C_state(s, a),
is the special case c_current = 0.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .config import N_ACTIONS, N_STATES, SentinelConfig, action_index


class CostModel:
    def __init__(self, action_cost, current_state_cost, post_state_cost):
        self.action_cost = np.asarray(action_cost, dtype=float).copy()
        self.current_state_cost = np.asarray(current_state_cost, dtype=float).copy()
        self.post_state_cost = np.asarray(post_state_cost, dtype=float).copy()
        if self.action_cost.shape != (N_ACTIONS,):
            raise ValueError("action_cost must have one entry per action.")
        if self.current_state_cost.shape != (N_STATES, N_ACTIONS):
            raise ValueError("current_state_cost must have shape (states, actions).")
        if self.post_state_cost.shape != (N_STATES,):
            raise ValueError("post_state_cost must have one entry per state.")
        for name, arr in (("action_cost", self.action_cost),
                          ("current_state_cost", self.current_state_cost),
                          ("post_state_cost", self.post_state_cost)):
            if not np.all(np.isfinite(arr)) or np.any(arr < 0):
                raise ValueError(f"{name} must be finite and non-negative.")

    @classmethod
    def from_config(cls, config: SentinelConfig) -> "CostModel":
        return cls(config.action_cost, config.current_state_cost, config.post_state_cost)

    def transition_cost(self, state: int, action, next_state: int) -> float:
        """Realised cost C(s, a, s') of one executed transition."""
        a = action_index(action)
        return float(self.action_cost[a] + self.current_state_cost[int(state), a]
                     + self.post_state_cost[int(next_state)])

    def reward(self, state: int, action, next_state: int) -> float:
        return -self.transition_cost(state, action, next_state)

    def cost_tensor(self) -> np.ndarray:
        """C[a, s, s'] for every action, current state and next state."""
        tensor = np.zeros((N_ACTIONS, N_STATES, N_STATES))
        for a in range(N_ACTIONS):
            tensor[a] = (self.action_cost[a] + self.current_state_cost[:, a][:, None]
                         + self.post_state_cost[None, :])
        return tensor

    def expected_cost_components(self, belief: np.ndarray, action, transition_matrix: np.ndarray) -> Dict[str, float]:
        """Decompose Q(b, a) into its three interpretable parts."""
        a = action_index(action)
        b = np.asarray(belief, dtype=float)
        predicted = b @ transition_matrix
        direct = float(self.action_cost[a])
        current = float(b @ self.current_state_cost[:, a])
        post = float(predicted @ self.post_state_cost)
        return {"direct": direct, "current_state": current, "post_transition": post,
                "total": direct + current + post, "predicted_belief": predicted}

    def one_step_cost_matrix(self, transition_tensor: np.ndarray) -> np.ndarray:
        """M[s, a] = sum_s' T(s'|s,a) C(s,a,s'), so that Q(b, a) = (b @ M)[a].

        Q is LINEAR in the belief; this is why each action is optimal on a
        convex region of the belief simplex.
        """
        return np.einsum("ast,ast->sa", np.asarray(transition_tensor, dtype=float), self.cost_tensor())

    def expected_cost_bellman(self, belief: np.ndarray, action, transition_matrix: np.ndarray) -> float:
        """Reference implementation: sum_s b(s) sum_s' T(s'|s,a) C(s,a,s')."""
        a = action_index(action)
        b = np.asarray(belief, dtype=float)
        return float(np.einsum("s,st,st->", b, transition_matrix, self.cost_tensor()[a]))
