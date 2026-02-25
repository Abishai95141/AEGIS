"""
AEGIS 3.0 Layer 4: Decision Engine — Real Implementation

NOT a mock. Uses:
- Action-Centered Bandits with real baseline subtraction
- ε-greedy exploration with multiplicative decay
- Real conjugate normal-normal Bayesian Thompson Sampling
- Counterfactual Thompson Sampling with precision-weighted updates
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    L4_EPSILON_INIT, L4_EPSILON_DECAY, L4_EPSILON_MIN,
    L4_PRIOR_VAR, L4_NOISE_VAR,
    L4_CF_LAMBDA_POPULATION, L4_CF_LAMBDA_DIRECT,
)


class ActionCenteredBandit:
    """
    Action-Centered Contextual Bandit.

    Decomposes reward: R_t = f(S_t) + A_t · τ(S_t) + ε_t
    Learns ONLY τ(S_t) — the treatment effect — treating f(S_t) as noise.

    This achieves variance reduction by orders of magnitude.
    """

    def __init__(self, n_arms, epsilon_init=None, epsilon_decay=None,
                 epsilon_min=None):
        self.n_arms = n_arms
        self.epsilon = epsilon_init or L4_EPSILON_INIT
        self.epsilon_decay = epsilon_decay or L4_EPSILON_DECAY
        self.epsilon_min = epsilon_min or L4_EPSILON_MIN

        self.counts = np.zeros(n_arms)
        self.sum_rewards = np.zeros(n_arms)
        self.means = np.zeros(n_arms)

        # Baseline tracking for action centering
        self.baseline_sum = 0.0
        self.baseline_count = 0
        self.baseline_mean = 0.0

        self.t = 0

    def select_arm(self):
        """ε-greedy arm selection with action centering."""
        self.t += 1

        # Force exploration for unvisited arms
        for a in range(self.n_arms):
            if self.counts[a] == 0:
                return a

        # ε-greedy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            # Select based on centered rewards
            return int(np.argmax(self.means - self.baseline_mean))

    def update(self, arm, reward, baseline=None):
        """
        Update arm statistics with optional baseline subtraction.

        When baseline is provided (the f(S_t) component), we subtract it
        to learn only the treatment effect.
        """
        self.counts[arm] += 1

        if baseline is not None:
            centered_reward = reward - baseline
            self.baseline_count += 1
            self.baseline_sum += baseline
            self.baseline_mean = self.baseline_sum / self.baseline_count
        else:
            centered_reward = reward

        self.sum_rewards[arm] += centered_reward
        self.means[arm] = self.sum_rewards[arm] / self.counts[arm]

        # Decay epsilon
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)


class CounterfactualThompsonSampling:
    """
    CTS with real conjugate normal-normal Bayesian updates.

    Key innovation: When an arm is blocked by safety, the posterior is
    still updated using Digital Twin imputed outcomes, preventing
    posterior collapse for blocked arms.

    Uses precision-weighted updates with λ-discounting.
    """

    def __init__(self, n_arms, prior_mean=0.0, prior_var=None,
                 noise_var=None):
        self.n_arms = n_arms
        self.noise_var = noise_var or L4_NOISE_VAR
        self.prior_mean = prior_mean
        self.prior_var = prior_var or L4_PRIOR_VAR

        # Bayesian posteriors (conjugate normal-normal)
        self.post_means = np.full(n_arms, prior_mean, dtype=float)
        self.post_vars = np.full(n_arms, self.prior_var, dtype=float)

        # Observation counts
        self.counts = np.zeros(n_arms)
        self.sum_rewards = np.zeros(n_arms)
        self.cf_counts = np.zeros(n_arms)

        # Digital Twin knowledge: global statistics
        self.global_sum = 0.0
        self.global_count = 0
        self.global_mean = prior_mean

    def select_arm(self):
        """Thompson Sampling: sample from posterior, pick argmax."""
        samples = np.array([
            np.random.normal(self.post_means[a],
                             np.sqrt(max(self.post_vars[a], 1e-10)))
            for a in range(self.n_arms)
        ])
        return int(np.argmax(samples))

    def update(self, arm, reward):
        """
        Standard Bayesian update for observed arm.

        Conjugate normal-normal: prior N(μ₀, σ₀²) + obs ~ N(μ, σ²)
        → posterior N(μ_post, σ_post²)
        """
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward

        # Update global statistics (Digital Twin knowledge)
        self.global_count += 1
        self.global_sum += reward
        self.global_mean = self.global_sum / self.global_count

        # Conjugate update
        n = self.counts[arm]
        prior_prec = 1.0 / self.prior_var
        obs_prec = n / self.noise_var
        post_prec = prior_prec + obs_prec

        self.post_means[arm] = (
            prior_prec * self.prior_mean +
            self.sum_rewards[arm] / self.noise_var
        ) / post_prec
        self.post_vars[arm] = 1.0 / post_prec

    def counterfactual_update(self, blocked_arm):
        """
        CTS counterfactual update for a safety-blocked arm.

        When arm is blocked:
        1. Impute outcome from best available estimate (Digital Twin)
        2. Apply precision update with λ weight (discounted for uncertainty)
        3. Accumulates over many blocked rounds to reduce posterior variance
        """
        self.cf_counts[blocked_arm] += 1

        # Choose imputation source and confidence
        if self.counts[blocked_arm] > 0:
            # We have direct observations — use our posterior mean
            imputed_outcome = self.post_means[blocked_arm]
            lambda_weight = L4_CF_LAMBDA_DIRECT
        else:
            # No direct observations — use Digital Twin (population mean)
            imputed_outcome = (self.global_mean if self.global_count > 0
                               else self.prior_mean)
            lambda_weight = L4_CF_LAMBDA_POPULATION

        # Precision-weighted Bayesian update (discounted likelihood)
        current_prec = 1.0 / self.post_vars[blocked_arm]
        virtual_prec = lambda_weight / self.noise_var
        new_prec = current_prec + virtual_prec

        # Mean update (weighted by precision)
        new_mean = (
            current_prec * self.post_means[blocked_arm] +
            virtual_prec * imputed_outcome
        ) / new_prec

        self.post_means[blocked_arm] = new_mean
        self.post_vars[blocked_arm] = 1.0 / new_prec

    def predict_counterfactual(self, arm):
        """
        Predict counterfactual outcome for an arm.

        Returns: (mean, ci_lower, ci_upper)
        """
        mean = self.post_means[arm]
        pred_var = self.post_vars[arm] + self.noise_var
        ci_lower = mean - 1.96 * np.sqrt(pred_var)
        ci_upper = mean + 1.96 * np.sqrt(pred_var)
        return mean, ci_lower, ci_upper


class DecisionEngine:
    """
    Full Layer 4 implementation combining ACB and CTS.

    Interface for the AEGIS pipeline.
    """

    def __init__(self, actions=None):
        self.actions = actions or [0, 0.5, 1.0, 2.0]
        self.n_arms = len(self.actions)

        self.acb = ActionCenteredBandit(self.n_arms)
        self.cts = CounterfactualThompsonSampling(self.n_arms)

        self.decision_history = []

    def select_action(self, glucose, effect_estimates=None):
        """
        Select treatment action.

        Uses CTS (Thompson Sampling) for selection,
        ACB for variance reduction in reward learning.

        Returns: (action_idx, info_dict)
        """
        # Safety overrides (before bandit logic)
        if glucose < 80:
            return 0, {'action': 0, 'reason': 'Low glucose — no correction'}
        elif glucose > 250:
            return self.n_arms - 1, {
                'action': self.actions[-1],
                'reason': 'High glucose — max correction'
            }

        # Thompson Sampling selection
        arm = self.cts.select_arm()

        # Also use ε-greedy for exploration
        acb_arm = self.acb.select_arm()

        # Use CTS selection but track ACB for variance analysis
        info = {
            'action': self.actions[arm],
            'cts_arm': arm,
            'acb_arm': acb_arm,
            'epsilon': self.acb.epsilon,
            'post_means': self.cts.post_means.copy(),
            'post_vars': self.cts.post_vars.copy(),
        }

        self.decision_history.append(info)
        return arm, info

    def update(self, action_idx, reward, baseline=None):
        """Update both ACB and CTS with observed reward."""
        self.acb.update(action_idx, reward, baseline)
        self.cts.update(action_idx, reward)

    def counterfactual_update(self, action_idx):
        """Apply CTS counterfactual update for blocked arm."""
        self.cts.counterfactual_update(action_idx)
