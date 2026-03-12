"""
Confounder injection for the causal evaluation DGP.

Generates three latent time-varying confounders that create unmeasured
confounding in the treatment-outcome relationship:

  1. stress_level — AR(1) with circadian pattern, acute events
  2. exercise_intensity — decay process with scheduled events
  3. sleep_quality — derived from fatigue (correlated with stress)

Causal pathways (system_design.md §2):
  U (stress) → Glucose: cortisol adds 2% of Q1 to dQ1
  U (stress) → Treatment: stress reduces bolus adherence by up to 30%
  Exercise → Glucose: uptake subtracts 0.5% of Q1 from dQ1
  Sleep quality → next-day insulin sensitivity (fatigue → fatigue_level)

All three confounders are HIDDEN from estimators during fitting.
They are recorded in trajectories only for ground truth computation
and proxy generation.

NOTE: The simulator (patient.py) already generates these latent variables
internally via _update_latent_variables(). This module provides an
ADDITIONAL confounder injection layer that:
  (a) controls the confounder-treatment coupling strength
  (b) provides a clean interface for proxy generation
  (c) documents the causal structure explicitly
"""

import numpy as np

# ---------------------------------------------------------------------------
# Constants — each has scientific justification
# ---------------------------------------------------------------------------

# AR(1) persistence for stress: 0.9 means stress has ~50-step half-life
# (~4 hours at 5-min steps). Matches cortisol clearance kinetics.
STRESS_AR_COEFF = 0.9

# Circadian stress baselines (system_design.md §2, causal_audit.md §4):
# Work hours (9-17): elevated cortisol from occupational demands
# Night (22-6): low cortisol during sleep
# Transition (6-9, 17-22): moderate
STRESS_CIRCADIAN_WORK = 0.3     # during 9-17h
STRESS_CIRCADIAN_NIGHT = 0.05   # during 22-6h
STRESS_CIRCADIAN_OTHER = 0.15   # transitions

# Stress innovation noise std: small enough for smooth dynamics,
# large enough for measurable variation
STRESS_NOISE_STD = 0.05

# Acute stress event probability per 5-min step: 5%/hr ÷ 12 steps/hr
STRESS_EVENT_PROB = 0.05 / 12.0

# Acute stress event magnitude
STRESS_EVENT_MAGNITUDE = 0.3

# Fatigue AR(1) coefficient: slower dynamics than stress (fatigue persists)
FATIGUE_AR_COEFF = 0.85

# Fatigue coupling to stress: fatigue accumulates during high-stress periods
FATIGUE_STRESS_COUPLING = 0.10

# Fatigue innovation noise std
FATIGUE_NOISE_STD = 0.03

# Exercise decay per step: 0.98^12 ≈ 0.78 after 1 hour (gradual effect fade)
EXERCISE_DECAY = 0.98

# Exercise event magnitude when triggered
EXERCISE_EVENT_MAGNITUDE = 0.1

# Confounder-treatment coupling: stress reduces bolus adherence by this fraction
# P(bolus | meal) = 1 - TREATMENT_COUPLING * stress_level
# At max stress=1.0, adherence drops by 30% (system_design.md §2)
TREATMENT_COUPLING = 0.3


class ConfounderInjector:
    """
    Generates and manages latent time-varying confounders for the DGP.

    The injector maintains three confounder processes and provides:
      - step(): advance all confounders one timestep
      - get_treatment_probability_modifier(): how confounders affect treatment
      - get_confounders(): current values for proxy generation
      - generate_trajectory(): full confounder trajectory for a patient
    """

    def __init__(self, seed=42):
        """
        Args:
            seed: random seed for reproducible confounder generation
        """
        self.rng = np.random.RandomState(seed)
        self.stress = 0.0
        self.fatigue = 0.0
        self.exercise = 0.0

    def reset(self):
        """Reset all confounders to initial values."""
        self.stress = 0.0
        self.fatigue = 0.0
        self.exercise = 0.0

    def step(self, hour_of_day, exercise_event=False):
        """
        Advance all confounder processes one timestep.

        Args:
            hour_of_day: float in [0, 24) for circadian pattern
            exercise_event: True if patient is exercising this step

        Returns:
            dict with current confounder values
        """
        # --- Stress: AR(1) with circadian baseline ---
        if 9 <= hour_of_day <= 17:
            circadian = STRESS_CIRCADIAN_WORK
        elif hour_of_day >= 22 or hour_of_day <= 6:
            circadian = STRESS_CIRCADIAN_NIGHT
        else:
            circadian = STRESS_CIRCADIAN_OTHER

        self.stress = (STRESS_AR_COEFF * self.stress
                       + (1.0 - STRESS_AR_COEFF) * circadian
                       + STRESS_NOISE_STD * self.rng.randn())

        # Acute stress events
        if self.rng.random() < STRESS_EVENT_PROB:
            self.stress += STRESS_EVENT_MAGNITUDE

        self.stress = np.clip(self.stress, 0.0, 1.0)

        # --- Fatigue: AR(1) coupled to stress ---
        self.fatigue = (FATIGUE_AR_COEFF * self.fatigue
                        + FATIGUE_STRESS_COUPLING * self.stress
                        + FATIGUE_NOISE_STD * self.rng.randn())
        self.fatigue = np.clip(self.fatigue, 0.0, 1.0)

        # --- Exercise: decay with event-driven jumps ---
        self.exercise *= EXERCISE_DECAY
        if exercise_event:
            self.exercise = min(self.exercise + EXERCISE_EVENT_MAGNITUDE, 1.0)

        return self.get_confounders()

    def get_confounders(self):
        """Current confounder values."""
        return {
            'stress': float(self.stress),
            'fatigue': float(self.fatigue),
            'exercise': float(self.exercise),
        }

    def get_treatment_probability_modifier(self):
        """
        How confounders modify treatment probability.

        Returns a multiplier in [1 - TREATMENT_COUPLING, 1.0] that
        reduces treatment adherence under high stress.

        This creates the U → A pathway that makes confounding.
        Higher stress → lower probability of taking insulin → confounding.
        """
        return 1.0 - TREATMENT_COUPLING * self.stress

    def generate_confounder_trajectory(self, n_steps, exercise_schedule=None):
        """
        Generate a complete confounder trajectory.

        Args:
            n_steps: number of 5-minute timesteps
            exercise_schedule: list of booleans indicating exercise at each step.
                             If None, no exercise events.

        Returns:
            dict of arrays: 'stress', 'fatigue', 'exercise', each of length n_steps
        """
        self.reset()

        if exercise_schedule is None:
            exercise_schedule = [False] * n_steps

        trajectory = {
            'stress': np.zeros(n_steps),
            'fatigue': np.zeros(n_steps),
            'exercise': np.zeros(n_steps),
        }

        for t in range(n_steps):
            hour = (t * 5.0 / 60.0) % 24.0  # 5-min steps
            ex_event = exercise_schedule[t] if t < len(exercise_schedule) else False
            vals = self.step(hour, exercise_event=ex_event)
            trajectory['stress'][t] = vals['stress']
            trajectory['fatigue'][t] = vals['fatigue']
            trajectory['exercise'][t] = vals['exercise']

        return trajectory


def compute_confounder_treatment_correlation(trajectory, treatment):
    """
    Verify that confounders are correlated with treatment decisions.

    This correlation is what creates unmeasured confounding.
    If confounders are independent of treatment, there is no
    confounding to adjust for and the experiment is trivial.

    Args:
        trajectory: dict with 'stress', 'fatigue', 'exercise' arrays
        treatment: binary treatment array

    Returns:
        dict with correlation coefficients
    """
    correlations = {}
    for name in ['stress', 'fatigue', 'exercise']:
        if name in trajectory and len(trajectory[name]) == len(treatment):
            corr = np.corrcoef(trajectory[name], treatment)[0, 1]
            correlations[name] = float(corr)
    return correlations
