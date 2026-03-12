"""
Thin wrapper around the Hovorka patient simulator for causal evaluation.

Imports simulator/patient.py directly. Does NOT modify the simulator.
Adds:
  - State save/restore for finite-difference ground truth computation
  - Micro-randomization mechanism with known propensity (system_design.md §4)
  - Patient cohort generation with specified type distribution
"""

import numpy as np
import sys
import os
import copy

# ---------------------------------------------------------------------------
# Import the existing simulator from the verification directory.
# The simulator is NOT copied or modified — it is used as a dependency.
# ---------------------------------------------------------------------------
_VERIFICATION_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "verification",
)
sys.path.insert(0, _VERIFICATION_DIR)

from simulator.patient import HovorkaPatientSimulator

# ---------------------------------------------------------------------------
# Constants — every number has a scientific justification
# ---------------------------------------------------------------------------

# Micro-randomization bounds: positivity condition for G-estimation.
# Treatment probability is clipped to [MR_LOW, MR_HIGH] so that
# P(A=1|S) > 0 and P(A=0|S) > 0 for all states S.
# Values from system_design.md §4 and config.py L3_RANDOMIZATION_RANGE.
MR_LOW = 0.3
MR_HIGH = 0.7

# Sigmoid center and scale for the micro-randomization function.
# glucose = 150 mg/dL is the midpoint of the euglycemic range [70, 180].
# scale = 30 mg/dL gives a smooth transition across the range.
MR_GLUCOSE_CENTER = 150.0  # mg/dL
MR_GLUCOSE_SCALE = 30.0    # mg/dL

# Timestep for the simulator — matches SIM_DT_MINUTES in config.py
DT_MINUTES = 5

# Observations per day at 5-minute intervals: 24*60/5 = 288
STEPS_PER_DAY = 288

# Patient cohort composition from system_design.md §6:
# 15 children, 15 adolescents, 20 adults out of 50 total
COHORT_CHILDREN = 15
COHORT_ADOLESCENTS = 15
COHORT_ADULTS = 20

# Seed offsets for reproducibility (system_design.md §6)
PATIENT_SEED_OFFSET = 1000  # seed = 42 + patient_id * 1000


def micro_randomize(glucose, low=MR_LOW, high=MR_HIGH, rng=None):
    """
    Known treatment probability for G-estimation identification.

    Maps current glucose to a treatment probability in [low, high].
    Higher glucose → higher probability of insulin delivery.

    The returned probability p is passed to estimators as the known
    propensity score — this is the identifying condition for G-estimation.

    Args:
        glucose: current glucose in mg/dL
        low: minimum randomization probability (positivity lower bound)
        high: maximum randomization probability (positivity upper bound)
        rng: numpy RandomState for reproducibility

    Returns:
        action: 0 or 1 (binary treatment indicator)
        propensity: P(A=1|glucose), the known assignment probability
    """
    sigmoid = 1.0 / (1.0 + np.exp(-(glucose - MR_GLUCOSE_CENTER) / MR_GLUCOSE_SCALE))
    p = np.clip(low + (high - low) * sigmoid, low, high)
    if rng is None:
        draw = np.random.random()
    else:
        draw = rng.random()
    return int(draw < p), float(p)


class HovorkaWrapper:
    """
    Wraps HovorkaPatientSimulator for the causal evaluation experiment.

    Provides:
      - save_state() / restore_state() for finite-difference ground truth
      - generate_trajectory() for producing a single patient's dataset
      - Access to simulator internals (state, params, latent variables)
    """

    def __init__(self, patient_id=0, seed=42, patient_type='adult'):
        """
        Args:
            patient_id: integer patient identifier
            seed: random seed for parameter generation and simulation
            patient_type: one of 'child', 'adolescent', 'adult'
        """
        self.patient_id = patient_id
        self.patient_type = patient_type
        self.seed = seed
        self.sim = HovorkaPatientSimulator(
            patient_id=patient_id,
            seed=seed,
            patient_type=patient_type,
        )

    # ------------------------------------------------------------------
    # State save / restore — required for ground truth finite differencing
    # ------------------------------------------------------------------

    def save_state(self):
        """
        Save the complete simulator state: ODE state vector + latent variables + RNG.

        Returns a snapshot dict that can be passed to restore_state().
        The snapshot is a deep copy — modifying the simulator after saving
        does not affect the saved snapshot.
        """
        return {
            'state': self.sim.state.copy(),
            'stress_level': float(self.sim.stress_level),
            'fatigue_level': float(self.sim.fatigue_level),
            'exercise_state': float(self.sim.exercise_state),
            'rng_state': copy.deepcopy(self.sim.rng.get_state()),
        }

    def restore_state(self, snapshot):
        """
        Restore the simulator to a previously saved state.

        After calling restore_state(snapshot), the simulator is in the
        exact same state as when save_state() produced that snapshot.
        This includes the RNG state, so subsequent calls to the simulator
        will produce identical random sequences.
        """
        self.sim.state = snapshot['state'].copy()
        self.sim.stress_level = snapshot['stress_level']
        self.sim.fatigue_level = snapshot['fatigue_level']
        self.sim.exercise_state = snapshot['exercise_state']
        self.sim.rng.set_state(copy.deepcopy(snapshot['rng_state']))

    # ------------------------------------------------------------------
    # Simulation interface
    # ------------------------------------------------------------------

    def step(self, insulin_units, carb_grams=0.0):
        """
        Advance the Hovorka ODE one timestep.

        Args:
            insulin_units: total insulin for this step (Units)
            carb_grams: carbohydrate input for this step (grams)

        Returns:
            glucose_mg_dl: glucose after this step
        """
        self.sim.state = self.sim._rk4_step(
            self.sim.state, insulin_units, carb_grams, DT_MINUTES
        )
        return self.get_glucose()

    def get_glucose(self):
        """Current glucose in mg/dL."""
        return self.sim._get_glucose_mg_dl()

    def update_latent_variables(self, hour_of_day, day=0):
        """
        Update the latent confounder variables (stress, fatigue, exercise).

        This must be called at each timestep to advance the latent state.
        The latent variables affect glucose through the ODE (stress, exercise)
        and are used to generate proxy variables.
        """
        self.sim._update_latent_variables(hour_of_day, day)

    @property
    def stress_level(self):
        return self.sim.stress_level

    @property
    def fatigue_level(self):
        return self.sim.fatigue_level

    @property
    def exercise_state(self):
        return self.sim.exercise_state

    @property
    def params(self):
        """Hovorka ODE parameters (read-only access)."""
        return self.sim.params

    @property
    def BW(self):
        """Patient body weight in kg."""
        return self.sim.BW

    def init_scenario(self, n_days=1):
        """Initialize the scenario (meals, exercise) for n_days."""
        return self.sim.init_scenario(n_days=n_days, dt_min=DT_MINUTES)

    def get_scenario(self, step_idx=None):
        """Get environment state for a timestep (meals, exercise, latent vars)."""
        return self.sim.get_scenario(step_idx=step_idx)

    def generate_trajectory(self, n_steps=STEPS_PER_DAY, rng=None):
        """
        Generate a single patient trajectory with micro-randomized treatment.

        This is the main data generation function for the experiment.
        At each step:
          1. Advance latent variables (stress, fatigue, exercise)
          2. Determine carb input from the scenario
          3. Micro-randomize treatment assignment with known propensity
          4. Step the ODE forward
          5. Record all observables AND latent variables

        The latent variables (stress, fatigue, exercise) are recorded for
        ground truth computation but are HIDDEN from all estimators.

        Args:
            n_steps: number of 5-minute timesteps
            rng: numpy RandomState for micro-randomization

        Returns:
            trajectory: dict of arrays, each of length n_steps
        """
        if rng is None:
            rng = np.random.RandomState(self.seed + 9999)

        # Initialize the scenario for 1 day
        self.init_scenario(n_days=max(1, n_steps // STEPS_PER_DAY + 1))

        trajectory = {
            'glucose': np.zeros(n_steps),
            'insulin': np.zeros(n_steps),
            'carbs': np.zeros(n_steps),
            'treatment': np.zeros(n_steps, dtype=int),
            'propensity': np.zeros(n_steps),
            'hour': np.zeros(n_steps),
            'stress': np.zeros(n_steps),
            'fatigue': np.zeros(n_steps),
            'exercise': np.zeros(n_steps),
        }

        for t in range(n_steps):
            # Get scenario for this step (updates latent variables internally)
            scenario = self.get_scenario()
            glucose = scenario['glucose_mg_dl']
            carb_input = scenario['carb_input']
            hour = scenario['hour']

            # Micro-randomize treatment
            action, propensity = micro_randomize(glucose, rng=rng)

            # Compute insulin dose: if treated, deliver a meal-proportional dose
            # If no meal, a small correction bolus proportional to glucose excess
            if action == 1:
                if carb_input > 0:
                    # Insulin-to-carb ratio ~10g/U with noise
                    icr = 10.0 + 2.0 * rng.randn()
                    insulin = max(0.0, carb_input / max(icr, 3.0))
                else:
                    # Small correction if glucose > target
                    correction = max(0.0, (glucose - 120.0) / 50.0) * 0.5
                    insulin = correction
            else:
                insulin = 0.0

            # Add basal rate
            basal = scenario['basal_rate'] * (DT_MINUTES / 60.0)
            total_insulin = insulin + basal

            # Step the ODE
            new_glucose = self.sim.sim_step(total_insulin, carb_input)

            # Record
            trajectory['glucose'][t] = new_glucose
            trajectory['insulin'][t] = total_insulin
            trajectory['carbs'][t] = carb_input
            trajectory['treatment'][t] = action
            trajectory['propensity'][t] = propensity
            trajectory['hour'][t] = hour
            # Latent variables — recorded for ground truth, hidden from estimators
            trajectory['stress'][t] = self.sim.stress_level
            trajectory['fatigue'][t] = self.sim.fatigue_level
            trajectory['exercise'][t] = self.sim.exercise_state

        return trajectory


def create_cohort(n_patients=50, base_seed=42):
    """
    Create a cohort of HovorkaWrapper instances with specified type distribution.

    Distribution (system_design.md §6):
      - 15 children (BW~30kg)
      - 15 adolescents (BW~55kg)
      - 20 adults (BW~75kg)

    Each patient gets a unique seed: base_seed + patient_id * PATIENT_SEED_OFFSET

    Args:
        n_patients: total number of patients (default 50)
        base_seed: base random seed

    Returns:
        list of HovorkaWrapper instances
    """
    types = (
        ['child'] * COHORT_CHILDREN
        + ['adolescent'] * COHORT_ADOLESCENTS
        + ['adult'] * COHORT_ADULTS
    )
    # If n_patients differs from 50, extend or truncate the type list
    if n_patients > len(types):
        extra = n_patients - len(types)
        types += ['adult'] * extra
    types = types[:n_patients]

    cohort = []
    for pid in range(n_patients):
        wrapper = HovorkaWrapper(
            patient_id=pid,
            seed=base_seed + pid * PATIENT_SEED_OFFSET,
            patient_type=types[pid],
        )
        cohort.append(wrapper)
    return cohort
