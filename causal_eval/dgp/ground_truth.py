"""
Ground truth treatment effect computation via finite differencing.

The true individual treatment effect τ_i(t, k) for patient i is the
expected change in glucose at time t+k per unit of insulin delivered
at time t, holding all other inputs constant.

This is computed by running the Hovorka ODE forward from the same state
twice — once with baseline dose u₀ and once with u₀ + δ — and taking the
glucose difference divided by δ.

CRITICAL IMPLEMENTATION DETAIL (system_design.md §5):
The latent variable evolution uses self.rng which has internal state.
Both baseline and perturbed runs must see identical latent variable
trajectories. We save and restore the full RNG state along with the
ODE state to guarantee this.

Validation requirements (must pass before any estimator runs):
  1. Range test: τ_i varies by ≥ 0.5 mg/dL/U across 10 patients
  2. Stability test: |τ(δ=0.01) - τ(δ=0.001)| / |τ(δ=0.001)| < 0.01
  3. Sign test: τ_i < 0 for all patients (insulin lowers glucose)
  4. State-dependence test: τ_i differs at glucose=200 vs glucose=100
"""

import numpy as np
from .hovorka_wrapper import HovorkaWrapper, DT_MINUTES, STEPS_PER_DAY

# ---------------------------------------------------------------------------
# Constants (system_design.md §5)
# ---------------------------------------------------------------------------

# Primary finite-difference step size: 0.01 Units of insulin.
# Small enough to be in the linear regime of the ODE, large enough
# to produce numerically stable differences.
DELTA_PRIMARY = 0.01  # Units

# Validation step size: 0.001 Units. Used only for stability check.
DELTA_VALIDATION = 0.001  # Units

# Forward horizon: 12 steps × 5 min = 60 minutes.
# Captures insulin action peak for Hovorka model (tmaxI ≈ 55 min).
HORIZON_STEPS = 12

# Sampling interval for time-varying τ: every 12th timestep (hourly)
# across a 288-step day → 24 measurements of τ_i(t, k).
SAMPLING_INTERVAL = 12  # steps (= 1 hour)

# Minimum range of τ_i across patients for the experiment to be valid.
# If all patients have the same τ_i, we cannot demonstrate individualization.
MIN_TAU_RANGE = 0.5  # mg/dL per Unit

# Maximum relative difference between δ=0.01 and δ=0.001 estimates.
MAX_STABILITY_RELDIFF = 0.01  # 1%


class GroundTruthComputer:
    """
    Computes the true individual treatment effect τ_i for a patient
    via finite differencing on the Hovorka simulator.
    """

    def __init__(self, delta=DELTA_PRIMARY, horizon_steps=HORIZON_STEPS):
        """
        Args:
            delta: finite-difference step size in insulin Units
            horizon_steps: number of forward steps (k) to measure glucose change
        """
        self.delta = delta
        self.horizon_steps = horizon_steps

    def compute_tau_at_time(self, wrapper, carb_schedule=None):
        """
        Compute τ_i(t, k) = (G_perturbed - G_baseline) / δ at the current
        simulator state.

        The procedure:
          1. Save full state (ODE + latent + RNG)
          2. Run forward k steps with dose u₀ = 0 → G_baseline
          3. Restore to saved state
          4. Run forward k steps with dose u₀ + δ → G_perturbed
          5. τ = (G_perturbed - G_baseline) / δ

        Carb inputs are identical in both runs (held constant).
        Latent variable trajectories are identical because RNG state is restored.

        Args:
            wrapper: HovorkaWrapper instance at the desired state
            carb_schedule: list of carb inputs (grams) for the k forward steps.
                          If None, uses zeros (no meal during measurement).

        Returns:
            tau: treatment effect in mg/dL per Unit of insulin
        """
        if carb_schedule is None:
            carb_schedule = [0.0] * self.horizon_steps

        assert len(carb_schedule) >= self.horizon_steps, (
            f"carb_schedule must have at least {self.horizon_steps} entries"
        )

        # --- Baseline run: dose = 0 ---
        snapshot = wrapper.save_state()
        for step in range(self.horizon_steps):
            # Update latent variables to advance the RNG consistently
            hour = (step * DT_MINUTES / 60.0) % 24.0
            wrapper.update_latent_variables(hour)
            wrapper.step(insulin_units=0.0, carb_grams=carb_schedule[step])
        g_baseline = wrapper.get_glucose()

        # --- Perturbed run: dose = δ (delivered at the first step only) ---
        wrapper.restore_state(snapshot)
        for step in range(self.horizon_steps):
            hour = (step * DT_MINUTES / 60.0) % 24.0
            wrapper.update_latent_variables(hour)
            # Deliver the perturbation dose only at the first step
            dose = self.delta if step == 0 else 0.0
            wrapper.step(insulin_units=dose, carb_grams=carb_schedule[step])
        g_perturbed = wrapper.get_glucose()

        # --- Restore to original state for subsequent calls ---
        wrapper.restore_state(snapshot)

        tau = (g_perturbed - g_baseline) / self.delta
        return tau

    def compute_patient_tau(self, wrapper, n_steps=STEPS_PER_DAY,
                            sampling_interval=SAMPLING_INTERVAL):
        """
        Compute τ_i for a patient: the time-averaged treatment effect
        across a simulated day.

        Runs the simulator forward, sampling τ at regular intervals.
        At each sampled time point, pauses the simulation, computes τ
        by finite differencing, then continues.

        Args:
            wrapper: HovorkaWrapper instance (freshly initialized)
            n_steps: total timesteps to simulate (default: 1 day = 288)
            sampling_interval: compute τ every this many steps (default: 12 = hourly)

        Returns:
            tau_mean: time-averaged τ_i (scalar, mg/dL per Unit)
            tau_curve: array of τ values at each sampled time point
            hours: array of hours at which τ was sampled
        """
        # Initialize the scenario to generate meals/exercise
        wrapper.init_scenario(n_days=max(1, n_steps // STEPS_PER_DAY + 1))

        tau_values = []
        tau_hours = []

        for t in range(n_steps):
            # Get scenario to advance latent variables via get_scenario()
            scenario = wrapper.get_scenario()
            carb_input = scenario['carb_input']
            hour = scenario['hour']

            # At sampling points, compute tau via finite differencing
            if t > 0 and t % sampling_interval == 0:
                # Build a carb schedule for the forward horizon
                # Use zero carbs during measurement to isolate insulin effect
                carb_schedule = [0.0] * self.horizon_steps
                tau_t = self.compute_tau_at_time(wrapper, carb_schedule)
                tau_values.append(tau_t)
                tau_hours.append(hour)

            # Advance the simulation with basal insulin only (no bolus)
            # This gives us a representative trajectory to sample τ from
            basal_rate = scenario.get('basal_rate', 0.8)
            basal_insulin = basal_rate * (DT_MINUTES / 60.0)
            wrapper.sim.sim_step(basal_insulin, carb_input)

        tau_curve = np.array(tau_values)
        hours = np.array(tau_hours)
        tau_mean = np.mean(tau_curve) if len(tau_curve) > 0 else 0.0

        return tau_mean, tau_curve, hours

    def compute_cohort_tau(self, cohort):
        """
        Compute τ_i for every patient in the cohort.

        Args:
            cohort: list of HovorkaWrapper instances

        Returns:
            tau_means: array of shape (n_patients,) — time-averaged τ per patient
            tau_curves: list of arrays — τ(t) curve per patient
            hours: array — hours at which τ was sampled (same for all patients)
        """
        tau_means = []
        tau_curves = []
        hours = None

        for wrapper in cohort:
            # Re-initialize the simulator to start fresh for each patient
            wrapper.sim._init_state()
            wrapper.sim.stress_level = 0.0
            wrapper.sim.fatigue_level = 0.0
            wrapper.sim.exercise_state = 0.0

            tau_mean, tau_curve, h = self.compute_patient_tau(wrapper)
            tau_means.append(tau_mean)
            tau_curves.append(tau_curve)
            if hours is None:
                hours = h

        return np.array(tau_means), tau_curves, hours


def validate_ground_truth(cohort_subset, verbose=True):
    """
    Run all four validation checks from system_design.md §5.

    Args:
        cohort_subset: list of at least 10 HovorkaWrapper instances
        verbose: print results

    Returns:
        results: dict with pass/fail for each validation check
    """
    assert len(cohort_subset) >= 10, "Need at least 10 patients for validation"

    results = {}

    # --- Test 1: Range test ---
    # τ_i must span at least MIN_TAU_RANGE across patients
    gt = GroundTruthComputer(delta=DELTA_PRIMARY)
    tau_means, _, _ = gt.compute_cohort_tau(cohort_subset[:10])
    tau_range = np.max(tau_means) - np.min(tau_means)
    results['range_test'] = {
        'pass': tau_range >= MIN_TAU_RANGE,
        'tau_range': float(tau_range),
        'threshold': MIN_TAU_RANGE,
        'tau_values': tau_means.tolist(),
    }
    if verbose:
        print(f"[Range test] τ range = {tau_range:.4f} mg/dL/U "
              f"(threshold ≥ {MIN_TAU_RANGE}) → "
              f"{'PASS' if results['range_test']['pass'] else 'FAIL'}")
        print(f"  τ values: {[f'{v:.4f}' for v in tau_means]}")

    # --- Test 2: Stability test ---
    # |τ(δ=0.01) - τ(δ=0.001)| / |τ(δ=0.001)| < 0.01 for 10 patients
    gt_fine = GroundTruthComputer(delta=DELTA_VALIDATION)
    stability_ok = True
    max_reldiff = 0.0

    for i, wrapper in enumerate(cohort_subset[:10]):
        # Re-initialize for primary delta
        wrapper.sim._init_state()
        wrapper.sim.stress_level = 0.0
        wrapper.sim.fatigue_level = 0.0
        wrapper.sim.exercise_state = 0.0
        wrapper.init_scenario(n_days=1)

        # Warm up the simulator to a representative state
        for _ in range(24):  # 2 hours of warm-up
            scenario = wrapper.get_scenario()
            wrapper.sim.sim_step(0.05, scenario['carb_input'])

        # Compute τ with primary δ
        tau_primary = gt.compute_tau_at_time(wrapper)

        # Restore to same state for fine δ
        snapshot = wrapper.save_state()
        tau_fine = gt_fine.compute_tau_at_time(wrapper)
        wrapper.restore_state(snapshot)

        if abs(tau_fine) > 1e-10:
            reldiff = abs(tau_primary - tau_fine) / abs(tau_fine)
        else:
            reldiff = 0.0

        max_reldiff = max(max_reldiff, reldiff)
        if reldiff > MAX_STABILITY_RELDIFF:
            stability_ok = False

    results['stability_test'] = {
        'pass': stability_ok,
        'max_relative_diff': float(max_reldiff),
        'threshold': MAX_STABILITY_RELDIFF,
    }
    if verbose:
        print(f"[Stability test] max |τ(0.01)-τ(0.001)|/|τ(0.001)| = {max_reldiff:.6f} "
              f"(threshold < {MAX_STABILITY_RELDIFF}) → "
              f"{'PASS' if stability_ok else 'FAIL'}")

    # --- Test 3: Sign test ---
    # τ_i should be negative for all patients (insulin lowers glucose)
    all_negative = all(t < 0 for t in tau_means)
    results['sign_test'] = {
        'pass': all_negative,
        'tau_values': tau_means.tolist(),
    }
    if verbose:
        print(f"[Sign test] All τ < 0? {all_negative} → "
              f"{'PASS' if all_negative else 'FAIL'}")
        if not all_negative:
            pos_idx = [i for i, t in enumerate(tau_means) if t >= 0]
            print(f"  Positive τ at patient indices: {pos_idx}")

    # --- Test 4: State-dependence test ---
    # τ at glucose=200 should differ from τ at glucose=100 for the same patient
    wrapper = cohort_subset[0]
    wrapper.sim._init_state()
    wrapper.sim.stress_level = 0.0
    wrapper.sim.fatigue_level = 0.0
    wrapper.sim.exercise_state = 0.0

    # Set glucose to ~200 mg/dL by adjusting Q1
    Vg = wrapper.params['Vg']
    BW = wrapper.BW
    wrapper.sim.state[6] = 200.0 * Vg * BW / 18.0  # Q1 for 200 mg/dL
    tau_high = gt.compute_tau_at_time(wrapper)

    # Set glucose to ~100 mg/dL
    wrapper.sim.state[6] = 100.0 * Vg * BW / 18.0  # Q1 for 100 mg/dL
    tau_low = gt.compute_tau_at_time(wrapper)

    state_dependent = abs(tau_high - tau_low) > 0.01  # non-trivial difference
    results['state_dependence_test'] = {
        'pass': state_dependent,
        'tau_at_200': float(tau_high),
        'tau_at_100': float(tau_low),
        'difference': float(abs(tau_high - tau_low)),
    }
    if verbose:
        print(f"[State-dependence test] τ(G=200) = {tau_high:.4f}, "
              f"τ(G=100) = {tau_low:.4f}, "
              f"|diff| = {abs(tau_high - tau_low):.4f} → "
              f"{'PASS' if state_dependent else 'FAIL'}")

    return results
