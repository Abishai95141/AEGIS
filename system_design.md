# AEGIS Causal Evaluation Framework — System Design Document

Date: 2026-03-10
Audience: A fresh coding agent with no knowledge of this project's development history.
Purpose: Everything needed to implement and run the evaluation experiment. Nothing else.

---

## Section 1: The Research Question

**One sentence:** Can individualized causal treatment effects be correctly identified from sparse N-of-1 longitudinal data when unmeasured time-varying confounders are present, by using text-derived proxy variables to satisfy the identification conditions for proximal causal inference?

**Formal version:** Let Y_{i,t+k} be the glucose outcome for patient i at time t+k, A_{i,t} be the binary treatment decision at time t, S_{i,t} be the observed state (glucose, insulin, carbs), and U_{i,t} be an unmeasured time-varying confounder (e.g., psychological stress) that affects both Y and A. Let Z_{i,t} and W_{i,t} be proxy variables derived from patient-generated text, satisfying:

- Z ⊥ Y | U, S, A (treatment-confounder proxy: caused by U, does not directly cause Y)
- W ⊥ A | U, S (outcome-confounder proxy: caused by U, not caused by A)

Define the individual treatment effect:

τ_i(t) = E[Y_{i,t+k}(a=1) - Y_{i,t+k}(a=0) | S_{i,t}]

The question is whether the proximal G-estimator using Z and W recovers τ_i(t) with significantly lower error than estimators that (a) ignore confounding entirely, (b) adjust for confounding but ignore individuality, or (c) individualize but ignore unmeasured confounding. The evaluation uses a simulator with known parameters so that the true τ_i can be computed by finite differencing, providing ground truth for comparison.

Every component in this system must serve this question. If a component does not contribute to answering whether proximal causal inference with text-derived proxies recovers individualized treatment effects better than alternatives, it does not belong.

---

## Section 2: The Experiment That Answers It

### Data Generating Process

The Hovorka patient simulator (`simulator/patient.py`) generates individual patient trajectories. Each patient has unique physiological parameters drawn from clinically calibrated distributions (±20% inter-patient variability on 15 Hovorka ODE parameters). The simulator runs at 5-minute intervals.

**Unmeasured confounders** are generated as latent time-varying variables within the simulator:

- `stress_level`: AR(1) process with circadian pattern (higher 9-17h, lower at night), acute event probability 5%/hr, range [0, 1]
- `exercise_state`: Decays at 0.98/step, jumps during exercise events, range [0, 1]
- `sleep_quality`: Derived from fatigue_level (0.85·prev + 0.1·stress + noise), range [0, 1]

These latent variables affect glucose through the ODE (stress adds 2% of Q1 to dQ1; exercise subtracts 0.5% of Q1 from dQ1) and are correlated with treatment decisions (stress reduces bolus adherence by up to 30%). They are hidden from all estimators during fitting.

**Treatment assignment** follows a micro-randomized trial design. At each meal-triggered decision point, treatment (bolus insulin) is assigned with known probability p_t ∈ [0.3, 0.7], where p_t depends on the current observed state S_t but NOT on the unmeasured confounder U_t. This known randomization probability is the identifying condition for G-estimation. Implementation: a 3-line function that computes p_t = clip(0.3 + 0.4 · sigmoid(glucose - 150), 0.3, 0.7) and draws A_t ~ Bernoulli(p_t).

**Proxy variables** are synthetic in this experiment — generated from the latent confounder values using a linear model with noise:

```
Z_t = β_Z · stress_level_t + ε_Z,    ε_Z ~ N(0, σ²_Z)
W_t = β_W · fatigue_level_t + ε_W,    ε_W ~ N(0, σ²_W)
```

Two signal strength conditions are evaluated:
- **Strong proxies:** β_Z = β_W = 0.8, σ_Z = σ_W = 0.2
- **Weak proxies:** β_Z = β_W = 0.3, σ_Z = σ_W = 0.5

### Four Estimators

All four estimators run on the same data for each patient. They differ only in what information they use.

1. **Naive OLS regression.** Regresses ΔG_{t+k} = Y_{t+k} - Y_t on insulin dose A_t, controlling for S_t = (glucose_t, carbs_t). Ignores unmeasured confounding. Ignores individuality (pools all patients). This is the baseline that everything must beat.

2. **Population AIPW estimator.** Uses the doubly-robust AIPW estimator from `core/layer3_causal.py` → `AIPWEstimator`. Handles confounding via propensity weighting (propensity is known from the micro-randomization design). Ignores individuality — estimates a single population ATE across all patients.

3. **Standard G-estimator without proxy adjustment.** Uses `core/layer3_causal.py` → `HarmonicGEstimator` fitted per-patient with known propensity. Handles individuality (one estimate per patient). Does NOT adjust for unmeasured confounding — the omitted variable U biases the estimate.

4. **Proximal G-estimator with text-derived proxy variables.** Uses `core/layer3_causal.py` → `ProximalGEstimator` fitted per-patient with known propensity and proxy variables Z, W. Handles both individuality and unmeasured confounding via the bridge function h*(W). This is the method being evaluated.

### Ground Truth Comparison

For each patient i and each estimator, compute:
- **Absolute error:** |τ̂_i - τ_i_true| where τ_i_true is computed by finite differencing on the simulator (see Section 5)
- **Mean Absolute Error (MAE):** averaged over N=50 patients
- **95th percentile error:** worst-case performance across patients
- **Coverage:** fraction of patients for whom the anytime-valid confidence sequence contains τ_i_true at all observed time points

The primary result is the 4-row error table (one row per estimator). The secondary result is coverage of the confidence sequences.

### Sample Size

- Each patient contributes 288 observations (24 hours × 12 observations/hour at 5-minute intervals)
- 50 patients total
- Total observations: 14,400
- This provides sufficient per-patient data for N-of-1 estimation (288 > 200, which is the validated small-sample threshold from R-L3-08) while providing enough patients (50) for reliable comparison statistics (paired tests, bootstrap CIs on MAE differences)

---

## Section 3: Directory Structure

```
causal_eval/
    dgp/
        hovorka_wrapper.py       # thin wrapper, imports simulator/patient.py
        confounder_injection.py   # generates latent stress/exercise/sleep
        ground_truth.py           # computes true τ from simulator parameters
        proxy_generator.py        # generates synthetic text proxies from latents
    estimators/
        naive_regression.py
        population_aipw.py
        standard_gestimation.py
        proximal_gestimation.py   # imports core/layer3_causal.py directly
    evaluation/
        experiment.py             # orchestrates all four estimators
        metrics.py                # MAE, coverage, error tables
        results/                  # output directory, not committed
    tests/
        test_dgp.py              # verify ground truth computation
        test_estimators.py       # verify each estimator on known simple case
        test_experiment.py       # smoke test full pipeline
```

The new system imports `core/layer3_causal.py`, `core/layer5_safety.py`, and `simulator/patient.py` directly. It does not import Layer 1, Layer 2, or Layer 4. It does not duplicate any validated code.

---

## Section 4: What Is Imported Versus What Is Reimplemented

### Imported unchanged

- **`core/layer3_causal.py`** in full. This contains `HarmonicGEstimator`, `AIPWEstimator`, `ProximalGEstimator`, and `ConfidenceSequence`. All four classes are validated by 8 passing tests (R-L3-01 through R-L3-08). No modifications needed.

- **`simulator/patient.py`** in full. This contains `HovorkaPatientSimulator` with the 10-state Hovorka ODE, latent variable generation, meal/exercise scheduling, and the interactive stepping API (`init_scenario`, `get_scenario`, `sim_step`). The `generate_dataset` method produces complete trajectories with all latent variables exposed for ground truth computation.

- **`ConfidenceSequence` from `core/layer3_causal.py`** — the anytime-valid CS machinery. It is self-contained (depends only on numpy) and can be used independently.

### Reimplemented cleanly

**Micro-randomization mechanism.** A minimal implementation that maintains known randomization probabilities within the positivity range. No MPC, no dose optimization, no pharmacokinetic modeling. The entire implementation:

```python
def micro_randomize(glucose, low=0.3, high=0.7):
    """Known treatment probability for G-estimation identification."""
    p = np.clip(0.3 + 0.4 / (1 + np.exp(-(glucose - 150) / 30)), low, high)
    return int(np.random.random() < p), p
```

The only purpose is to satisfy the G-estimation identifying condition that treatment assignment probabilities are known. The returned probability `p` is passed to the estimators as the known propensity score.

### Not included

- **Layer 1** semantic extraction — replaced by synthetic proxy generator
- **Layer 2** digital twin — not needed when the simulator IS the DGP
- **Layer 4** MPC, dose optimization, action-centered bandits, counterfactual Thompson Sampling, isotonic λ calibration — all control-layer components
- **All IOB guard implementations** — control layer
- **All ISF scaling** — control layer
- **All pharmacokinetic parameter fitting** — control layer (p1 adapter, bias correction)

---

## Section 5: The Ground Truth Computation

This is the most important technical specification in this document. The new evaluation stands or falls on the correctness of this computation.

### Definition

The true individual treatment effect τ_i(t, k) for patient i is the expected change in glucose at time t+k per unit of insulin delivered at time t, holding all other inputs constant.

### Computation

In the Hovorka simulator, this is recoverable by finite differencing:

1. Save the simulator's full state at time t: `state_t = sim.state.copy()`
2. Also save all latent variables: `stress_t, fatigue_t, exercise_t`
3. Deliver baseline dose u₀ (e.g., 0.0 units). Run the simulator forward k steps. Record glucose at t+k: `G_baseline = sim._get_glucose_mg_dl()`
4. Restore the simulator to the exact saved state: `sim.state = state_t.copy()` and restore latent variables
5. Deliver perturbed dose u₀ + δ (e.g., δ = 0.01 units). Run forward k steps with identical carb inputs and identical latent variable trajectories (use the same random seed or pre-generated noise). Record glucose: `G_perturbed = sim._get_glucose_mg_dl()`
6. Compute: `τ_i(t, k) = (G_perturbed - G_baseline) / δ`
7. Average over multiple time points t to get the patient-level τ_i

### Parameters

- δ = 0.01 units (primary), δ = 0.001 units (validation)
- k = 12 steps (60 minutes — captures insulin action peak for Hovorka model's tmaxI ≈ 55 min)
- Average over t: sample every 12th timestep (hourly) across the 288-step day to get 24 measurements of τ_i(t, k). Report both the time-averaged τ_i and the time-varying τ_i(t, k) curve.

### Critical implementation detail

The latent variable evolution (`_update_latent_variables`) uses `self.rng` which has internal state. When computing the finite difference, both the baseline and perturbed runs must see identical latent variable trajectories. The cleanest approach: pre-generate the full sequence of latent variables (stress, fatigue, exercise) for the entire day BEFORE the finite-difference computation, then replay them identically for both runs. Alternatively, save and restore the RNG state along with the ODE state.

### Validation (must pass before any estimator runs)

Implemented in `test_dgp.py`:

1. **Range test:** Compute τ_i for 50 patients. Confirm that the range of τ_i across patients is at least 0.5 mg/dL per unit. If all patients have the same τ_i, the experiment cannot distinguish individualized from population estimation.

2. **Stability test:** For 10 patients, compute τ_i with δ = 0.01 and δ = 0.001. Confirm less than 1% relative difference: |τ(δ=0.01) - τ(δ=0.001)| / |τ(δ=0.001)| < 0.01. This validates that the finite difference is in the linear regime.

3. **Sign test:** τ_i should be negative (insulin lowers glucose). Confirm τ_i < 0 for all patients.

4. **State-dependence test:** Compute τ_i at glucose = 200 mg/dL and glucose = 100 mg/dL for the same patient. Confirm they differ (the Hovorka ODE is nonlinear, so insulin sensitivity depends on glucose level).

---

## Section 6: Proxy Variable Specification

### Identification conditions

The proxy variables must satisfy two conditions for proximal causal identification:

1. **Relevance:** Z and W must be correlated with the unmeasured confounder U. Specifically, Z must predict U given S (Z ⊥̸ U | S) and W must predict U given S (W ⊥̸ U | S). Without this, the bridge function cannot recover U's effect.

2. **Exclusion:** Z must not directly affect the outcome Y given U, S, and A (Z ⊥ Y | U, S, A). W must not be caused by treatment A given U and S (W ⊥ A | U, S). Without this, the proximal adjustment introduces new bias instead of removing old bias.

### Synthetic proxy generation

In this experiment, proxies are generated directly from latent confounder values:

```python
def generate_proxies(stress_t, fatigue_t, beta, sigma, rng):
    """
    Generate synthetic proxy variables from latent confounders.
    
    Args:
        stress_t: latent stress level at time t (the unmeasured confounder U)
        fatigue_t: latent fatigue level at time t (correlated with U)
        beta: signal strength (controls relevance condition)
        sigma: noise std (controls proxy quality)
        rng: numpy RandomState for reproducibility
    
    Returns:
        Z: treatment-confounder proxy (from stress)
        W: outcome-confounder proxy (from fatigue)
    """
    Z = beta * stress_t + rng.normal(0, sigma)
    W = beta * fatigue_t + rng.normal(0, sigma)
    return Z, W
```

### Two experimental conditions

| Condition | β | σ | Signal-to-noise | Interpretation |
|-----------|---|---|-----------------|----------------|
| Strong proxies | 0.8 | 0.2 | 4.0 | Ideal: proxy is a reliable indicator of confounder |
| Weak proxies | 0.3 | 0.5 | 0.6 | Realistic: proxy contains substantial noise |

### Fixed parameter values (for reproducibility)

All runs use these exact values:

- Random seed for patient parameter generation: `seed = 42 + patient_id * 1000` (consistent with existing `generate_cohort`)
- Random seed for proxy noise: `seed = 7777 + patient_id * 100 + condition_id` where condition_id ∈ {0: strong, 1: weak}
- Finite difference δ: 0.01 units (primary), 0.001 units (validation only)
- Horizon k: 12 steps (60 minutes)
- Micro-randomization range: [0.3, 0.7]
- Number of patients: 50
- Observations per patient: 288 (one day at 5-minute intervals)
- Patient types: 15 children (BW~30kg), 15 adolescents (BW~55kg), 20 adults (BW~75kg)
- Confidence sequence α: 0.05

---

## Section 7: Success Criteria

The experiment succeeds if ALL THREE of the following conditions hold:

### Criterion 1: Proximal advantage over naive estimation

The proximal G-estimator's mean absolute error (MAE) must be at least 30% lower than naive OLS MAE, across 50 patients, under strong proxy conditions.

Formally: `(MAE_naive - MAE_proximal) / MAE_naive ≥ 0.30`

This is the minimum bar for the causal claim to be meaningful. If the proximal estimator does not substantially outperform a method that ignores confounding entirely, the additional complexity of proxy-based identification is not justified.

### Criterion 2: Confidence sequence coverage

The anytime-valid confidence sequences must achieve at least 90% empirical coverage of the true τ_i under BOTH proxy conditions (strong and weak).

Formally: For each condition c ∈ {strong, weak}: `(# patients where CS contains τ_i at all t) / 50 ≥ 0.90`

This validates the uncertainty quantification claim. If the CS fails to cover the truth, the method's interval estimates cannot be trusted for sequential decision-making.

### Criterion 3: Proxy quality sensitivity

Under weak proxy conditions, the proximal G-estimator's MAE must be higher than under strong proxy conditions.

Formally: `MAE_proximal(weak) > MAE_proximal(strong)`

This is a negative result that strengthens the paper. It demonstrates that proxy quality matters and the method is not trivially robust to proxy noise. If the method performs identically regardless of proxy quality, it is likely not actually using the proxy information, which would undermine the causal claim.

### If any criterion fails

Do not patch around a failed success criterion. If Criterion 1 fails, the confounding effect may be too weak in the DGP — increase the confounder's effect on Y or the confounder-treatment correlation. If Criterion 2 fails, the CS may be too narrow — check the variance estimation. If Criterion 3 fails, the bridge function may be ignoring the proxy — check that the KernelRidge is actually using Z and W features. In all cases, revise the system design document before resuming implementation.

---

## Section 8: What This Experiment Cannot Claim

This section must be included in any paper or report produced from this experiment.

**This experiment uses a synthetic simulator, not real patients.** The Hovorka model is a validated pharmacokinetic/pharmacodynamic model (Hovorka et al. 2004), but it is not a patient. Treatment effect heterogeneity in real populations may not match the simulator's parameter distributions. Clinical validation on real longitudinal patient data is required before any clinical claim can be made.

**The proxy variables are synthetic and perfectly satisfy the proxy independence conditions by construction.** In this experiment, Z is generated from stress_level and W from fatigue_level with no pathway that violates the exclusion restrictions. Real patient narratives may not satisfy these conditions. A patient who writes "I'm stressed and my blood sugar is high" creates a text that is correlated with both U (stress) and Y (glucose) through a pathway that may violate Z ⊥ Y | U, S. Empirical testing of proxy conditions on real text data is required before deploying this framework.

**The unmeasured confounders are simple.** Stress, exercise, and fatigue are modeled as low-dimensional scalar processes. Real unmeasured confounders include medication interactions, illness, hormonal cycles, psychological states, and social determinants that are high-dimensional, non-stationary, and interact in complex ways. The framework's performance on simple synthetic confounders does not guarantee performance on real confounders.

**The framework identifies causal effects; it does not prescribe treatment.** The output is an estimate of τ_i for research purposes. Converting a causal effect estimate into a dosing recommendation requires a separate decision-theoretic framework with safety constraints, regulatory approval, and clinical validation. This experiment evaluates the estimation quality, not the treatment quality.

**The micro-randomized trial design requires ethical approval.** In a real deployment, randomly withholding or providing insulin to a T1D patient requires IRB approval, informed consent, and safety monitoring. The randomization probabilities [0.3, 0.7] used here may not be ethically acceptable for all patients and all glucose states. The safety supervisor (Layer 5) exists to constrain randomization within ethical bounds, but its thresholds require clinical validation.

**The Hovorka simulator's inter-patient variability (±20% on each parameter) may underestimate real heterogeneity.** The 50-patient cohort samples from three patient types (child, adolescent, adult) with Gaussian parameter noise. Real T1D patients exhibit much wider variability including insulin resistance changes, beta-cell decline rates, and comorbidities not modeled here. The experiment's 50-patient sample may overestimate the framework's generalization to a real population.
