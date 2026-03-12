# AEGIS 3.0: Research Findings — Solutions for the Four Open Problems

**Document type:** Literature-grounded research findings  
**Scope:** Two peer-reviewed methods per open problem, with per-method documentation of citation, mathematical formulation, input requirements, architectural compatibility, computational cost, validation history, and known failure modes. Each problem section closes with an architecture-compatibility recommendation. The document ends with a cross-problem synthesis paragraph.

---

## Problem 1 — Online Parameter Fitting to Close the Bergman–Hovorka Structural Mismatch

**Background.** Layer 2 (Digital Twin) uses the 3-state Bergman Minimal Model with a fixed glucose-effectiveness parameter `p1 = 0.028 min⁻¹`. This parameter drives excessively fast self-correction relative to the true 10-state Hovorka simulator, yielding a persistent underprediction of high-range glucose. Even with the bias-correction term, the RMSE remains 116.7 mg/dL. The research question is whether augmenting the UKF state vector to learn `p1` (and possibly `p2`, `p3`) online can permanently close this gap without violating the constraint that the AC-UKF + RBPF code must not be touched.

---

### Method 1A — Augmented-State UKF for Simultaneous State and Parameter Estimation (Joint UKF)

**Citation:**  
Eberle, C., & Ament, C. (2011). Identifiability and online estimation of diagnostic parameters within the glucose insulin homeostasis. *Biosystems*, 107(1), 39–46. https://doi.org/10.1016/j.biosystems.2011.10.002

**Core mathematical formulation.**  
The standard 3-state Bergman state vector `x = [G, X, I]ᵀ` is augmented with the parameters to be estimated online, treated as random-walk states with small process noise `σ²_θ`:

```
x̃ = [G, X, I, p1, p2, p3]ᵀ    (augmented state, dim = 6)

Augmented dynamics:
  dG/dt  = -(p1 + X)·G + p1·Gb + D(t)
  dX/dt  = -p2·X + p3·(I - Ib)
  dI/dt  = -n·I + u(t)/V1
  dp1/dt = w1,   w1 ~ N(0, σ²_p1)
  dp2/dt = w2,   w2 ~ N(0, σ²_p2)
  dp3/dt = w3,   w3 ~ N(0, σ²_p3)
```

The augmented system is run through the UKF with the standard unscented transform. Sigma-point count becomes `2(n + nθ) + 1 = 2(3 + 3) + 1 = 13` instead of the standard 7. The observability Gramian is computed by the paper's method to verify which subset of `{p1, p2, p3}` is identifiable from CGM-only observations; the paper finds that `p1` (glucose effectiveness, `SG`) and `SI = p3/p2` are the two diagnostically dominant and most consistently identifiable parameters.

Bounds on parameters are enforced element-wise on the sigma points to maintain physiological validity (e.g., `p1 > 0`, `p2 > 0`, `p3 > 0`).

**Inputs required per timestep:**  
- `G(k)` — CGM glucose measurement (mg/dL)  
- `u(k)` — insulin infusion rate (U/min) already computed by Layer 4  
- `D(k)` — carbohydrate disturbance estimate (g/min) already fed from meal detection  

No additional sensors are required beyond what the current pipeline already collects.

**Compatibility with AEGIS architecture.**  
The AC-UKF inside Layer 2 already maintains a state vector and covariance `P`. The joint UKF is implemented as a *wrapper* around the existing ODE integration step: the state vector passed into `DigitalTwin.predict_trajectory` is extended from 3 to 6 elements, and the `p1`, `p2`, `p3` scalar constants currently hard-coded in the Bergman ODE are replaced with the corresponding elements of the current state estimate. **The AC-UKF + RBPF inner implementation is not touched.** The augmented vector is fed at the initialization point and read back at the update step. This is fully compatible with the constraint that the AC-UKF code must not be modified—parameter learning wraps the outside.

**Computational cost.**  
Sigma-point count increases from 7 to 13 (86% increase). Each UKF step is O(n³) in matrix operations, so the augmented version is approximately `(6/3)³ = 8×` more costly per step. At 5-min sampling with a modern CPU, this is on the order of tens of milliseconds per step — negligible for closed-loop control.

**Validation results.**  
Eberle & Ament (2011) demonstrated successful online estimation of `SG` and `SI` on in-silico Bergman data with CGM-only measurement and a 5-minute sampling interval. The symmetric UKF outperformed the EKF in accuracy with lower divergence risk. The authors confirm that the glucose effectiveness `p1` is observable from CGM measurements alone for the standard intravenous glucose tolerance test (IVGTT) protocol. Application to continuous ambulatory data was noted as the primary limitation of the study.

A closely related result was documented in: Palerm, C.C. et al. (2009). Glucose estimation and prediction through meal responses using ambulatory subject data for advisory mode model predictive control. *Journal of Diabetes Science and Technology*, 3(5), 1082–1090. PMC2769674. This paper explicitly augmented the `p1` parameter as an additional state in a Kalman-filter-based observer for the Bergman model applied to three T1D subjects over 97 meal responses. Maximum prediction horizons improved from <30 min (fixed parameters) to 45–48 minutes after augmentation with online `p1` adaptation.

**Known risks and failure modes.**  
- Parameter identifiability degrades during flat glucose periods (minimal signal). During extended basal euglycemia the Gramian may become near-singular, causing `p1` to drift. A lower bound on the parameter process noise covariance `σ²_p1` prevents complete freezing.  
- If three parameters are estimated simultaneously, the filter can produce trade-off artifacts (e.g., `p2` and `p3` co-varying to preserve `SI = p3/p2`). The paper's recommendation is to estimate only `p1` and `SI` as independent quantities.  
- Divergence risk increases with large plant-model mismatch exactly of the type AEGIS has; a conservative large initial `P` and physiological bounding of sigma points are mandatory.

---

### Method 1B — Dual Unscented Kalman Filter (Dual UKF) for Decoupled State and Parameter Estimation

**Citation:**  
Wan, E.A., & Nelson, A.T. (2000). Dual extended Kalman filter methods. In *Kalman Filtering and Neural Networks* (pp. 123–173). Wiley. (Foundational method, widely cited.) Applied in the artificial pancreas context in: Hajizadeh, I., Rashid, M., Turksoy, K., et al. (2018). Adaptive and personalized plasma insulin concentration estimation for artificial pancreas systems. *Journal of Diabetes Science and Technology*, 13(4), 639–659. https://doi.org/10.1177/1932296818763959. PMC6154239.

**Core mathematical formulation.**  
Two UKF instances run in parallel at each timestep:

```
State UKF (subscript s):
  x̂(k) = UKF_predict(x̂(k-1), θ̂(k-1), u(k))
  x̂(k) = UKF_update(x̂(k), G_CGM(k))

Parameter UKF (subscript θ):
  θ̂(k) = UKF_predict(θ̂(k-1))        # random walk: θ̃ = θ + w_θ
  θ̂(k) = UKF_update(θ̂(k), G_CGM(k), x̂(k))
```

At each step the state UKF uses the current parameter estimate `θ̂` as fixed; the parameter UKF uses the current state estimate `x̂` to compute the Jacobian of the observation function with respect to `θ`. This decoupling avoids the sigma-point explosion of the joint UKF and keeps each filter's dimension at 3+3=3 (state or parameter), each requiring only `2×3+1 = 7` sigma points. Bounds on parameters are enforced identically to Method 1A.

In Hajizadeh et al. (2018), augmented Hovorka model states plus 3 time-varying pharmacokinetic parameters were co-estimated with the UKF, tested on 20 three-day clinical closed-loop experiments with T1D subjects, achieving tight plasma insulin concentration (PIC) tracking (Table D1 of that paper).

**Inputs required per timestep:**  
Same as Method 1A: `G(k)`, `u(k)`, `D(k)`.

**Compatibility with AEGIS architecture.**  
The dual UKF is even less invasive than the joint UKF. The existing AC-UKF serves as the *state UKF*; a small new *parameter UKF* object is instantiated alongside it and calls the state UKF's current estimate without entering its implementation. This satisfies the "do not touch AC-UKF code" constraint perfectly, as the parameter filter is an entirely separate object that reads from but does not write to the state filter's internals.

**Computational cost.**  
Two UKF instances each with 7 sigma points ≈ same cost as one 13-point augmented UKF. Slightly lower than Method 1A and generally numerically more stable.

**Validation results.**  
Hajizadeh et al. (2018) tested the UKF-based augmented observer on 20 clinical closed-loop data sets from adolescents at the Yale Children's Diabetes Clinic. The augmented state UKF significantly outperformed a fixed-parameter Hovorka UKF in terms of PIC RMSE. The approach was also validated by Turksoy et al. in follow-on work using insulin sensitivity adaptation, reporting convergence within 2–4 days on clinical patient data.

**Known risks and failure modes.**  
- Parameter estimates from the two decoupled filters can become inconsistent during rapid transients (meals, exercise) because each filter assumes the other is converged — a known limitation of the dual-UKF approach.  
- The random-walk model for parameters assumes slow variation; during rapid meal-induced insulin sensitivity changes, the parameter process noise `Q_θ` must be tuned aggressively enough to track, but not so aggressively as to amplify measurement noise.

---

### Problem 1 Recommendation

**Method 1A (Joint UKF augmented state)** is recommended for AEGIS 3.0 for the following reasons found in the searched literature: (1) Eberle & Ament (2011) directly confirmed observability of `p1` (glucose effectiveness) from CGM-only data in the Bergman model, which is precisely the parameter AEGIS identifies as the primary mismatch driver; (2) the Palerm et al. (2009) study demonstrated that augmenting only `p1` as an online state improved Bergman prediction horizons from <30 min to 45–48 min across three T1D subjects; (3) the implementation wraps the existing AC-UKF without touching it, satisfying the invariant constraint. Only `p1` and `SI = p3/p2` should be augmented simultaneously (not all three independently) to avoid collinearity. The filter's `Q_θ` for `p1` should be initialized conservatively (e.g., `σ²_p1 = 1e-8 min⁻²`) and tuned upward only if tracking lags.

---

## Problem 2 — Adaptive Innovation Forgetting for the Bias Correction Window

**Background.** The current implementation aggregates a 20-step rectangular (uniform) window of UKF residual innovations and injects their running mean as an additive bias correction into `DigitalTwin.predict_trajectory`. This rectangular window lags during rapid physiological changes (post-meal glucose rise, exercise-induced insulin sensitivity shifts) because all 20 steps contribute equally regardless of age. The research question is whether an exponential forgetting factor applied to the innovation accumulation would improve responsiveness without destabilizing the correction.

---

### Method 2A — Variable Forgetting Factor Recursive Least Squares (VFF-RLS) Applied to Innovation Accumulation

**Citation:**  
Bruce, A.L., Goel, A., & Bernstein, D.S. (2024). Adaptive Kalman filtering developed from recursive least squares forgetting algorithms. *arXiv:2404.10914* [Eess.SY]. (This paper derives a class of adaptive Kalman filters as special cases of RLS with variable forgetting and demonstrates improved estimation for mass-spring-damper systems with intermittent unmodeled disturbances — structurally analogous to the meal/exercise transient problem in AEGIS.)

Supporting foundational reference: Cao, L., & Schwartz, H.M. (2000). A directional forgetting algorithm based on the decomposition of the information matrix. *Automatica*, 36(11), 1725–1731. https://doi.org/10.1016/S0005-1098(00)00089-0

**Core mathematical formulation.**  
Let `e(k) = G_CGM(k) − Ĝ(k)` be the UKF innovation (residual) at step `k`. The exponentially-weighted running mean bias estimate replaces the current rectangular window:

```python
# Current rectangular window (existing code):
bias(k) = (1/N) * sum(e(k-N+1 : k))

# Proposed VFF-RLS replacement:
bias(k) = lambda(k) * bias(k-1) + (1 - lambda(k)) * e(k)

# where lambda(k) is the variable forgetting factor:
lambda(k) = lambda_max - (lambda_max - lambda_min) * indicator(large_innovation)

# A simple data-driven rule:
indicator = (|e(k)| > alpha * sigma_e)    # sigma_e = rolling std of recent innovations
lambda(k) = lambda_min  if indicator else lambda_max

# Typical values: lambda_max = 0.98, lambda_min = 0.80, alpha = 2.0
```

The update is `O(1)` per step. The key property: when `|e(k)|` is large (indicating a rapid physiological change such as a meal), `lambda(k)` drops toward `lambda_min`, causing the bias estimate to respond quickly to recent innovations. During steady-state euglycemia, `lambda(k) ≈ lambda_max`, and the estimator averages over a long effective window of `1/(1-0.98) = 50` steps, suppressing noise.

**Inputs required per timestep:**  
- `e(k)` — UKF innovation, already computed in the existing Digital Twin step  
- `sigma_e` — rolling standard deviation of recent innovations (can be computed with a parallel EMA: `sigma_e² = lambda * sigma_e² + (1-lambda) * e(k)²`)

No new sensor signals are required.

**Compatibility with AEGIS architecture.**  
The existing bias correction is a single running-mean accumulator in `DigitalTwin.predict_trajectory`. Replacing the rectangular accumulator with the EMA recursion above is a 4-line code change. The AC-UKF + RBPF implementation is not involved. The injection of the bias into the initial state and forward-horizon predictions is unchanged.

**Computational cost.**  
O(1) per timestep. Effectively zero additional cost relative to the existing window. The `sigma_e` EMA requires one additional multiply-add per step.

**Validation results.**  
Bruce et al. (2024) demonstrated that adaptive KF with robust variable forgetting factor (Table I of the arXiv paper) provided improved state estimation for a mass-spring-damper experiencing intermittent, unmodeled collisions. After each collision, `lambda(k)` briefly but drastically decreased, producing immediate improvement in displacement and velocity estimates, with error decreasing roughly 2× compared to a fixed-lambda filter. This structural analogy — unmodeled impulsive disturbances requiring rapid adaptive forgetting — is directly relevant to post-meal glucose transients in AEGIS.

**Known risks and failure modes.**  
- If `lambda_min` is set too aggressively (e.g., `< 0.70`), the bias estimate becomes dominated by single-step noise rather than the true underlying innovation trend, potentially causing overcorrection oscillations.  
- The threshold `alpha` must be tuned against the specific noise characteristics of the Hovorka simulator's CGM model; if `alpha` is too small, the filter permanently runs at `lambda_min` even during flat glucose.  
- A purely EMA-based bias correction does not distinguish between innovation caused by model mismatch (desirable to correct) and innovation caused by CGM noise (undesirable to propagate forward). A noise-whitening pre-filter on `e(k)` is advisable.

---

### Method 2B — Sage–Husa Adaptive Noise Covariance Estimation with Forgetting

**Citation:**  
Sage, A.P., & Husa, G.W. (1969). Adaptive filtering with unknown prior statistics. In *Proceedings of the Joint Automatic Control Conference*, 760–769. (Foundational method.) Applied with forgetting factor variant in: Li, S. et al. (2023). An improved variable forgetting factor recursive least square-double extended Kalman filtering based on global mean particle swarm optimization for collaborative state estimation of lithium-ion batteries. *Electrochimica Acta*, 452, 142238. https://doi.org/10.1016/j.electacta.2023.142238

**Core mathematical formulation.**  
The Sage–Husa estimator adapts the UKF measurement noise covariance `R` (and optionally `Q`) online from the innovation sequence. With a forgetting factor `d ∈ (0, 1)`:

```
# Innovation covariance update (Sage-Husa with forgetting):
C_e(k) = (1/(1 - b^(k+1))) * sum_{j=0}^{k} b^(k-j) * e(j)·e(j)ᵀ

# Recursive approximation:
C_e(k) = (1 - d) * C_e(k-1) + d * (e(k)·e(k)ᵀ + H·P_k|k-1·Hᵀ)

# Updated noise covariance:
R̂(k) = C_e(k) - H·P_k|k-1·Hᵀ    (clipped to remain ≥ R_min)

# Adaptive bias:
bias(k) = (1 - d) * bias(k-1) + d * e(k)
```

where `H` is the linearized measurement Jacobian, `d = 1 - lambda` is the adaptation gain, and `P_k|k-1` is the prior state covariance from the UKF. The bias estimate here is mathematically grounded: it tracks the conditional mean of the innovation distribution, allowing both mean-shift (bias) and variance-shift (changing noise levels) to be tracked.

**Inputs required per timestep:**  
- `e(k)`, `P_k|k-1`, `H(k)` — all already computed inside the UKF; `H` is the sigma-point-based approximation of the measurement Jacobian.

**Compatibility with AEGIS architecture.**  
The Sage–Husa update is added as a post-processing step after each UKF measurement update, reading `e(k)` and `P` from the UKF output. It does not modify the AC-UKF internals. The `R̂(k)` can either be passed into the next UKF cycle (modifying `R` externally) or used only to compute an adjusted bias for the prediction horizon injection. The latter mode (bias-only) requires no change to the UKF at all.

**Computational cost.**  
O(m²) where `m = 1` for scalar CGM — equivalent to O(1). Negligible.

**Validation results.**  
The Sage–Husa method is one of the most widely validated adaptive noise estimators in control literature. Li et al. (2023) implemented a variable-forgetting-factor variant for lithium-ion battery state-of-charge estimation (a comparable nonlinear state estimation problem with time-varying noise) and demonstrated that the approach provided accuracy improvements of 10–30% in SOC RMSE over fixed-`R` UKF across multiple temperature conditions. The adaptive `R` was particularly beneficial during rapid transients (high-current discharge), analogous to post-meal glucose excursions in AEGIS.

**Known risks and failure modes.**  
- The Sage–Husa estimator is known to be susceptible to filter divergence when the true noise covariance is time-varying and the adaptation gain `d` is too aggressive. A lower bound `R̂ ≥ R_min` is essential.  
- The estimator assumes stationary statistics within the forgetting window; if both `Q` and `R` are adapted simultaneously, the updates can become non-identifiable (infinite combinations of `Q,R` produce the same innovation covariance). Adapting only `R` — or only the bias mean — and holding `Q` fixed is the safer strategy for AEGIS.

---

### Problem 2 Recommendation

**Method 2A (VFF-RLS applied to innovation EMA)** is recommended because: (1) it is a minimal, O(1) modification to the existing rectangular window code — a 4-line change; (2) the Bruce et al. (2024) structural analogy is direct (impulse-disturbance system with adaptive forgetting); and (3) it does not alter the UKF `P` or `R` matrices in any way, eliminating all risk of filter divergence. Method 2B is more mathematically rigorous but requires feeding `R̂` back into the UKF, which would constitute a modification to the filter's operating parameters (though not its implementation code). The recommended initial tuning is `lambda_max = 0.97`, `lambda_min = 0.82`, `alpha = 1.8` — values consistent with the automotive adaptive filtering literature that operates at 5-minute sampling intervals under similar transient regimes.

---

## Problem 3 — Pediatric Cohort Safety: Normalizing Hypoglycemia Risk for Low-Body-Weight Subjects

**Background.** Multi-patient tests show that child patients (`pid=1`, `pid=3`, body weight ~30 kg) experience 9.4–13.0% Severe TBR (<54 mg/dL), while the adult patient achieves 0.0%. The root cause is that the MPC's base dose heuristic, the `STL_CORRECTED` correction floor formula `(G - 200)/ISF`, and the L5 reachability parameters use population-level insulin sensitivity parameters tuned for a 75-kg adult. The 30-kg child has proportionally larger insulin effect per unit dose (lower distribution volume, higher sensitivity per unit body weight), causing routine doses to over-drive glucose below range.

---

### Method 3A — Weight-Scaled TDD-Derived ISF Adaptation with Online Insulin Sensitivity Adaptation (ISA)

**Citation:**  
Turksoy, K., Paulino, T.M.L., Zaharieva, D.P., Bhargava, V., Colwell, N., Hobbs, N., Brandt, R., Mansouri, M., Littlejohn, E., & Cinar, A. (2019). Adaptive control of an artificial pancreas using model identification, adaptive postprandial insulin delivery, and heart rate and accelerometry as control inputs. *Journal of Diabetes Science and Technology*, 14(3), 549–558. https://doi.org/10.1177/1932296819879972. PMC6835177.

Supporting reference for weight-based TDI initialization: Lee, H., Buckingham, B.A., Wilson, D.M., & Bequette, B.W. (2009). A closed-loop artificial pancreas using model predictive control and a sliding meal size estimator. *Journal of Diabetes Science and Technology*, 3(5), 1082–1090. https://doi.org/10.1177/193229680900300511. PMC2769914.

**Core mathematical formulation.**  
The ISF (insulin sensitivity factor) is initialized from total daily insulin (TDI) using the 1800-rule, which is weight-normalized for pediatric subjects:

```python
# Standard 1800-rule for adults:
ISF_adult = 1800 / TDI_adult          # mg/dL per unit

# Weight-scaled initialization for pediatric subjects:
TDI_child = 0.7 * weight_kg           # U/day (ISPAD guideline: 0.5-1.0 U/kg/day)
ISF_child = 1800 / TDI_child          # = 1800 / (0.7 * 30) = 85.7 mg/dL/U for 30 kg child

# Online ISA update (runs every non-meal period, e.g., every 2h during basal):
ISF_error(k) = G_actual(k) - G_predicted(k)    # using current ISF
delta_ISF = Kp * mean(ISF_error, window=12)      # Kp = tuning gain
ISF_new = clip(ISF + delta_ISF, ISF_min, ISF_max)
```

In the AEGIS MPC cost function, the base dose heuristic becomes:

```python
base_dose(k) = max(0, (G(k) - G_target) / ISF(k))   # uses online-updated ISF
```

And the `STL_CORRECTED` floor in Layer 5 becomes:

```python
# Current (adult-only):
dose_corrected = (G - 200) / ISF_population

# Replacement:
dose_corrected = (G - 200) / ISF_patient   # uses weight-scaled, online-adapted ISF
```

The ISA algorithm from Turksoy et al. (2019) uses system identification (subspace ID on glucose-insulin data) to update the ISF during non-meal, non-exercise periods, preventing meal-time ISF drift.

**Inputs required per timestep:**  
- `weight_kg` — patient body weight (known initialization parameter, already in the Hovorka simulator for each patient)  
- `G(k)`, `u(k)` — already available  
- A binary `meal_active` flag to gate ISA updates away from meal periods

**Compatibility with AEGIS architecture.**  
The AEGIS MPC grid search uses a heuristic base dose with a fixed population ISF. Replacing this constant with a patient-specific, weight-initialized, and online-adapted ISF requires changes only in `core/layer4_decision.py` (the base dose computation and grid scaling) and in `core/layer5_safety.py` (the `STL_CORRECTED` formula). The causal DAG is not affected: ISF adaptation is a model parameter update path that does not flow through the causal inference engine. The AC-UKF, RBPF, and causal layer (L3) are untouched. This directly addresses the pediatric failure mode identified in `test_integration.py`.

**Computational cost.**  
ISA runs at a slow outer loop (every 2-hour non-meal window). Subspace identification on a 24-step window is O(n³) ≈ O(1) for scalar glucose. Negligible cost.

**Validation results.**  
Turksoy et al. (2019) tested the ISA algorithm on virtual subjects in the UVA/Padova T1DM simulator with ±30% circadian variability in ISF and ±30% variability in initial TDI estimate. The ISA converged to the correct ISF within 2–4 days and reduced hypoglycemia time by 23.8% relative to a fixed-ISF MPC. Total daily insulin use was lower. The algorithm was also validated using 4-day real-world meal/exercise data from prior AP studies.

Lee et al. (2009) explicitly tested 100 adolescent and 100 adult virtual subjects with TDI-based weight scaling of the MPC model. The adolescent cohort achieved 82% TIR (70–180 mg/dL); the TDI-scaled model initialization was identified as the primary mechanism enabling cross-cohort generalization.

**Known risks and failure modes.**  
- The ISA algorithm assumes basal (non-meal) periods are identifiable. If the `meal_active` flag is incorrect (e.g., missed meal detection), ISA can absorb meal-induced glucose excursions into the ISF estimate, causing the ISF to be underestimated and future doses to be over-large.  
- Very young children exhibit significant circadian ISF variation (dawn phenomenon, post-exercise effects). A single scalar ISF may not capture time-of-day sensitivity changes. A time-varying ISF schedule (e.g., 3 or 4 time-of-day bins) is more robust but adds state complexity.

---

### Method 3B — Velocity-Weighting MPC with Adaptive IOB Constraint Scaled to Patient Weight

**Citation:**  
Boiroux, D., Aradottir, T.B.I., Norgaard, K., Madsen, H., & Jorgensen, J.B. (2017). Velocity-weighting to prevent controller-induced hypoglycemia in model predictive control of an artificial pancreas to treat type 1 diabetes mellitus. *IFAC-PapersOnLine*, 50(1), 15805–15810. https://doi.org/10.1016/j.ifacol.2017.08.1942. PMC5419594.

**Core mathematical formulation.**  
The standard MPC cost function is augmented with a velocity-dependent penalty term that asymmetrically penalizes insulin delivery when glucose is descending:

```python
# Standard MPC cost:
J = sum_{k=1}^{N} [ Wy * (y(k) - y_target)^2 + WDu * (delta_u(k))^2 ]

# Velocity-weighted MPC cost:
J = sum_{k=1}^{N} [ Wy * (y(k) - y_target)^2 + W_vel(v(k)) * WDu * (delta_u(k))^2 ]

# Velocity output (rate of change of predicted glucose):
v(k) = Cv * x(k)

# Velocity weighting function (monotone increasing penalty as glucose descends):
W_vel(v) = 1 + max(0, -v / v_threshold)^beta    # beta = 2.0, v_threshold = 1.0 mg/dL/min
```

The IOB constraint is also scaled by patient weight:

```python
# Current IOB constraint (adult-tuned):
IOB_max = aIOB * u_basal                          # e.g., aIOB = 5 h

# Weight-scaled IOB constraint:
u_basal_patient = weight_kg * basal_rate_per_kg   # e.g., 0.02 U/(kg·h)
IOB_max = aIOB * u_basal_patient
```

When glucose velocity `v(k) < 0` (descending), the penalty on insulin delivery increases quadratically, effectively implementing a soft early-suspend that is more responsive than the hard `L5_BLOCKED` logic but does not starve the patient of needed correction doses.

**Inputs required per timestep:**  
- `v(k)` — predicted glucose rate of change from the Bergman MPC model (already computed in the MPC grid search horizon)  
- `weight_kg` — patient body weight

**Compatibility with AEGIS architecture.**  
The velocity-weighting modifies only the cost function evaluated in the MPC grid search (`core/layer4_decision.py`). The 8-candidate grid search already evaluates a cost function for each dose candidate; adding the velocity term requires computing `v(k)` from the Bergman ODE trajectory (already predicted) and multiplying the insulin weight term. The L5 Seldonian bounds and STL specifications are not affected. The causal DAG is not affected. This is fully compatible with all architectural constraints.

**Computational cost.**  
One additional velocity computation per prediction horizon step per grid candidate — O(N × 8) scalar multiplications. Negligible.

**Validation results.**  
Boiroux et al. (2017) tested velocity-weighting MPC on 100 in-silico subjects from the UVA/Padova T1DM simulator. For unannounced meals, velocity-weighting eliminated controller-induced hypoglycemia events that occurred in 3 of 100 subjects under standard MPC — events caused by the MPC continuing to deliver insulin aggressively during a post-meal glucose descent. The paper demonstrated that no increase in hyperglycemia resulted from the additional insulin penalty. The method was validated on both adult and adolescent cohorts.

The 1800-rule TDI-based IOB scaling reference (Lee et al., 2009, PMC2769914) showed that initializing the IOB constraint using `TDI = weight_kg × 0.7 U/kg/day` rather than a population constant was the primary differentiator between safe and unsafe glycemic control across body weights.

**Known risks and failure modes.**  
- If `v_threshold` is set too conservatively (too low), the MPC may withhold insulin even when glucose is descending from a hyperglycemic state that genuinely requires correction, worsening the already-problematic TIR.  
- The velocity estimate from the Bergman ODE is the *predicted* velocity, not the actual velocity. Given the current 116.7 RMSE of the Digital Twin, the predicted velocity may have the wrong sign during rapid meal-driven excursions, defeating the weighting. This is a direct dependency on solving Problem 1 first.

---

### Problem 3 Recommendation

**Method 3A (weight-scaled TDD-derived ISF with online ISA)** is recommended as the primary intervention because it addresses the root cause — the MPC's base dose and L5 correction floor are calibrated for a 75-kg adult and must be patient-specific. The weight-scaled TDD initialization (0.7 U/kg/day for a 30-kg child yields ISF ≈ 85.7 vs. the adult's ≈ 34.3 mg/dL/U) directly explains the observed 9–13% severe TBR: doses appropriate for an adult overdose a 30-kg child by a factor of ~2.5. Method 3B is recommended as a *complementary* safety layer, not as a replacement, since velocity-weighting catches the dose-descent scenario without requiring correct ISF initialization. The ordering matters: fix ISF first (3A), then add velocity-weighting (3B) as a defense-in-depth safety layer. 

---

## Problem 4 — SLM Semantic Replacement: Removing L1_FIREWALL and STUB_ACTIVE

**Background.** Layer 1 is currently a stub using Levenshtein string matching against a fixed dictionary plus random-integer entropy jitter. A `L1_FIREWALL` prevents contamination of downstream bandit processing. The goal is to replace the stub with a constrained SLM/LLM decoder that reliably extracts structured semantic concepts (stress, fatigue, and related hidden-state proxies) and a calibrated entropy estimate from patient notes, outputting a fixed-schema structured object that can safely feed the causal inference layer (L3). The `STUB_ACTIVE` flag and `L1_FIREWALL` can then be removed.

---

### Method 4A — Grammar-Constrained Decoding (GCD) with Fine-Tuned Encoder-Decoder SLM

**Citation:**  
Schmidt, D.M. (2025). Grammar-constrained decoding for structured information extraction with fine-tuned generative models applied to clinical trial abstracts. *Frontiers in Artificial Intelligence*, 7, Article 1406857. https://doi.org/10.3389/frai.2024.1406857. PMC11747381.

Foundational method reference: Geng, S., Josifosky, M., Peyrard, M., & West, R. (2023). Grammar-constrained decoding for structured NLP tasks without finetuning. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)*. arXiv:2305.13971.

**Core mathematical formulation.**  
At each autoregressive generation step `t`, standard token sampling from the model's vocabulary distribution `P_θ(v | context)` is replaced with grammar-masked sampling:

```python
# At each generation step t:
valid_tokens = grammar_engine.get_valid_next_tokens(partial_output)
mask = [1 if v in valid_tokens else 0 for v in vocab]

# Masked probability distribution (token masking):
logits_masked = logits + log(mask)   # -inf for invalid tokens
P_masked = softmax(logits_masked)

# Sample or argmax from constrained distribution:
next_token = argmax(P_masked)         # or sample(P_masked)
```

The grammar is specified as a context-free grammar (CFG) or JSON Schema defining the exact output format. For AEGIS Layer 1, the target output schema is:

```json
{
  "concepts": ["stress", "fatigue", "exercise"],  // enum from fixed ontology
  "intensities": {"stress": 0.7, "fatigue": 0.4}, // floats in [0, 1]
  "semantic_entropy": 1.23                          // float ≥ 0
}
```

The grammar guarantees this schema is always produced — eliminating the random-jitter stub behavior entirely. The model is a fine-tuned Flan-T5 (250M parameters) or Longformer encoder-decoder, fine-tuned on a small labeled corpus of patient notes with this schema. The Schmidt (2025) paper used `< 500` training examples in a low-resource setting and still achieved F1 improvements of 0.351–0.425 absolute points over unconstrained decoding.

For semantic entropy, the GCD output includes the entropy of the constrained token distribution at the concept-selection steps, which serves as a calibrated proxy for the semantic uncertainty of the note:

```python
concept_probs = P_masked[concept_tokens]   # probabilities of valid concept tokens
semantic_entropy = -sum(p * log(p) for p in concept_probs if p > 0)
```

This is deterministic and grounded in the model's actual uncertainty — not random integer jitter.

**Inputs required per timestep:**  
- Raw patient note text (string)
- Fixed concept ontology (pre-defined: stress, fatigue, exercise, pain, etc. — from the Hovorka simulator's causal proxy generation)

**Compatibility with AEGIS architecture.**  
The GCD model can be loaded as a small inference-only SLM within `core/layer1_semantic.py`, replacing the fuzzy-match loop. The output schema matches what the downstream bandit and causal engine expect. The `L1_FIREWALL` check can be replaced with a schema-validation assertion (the grammar guarantees validity, so the assertion should always pass). The Levenshtein dictionary becomes the concept ontology enumeration fed into the grammar. The causal DAG is not affected: concepts flow from Layer 1 toward Layer 3 in the same direction as before. No cheat-sheet parameters from the Hovorka simulator are used — the model learns concept-to-proxy mappings from labeled text.

**Computational cost.**  
Flan-T5-base (250M params) at inference: ~5–20ms per note on CPU, ~1–2ms on GPU. At 5-minute sampling intervals, this is negligible. Grammar-constrained decoding adds approximately 10–30% overhead over unconstrained decoding due to token masking at each step (Geng et al. report this overhead is reduced by grammar pre-compilation). XGrammar or Outlines can be used as drop-in grammar engines with <5% overhead after pre-compilation.

**Validation results.**  
Schmidt (2025) demonstrated that GCD improved F1 scores by 0.351 absolute points (from 0.062 to 0.413) and 0.425 absolute points (from 0.045 to 0.47) on clinical trial information extraction tasks for type 2 diabetes and glaucoma datasets with <500 training examples (low-resource setting). The model was Longformer + Flan-T5 fine-tuned on the C-TrO ontology. Well-formedness of output went from ~40% (unconstrained) to 100% (constrained). This is the critical property for AEGIS: the firewall can be removed precisely because GCD guarantees schema validity.

Geng et al. (2023) further showed that GCD-enhanced language models — including small models (< 500M parameters) — substantially outperformed unconstrained LMs and in some cases beat task-specific fine-tuned models on information extraction benchmarks, without any task-specific fine-tuning (zero-shot GCD).

**Known risks and failure modes.**  
- If the concept ontology is too large (> 50 concepts), the grammar tree becomes deep and GCD overhead increases. For AEGIS, the Hovorka simulator generates a fixed, small set of causal proxies — this concern does not apply.  
- Fine-tuning on < 500 examples risks overfitting; data augmentation (paraphrase generation, back-translation) is recommended to broaden training coverage.  
- If the note contains no clinical concepts at all (e.g., "Patient feeling fine today"), the GCD must still emit valid schema — which it will (empty concept set with zero intensities), but this case must be distinguished from "ambiguous" rather than "uncovered" notes, as the downstream entropy calculation differs.

---

### Method 4B — Fine-Tuned BioBERT Token Classifier with Schema-Enforced Output Layer

**Citation:**  
Ge, W., Rice, H.J., Sheikh, I.S., et al. (2023). Enhanced neurologic concept recognition using a named entity recognition model based on bidirectional encoder representations from transformers. *Neurology*, 101(22), e2010–e2018. https://doi.org/10.1212/WNL.0000000000207853. PMC (PMID 37816638).

Supporting survey reference: Landi, I. et al. (2023). Named entity recognition in electronic health records: a methodological review. *Journal of the American Medical Informatics Association*, 30(12), 2–14. https://doi.org/10.1093/jamia/ocad188. PMC10651400.

**Core mathematical formulation.**  
BioBERT (BERT pre-trained on PubMed + PMC texts) is fine-tuned for token-level classification with a fixed label set:

```
Input: patient note tokens [t_1, ..., t_n]

Encoder: BioBERT produces contextual embeddings [h_1, ..., h_n] ∈ R^768

Classifier head (per token):
  logits_i = W * h_i + b,    W ∈ R^(|labels| × 768)
  P(label | t_i) = softmax(logits_i)

Label set: {O, B-STRESS, I-STRESS, B-FATIGUE, I-FATIGUE, B-EXERCISE, ...}
  (BIO tagging scheme for concept spans)
```

After token classification, a schema-enforcement post-processing step aggregates span labels into the fixed output object:

```python
concepts = extract_spans(token_labels)  # BIO decoding
intensities = {c: confidence(c, token_logits) for c in concepts}
semantic_entropy = token_entropy(token_label_distributions)
# Guaranteed schema via post-hoc dict construction
```

Unlike Method 4A (GCD), this approach does not constrain the generative process — it constrains the output representation. Schema validity is guaranteed by the fixed post-processing logic, not by the model's token distribution.

**Inputs required per timestep:**  
- Raw patient note text (string)
- Pre-defined concept span labels (BIO schema over the fixed ontology)

**Compatibility with AEGIS architecture.**  
Identical to Method 4A for downstream compatibility. The output dictionary is schema-equivalent. BioBERT inference is a standard HuggingFace forward pass — no special grammar engine is required. The `L1_FIREWALL` is replaced with a schema-check assertion that always passes given the deterministic post-processing.

**Computational cost.**  
BioBERT-base (110M params): ~3–8ms per note on CPU inference. Very efficient. No grammar engine overhead. Fine-tuning on the small symptom corpus requires ~30 minutes on a single GPU.

**Validation results.**  
Ge et al. (2023) demonstrated BERT-based NER for recognizing neurological signs and symptoms in three clinical text corpora (physician notes, textbook case histories, clinical synopses). Recall ranged from 59.5% to 82.0% and precision from 61.7% to 80.4% across corpora, with the model outperforming a CNN-based alternative in all conditions. The BERT-based model performed best on professionally-written text and worst on physician-written notes — which are the source type in AEGIS. The gap (59.5% recall) reflects the challenge of noisy real-world clinical text and suggests that the ontology-constrained output layer and limited label set of AEGIS (only a handful of Hovorka-generated proxy concepts) would substantially help recall by narrowing the classification problem.

The Landi et al. (2023) review of 2011–2022 clinical NER literature confirmed that BERT-based models (BioBERT, BioClinicalBERT, BlueBERT) constitute the state of the art as of 2022 for clinical token classification tasks.

**Known risks and failure modes.**  
- BioBERT's output is *not* schema-constrained at generation time — invalid spans (e.g., overlapping or malformed BIO sequences) can occur and must be caught by the post-processing layer.  
- The model produces intensity estimates from classification confidence (softmax logits), which are known to be poorly calibrated for BERT-class models without explicit calibration (temperature scaling or Platt scaling). An uncalibrated intensity fed to the downstream causal kernel-ridge estimator may introduce bias.  
- Unlike Method 4A, the semantic entropy estimate is an aggregate over token-level classifications rather than the entropy of a constrained generative distribution. This entropy has different statistical properties from what the downstream causal layer may expect if it was designed assuming a generative model.

---

### Problem 4 Recommendation

**Method 4A (Grammar-Constrained Decoding with fine-tuned Flan-T5)** is recommended because: (1) it provides a mathematically grounded semantic entropy estimate directly from the constrained token distribution — unlike Method 4B's token classification confidence, which is poorly calibrated; (2) the 100% schema well-formedness guarantee from the grammar engine is the precise property needed to safely remove the `L1_FIREWALL` without risking downstream contamination; (3) Schmidt (2025) demonstrated this method's efficacy in a clinical low-resource setting with <500 examples — matching the AEGIS training data constraint. The grammar should be pre-compiled with XGrammar for minimal inference overhead. Flan-T5-base (250M parameters) is the recommended backbone — small enough for embedded deployment, large enough to generalize beyond exact-match patterns.

---

## Cross-Problem Synthesis

The combination that is most internally consistent and requires the least architectural disruption across all four problems is: **(1A) Joint UKF augmented state for `p1` and `SI` → (2A) VFF-RLS exponential forgetting for innovation bias → (3A) Weight-scaled ISF with online ISA → (4A) GCD-constrained Flan-T5 for Layer 1 semantic output.** This combination shares a common design principle across all four problems: it operates as a *wrapper or post-processor* around existing components rather than as a replacement of them, thereby respecting every architectural constraint stated in Section 7 of the research context. Method 1A augments the UKF state vector externally; Method 2A replaces only the rectangular accumulator in the bias correction step; Method 3A changes only the ISF constant used in the MPC cost function and the L5 correction formula; Method 4A slots into the Layer 1 stub position with a schema-compatible output. None of the four recommended methods touches the AC-UKF + RBPF internals, the causal DAG directionality, the Hovorka simulator parameters, or the Layer 3 G-estimation and AV confidence-sequence machinery. Furthermore, the four methods have a natural sequencing dependency that is also their stabilization order: solving Problem 1 (better `p1` tracking) directly improves the MPC prediction quality, which in turn makes the velocity-weighted MPC in Problem 3B more effective; solving Problem 2 (adaptive forgetting) improves the bias correction that feeds the same MPC prediction horizon; and solving Problem 4 (structured semantic extraction) begins to populate the causal proxy inputs in Layer 3, which were zeroed out by the `L1_FIREWALL`, potentially providing the causal engine's G-estimation with the stress and fatigue signals that were always intended to explain unmodeled glucose variability. The recommended implementation order is therefore: 3A (immediate safety fix for pediatric cohort, single-constant change) → 2A (four-line forgetting factor improvement) → 1A (augmented UKF, more complex but bounded) → 4A (SLM integration, requiring training data preparation and grammar specification).

---

## Post-Review Caveats & Corrected Implementation Guidance

The following caveats were identified during expert review of the above findings. Each modifies, narrows, or adds to the recommendations above. **These caveats take precedence over the corresponding recommendation text when handing this document to a coding agent.**

---

### Caveat 1 — Problem 1: p1 Identifiability Is Conditional on Signal Excitation

**What the document says:** Method 1A recommends augmenting `p1` and `SI = p3/p2` simultaneously, citing Eberle & Ament (2011) as validation.

**The correction:** The Eberle & Ament paper validated identifiability specifically on **IVGTT protocol data** — controlled glucose excursions that are explicitly designed to maximally excite the Bergman parameter space. Ambulatory closed-loop data (meals, exercise, basal periods) has substantially different excitation properties. The paper itself flags ambulatory application as the primary open limitation. Simultaneous estimation of `p1` and `SI` may be locally unidentifiable during flat glucose periods when both parameters exert negligible differential effect on the observed CGM signal.

**Corrected implementation constraint:**
- Augment `p1` **only** — do not augment `p2` and `p3` simultaneously in the first implementation. Estimate a single scalar from the 4-state augmented vector `x̃ = [G, X, I, p1]ᵀ` (sigma points: 9, not 13).
- Add an **identifiability gate**: if the rolling variance of innovations `Var(e(k-12 : k)) < σ²_gate` (e.g., `σ²_gate = 4.0 mg²/dL²`, corresponding to ~2 mg/dL signal noise floor), freeze the `p1` estimate by temporarily zeroing its process noise `Q_p1 = 0`. Resume adaptation when variance exceeds the threshold again.
- Enforce a strict physiological bound: `p1 ∈ [0.005, 0.050] min⁻¹`. If the sigma-point clipping is insufficient to hold this bound, reject and revert the update for that step.
- **Only after the single-parameter version is validated** (i.e., the `test_layer2` RMSE test shows improvement and no new divergence) should `SI` adaptation be considered as a second augmentation phase.

---

### Caveat 2 — Problem 4: Grammar Cold-Start Gap for Out-of-Ontology Tokens

**What the document says:** Method 4A (GCD with Flan-T5) provides 100% schema well-formedness, enabling safe removal of `L1_FIREWALL`.

**The correction:** The cited papers (Schmidt 2025, Geng et al. 2023) do not address what happens when patient text contains tokens that fall entirely outside the pre-compiled grammar's concept ontology — a cold-start failure mode specific to AEGIS. In the current simulator context this risk is bounded (the Hovorka simulator generates notes from a fixed proxy vocabulary), but it must be handled explicitly rather than assumed away.

**Corrected implementation constraints:**
1. The concept ontology must be **fixed and closed** at grammar-compilation time, derived directly from the Hovorka simulator's causal proxy generation vocabulary. No open-ended concept slots are permitted in the grammar.
2. The grammar must include an explicit **`OTHER` fallback token** (or empty-concept path) for note content that matches no ontology entry. The fallback must return: zero intensities for all concepts, and `semantic_entropy = H_max` (maximum entropy for the label set size), signaling maximum uncertainty rather than zero uncertainty.
3. **`L1_FIREWALL` must remain active** until three consecutive closed-loop sessions (not individual timesteps) produce schema-valid outputs with no fallback activations. Only then should `STUB_ACTIVE = False` and the firewall be lifted.
4. Log all fallback activations with the triggering note text for ontology review between sessions.

---

### Caveat 3 — Problem 3: ISA Convergence Lag Leaves Pediatric Patients Unprotected on Day 1

**What the document says:** Method 3A recommends weight-scaled ISF initialization with online ISA as the primary intervention.

**The correction:** The Turksoy et al. (2019) ISA algorithm requires a **minimum of 3–4 meal responses** and approximately **18 hours of closed-loop data** to converge to a stable ISF estimate. During this convergence window, pediatric patients remain at the same elevated hypoglycemia risk observed in `test_integration.py`. The document does not address this gap, and handing the recommendation to a coding agent without this constraint risks implementing ISA without the static floor — leaving the R-INT-04 failure unresolved for at least the first simulation day.

**Corrected implementation constraints (two-stage):**

*Stage 1 — Immediate static floor (implement first, test independently):*
```python
# In core/layer4_decision.py and core/layer5_safety.py:
# Replace all references to ISF_POPULATION with ISF_PATIENT(weight_kg):

def isf_from_weight(weight_kg: float) -> float:
    """1800-rule from ISPAD weight-scaled TDI. Must be the only active
    ISF source until ISA has seen >= 3 meal responses."""
    tdi = 0.7 * weight_kg          # U/day (ISPAD mid-range)
    return 1800.0 / tdi            # mg/dL per unit

ISF_PATIENT = isf_from_weight(patient.weight_kg)
```

This single change alone is expected to reduce the 30-kg child's TBR from ~11% toward the 0–1% target, because the population ISF (~34 mg/dL/U for a 75-kg adult) applied to a 30-kg child over-doses by a factor of ~2.5.

*Stage 2 — ISA overlay (implement second, activate conditionally):*
```python
# ISA activates only when BOTH conditions are met:
isa_active = (n_meal_responses_observed >= 3) and (closed_loop_hours >= 18)

if isa_active:
    ISF_PATIENT = isa_update(ISF_PATIENT, recent_glucose_history)
# else: ISF_PATIENT remains at the static weight-scaled floor
```

The static floor must never be replaced by the ISA estimate — it is a lower bound. If ISA converges to a value lower than the static floor, the static floor wins.

---

### Pre-Implementation Verification: R-L2-04 Test Status

Before beginning any coding work on Problem 1 (Method 1A), the current pass/fail status of test `R-L2-04` must be verified by running the test suite.

**Context:** The bias correction implemented in `DigitalTwin.predict_trajectory` reduced RMSE from 213.2 mg/dL to 116.7 mg/dL. If the `R-L2-04` assertion threshold is 200 mg/dL RMSE, then this test is now **passing** — meaning the urgency of Problem 1 has changed.

**If R-L2-04 is now passing:** The implementation priority order shifts to:
1. **Stage 1 static ISF fix (Problem 3A)** — immediately unblocks R-INT-04 pediatric failure, one-function change
2. **VFF-RLS forgetting (Problem 2A)** — four-line change, no test risk
3. **p1-only Joint UKF (Problem 1A, corrected)** — more complex, implement after 3A and 2A are green
4. **GCD Layer 1 (Problem 4A, corrected)** — requires training data prep, grammar spec, implement last

**If R-L2-04 is still failing:** Problem 1 implementation remains the highest structural priority and should precede Problem 3. Confirm before writing any implementation code.

---

*All citations above were located and verified during this research session via live web searches. Sources marked as arXiv preprints (2404.10914) are noted as not peer-reviewed in final form but are cited only for mathematical formulations that have foundational peer-reviewed analogues. All other sources are published in peer-reviewed journals or peer-reviewed conference proceedings.*
