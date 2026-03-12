# AEGIS 3.0 — Coding Agent Implementation Prompt

**Read this document completely before writing a single line of code.**  
This prompt is sequenced. Each phase has a mandatory verification gate before the next phase begins. Do not batch phases together. Do not implement Phase N+1 if Phase N's gate fails.

---

## Pre-Phase 0 — Verify R-L2-04 Test Status (No Code Changes)

Before any implementation, run the test suite and report the pass/fail result of test `R-L2-04` specifically.

```bash
python run_all.py 2>&1 | grep -E "R-L2-04|PASS|FAIL"
```

Report the exact assertion threshold and the current RMSE value the test measures. Do not proceed until this result is known.

**Branch on result:**
- If **R-L2-04 is PASSING** → proceed to Phase 1 (static ISF fix) as the first coding task.
- If **R-L2-04 is FAILING** → report the failing RMSE and threshold before proceeding. The Phase ordering below still applies, but flag this for review.

---

## Phase 1 — Static Weight-Scaled ISF Fix (Problem 3, Stage 1)

**Source:** research_context.md, Problem 3 / Caveat 3 — Stage 1.  
**Estimated complexity:** Low. Target: 1–2 functions changed, zero new dependencies.  
**Test gate:** R-INT-04 (pediatric multi-patient safety) must move from FAIL to PASS before Phase 2 begins.

### What to implement

Add a `isf_from_weight(weight_kg: float) -> float` function implementing the weight-scaled 1800-rule. Replace every hardcoded population ISF constant in `core/layer4_decision.py` and `core/layer5_safety.py` with a call to this function using the patient's body weight.

```python
def isf_from_weight(weight_kg: float) -> float:
    """
    Compute patient-specific Insulin Sensitivity Factor from body weight.
    Uses the 1800-rule applied to ISPAD weight-scaled TDI (0.7 U/kg/day).
    This is the ONLY active ISF source until ISA is enabled in Phase 3.
    
    Examples:
      weight_kg=30  → ISF = 1800 / (0.7 * 30) = 85.7 mg/dL/U
      weight_kg=75  → ISF = 1800 / (0.7 * 75) = 34.3 mg/dL/U
    """
    tdi = 0.7 * weight_kg
    return 1800.0 / tdi
```

The `STL_CORRECTED` minimum correction dose in `core/layer5_safety.py` must also use the weight-scaled ISF:

```python
# Replace:
dose_corrected = (glucose - 200.0) / ISF_POPULATION

# With:
dose_corrected = (glucose - 200.0) / isf_from_weight(patient.weight_kg)
```

### Hard constraints

- **Do not add ISA logic in this phase.** The ISA overlay is Phase 3. This phase is static-only.
- `patient.weight_kg` must be read from the existing patient parameter object. Do not add a new parameter or config entry.
- The population ISF constant must not remain in any code path that affects dose computation or safety correction. Search for it explicitly and verify removal.
- Do not modify `core/layer3_causal.py`, `core/layer2_digital_twin.py`, the AC-UKF, the RBPF, or the Hovorka simulator.

### Gate

Run the full 43-test suite. Report pass count overall and specifically: did R-INT-04 pass? Did any previously passing test regress? Do not proceed to Phase 2 if any previously passing test now fails.

---

## Phase 2 — VFF-RLS Exponential Forgetting for Innovation Bias (Problem 2)

**Source:** research_context.md, Problem 2 / Method 2A.  
**Estimated complexity:** Low. Target: ~10 lines changed inside `DigitalTwin.predict_trajectory`, zero new dependencies.  
**Test gate:** The R-L2-04 RMSE value must not increase relative to the pre-Phase-2 baseline. Run a 2-day single-patient simulation (`seed=42`) and report RMSE before and after.

### What to implement

Replace the existing 20-step rectangular window running-mean bias accumulator with an exponentially-weighted moving average (EMA) using a variable forgetting factor.

```python
# Class-level state (add to DigitalTwin.__init__):
self.bias_ema: float = 0.0          # replaces the rectangular window mean
self.innovation_var_ema: float = 1.0 # for computing sigma_e dynamically

# Configuration constants (add near top of file, not hardcoded inline):
LAMBDA_MAX: float = 0.97     # slow forgetting during steady-state
LAMBDA_MIN: float = 0.82     # fast forgetting during rapid transients
ALPHA_THRESHOLD: float = 1.8 # innovations > alpha * sigma_e trigger fast mode

# Per-timestep update (replaces the window accumulator logic):
def _update_bias_ema(self, innovation: float) -> None:
    """
    Variable forgetting factor EMA bias update.
    innovation = G_CGM(k) - G_predicted(k)  (already computed by UKF)
    """
    # Update innovation variance EMA (for threshold detection):
    self.innovation_var_ema = (
        LAMBDA_MAX * self.innovation_var_ema + (1 - LAMBDA_MAX) * innovation ** 2
    )
    sigma_e = max(self.innovation_var_ema ** 0.5, 1e-3)

    # Choose forgetting factor based on innovation magnitude:
    if abs(innovation) > ALPHA_THRESHOLD * sigma_e:
        lam = LAMBDA_MIN   # rapid transient — respond quickly
    else:
        lam = LAMBDA_MAX   # steady state — smooth over long window

    # EMA bias update:
    self.bias_ema = lam * self.bias_ema + (1 - lam) * innovation
```

The bias injection into the prediction horizon (initial-state additive correction and per-step recursive attenuation) remains unchanged in structure — only the `self.bias_ema` value replaces the old rectangular mean wherever it was used.

### Hard constraints

- **Do not change the injection logic** — only the accumulator that produces the bias value.
- Do not modify the AC-UKF or RBPF code paths.
- `LAMBDA_MAX`, `LAMBDA_MIN`, and `ALPHA_THRESHOLD` must be named constants at module scope, not magic numbers embedded in the function body.
- If the post-Phase-2 RMSE is higher than the pre-Phase-2 RMSE by more than 5 mg/dL, revert and report the discrepancy. Do not proceed to Phase 3 in that case.

### Gate

Run the full 43-test suite. Report RMSE change relative to Phase 1 baseline. No previously passing test may regress.

---

## Phase 3 — p1-Only Augmented UKF (Problem 1)

**Source:** research_context.md, Problem 1 / Method 1A, as corrected by Caveat 1.  
**Estimated complexity:** Medium. Target: new wrapper class around DigitalTwin, no modification of AC-UKF internals.  
**Test gate:** R-L2-04 RMSE must decrease relative to Phase 2 baseline, AND no new test failures.

### What to implement

Implement a `BergmanParameterAdapter` class that wraps the existing `DigitalTwin` and maintains a single augmented state: `x̃ = [G, X, I, p1]ᵀ`. This class calls into `DigitalTwin` normally but substitutes the current `p1` estimate into the ODE before each prediction step.

```python
class BergmanParameterAdapter:
    """
    Augments the DigitalTwin with online p1 (glucose effectiveness) estimation.
    Implements the single-parameter Joint UKF from Eberle & Ament (2011),
    restricted to p1-only as required by the identifiability analysis.
    
    CONSTRAINT: Does NOT modify DigitalTwin, AC-UKF, or RBPF internals.
    All augmentation happens via external state substitution.
    """

    # Physiological hard bounds on p1 (min⁻¹):
    P1_MIN: float = 0.005
    P1_MAX: float = 0.050

    # Process noise for p1 random-walk model:
    Q_P1_ACTIVE: float = 1e-8    # when identifiability gate is open
    Q_P1_FROZEN: float = 0.0     # when gate is closed (flat glucose)

    # Identifiability gate threshold:
    INNOVATION_VAR_GATE: float = 4.0  # mg²/dL² — ~2 mg/dL noise floor

    def __init__(self, digital_twin: DigitalTwin, p1_init: float = 0.028):
        self.twin = digital_twin
        self.p1_est: float = p1_init
        self.p1_var: float = 1e-4    # initial uncertainty on p1

    def _identifiability_gate_open(self) -> bool:
        """Returns True if innovation variance exceeds the threshold."""
        return self.twin.innovation_var_ema >= self.INNOVATION_VAR_GATE

    def update(self, glucose_obs: float) -> float:
        """
        Run one UKF step with p1-augmented state.
        Returns updated p1 estimate.
        """
        # Freeze adaptation during flat glucose (unobservable regime):
        q_p1 = self.Q_P1_ACTIVE if self._identifiability_gate_open() else self.Q_P1_FROZEN

        # --- Prediction step for p1 (random walk) ---
        p1_pred = self.p1_est
        p1_var_pred = self.p1_var + q_p1

        # --- Substitute current p1 into DigitalTwin ODE ---
        self.twin.set_p1(p1_pred)   # thin setter — adds one attribute, no internals

        # --- Innovation from DigitalTwin's last UKF step ---
        innovation = self.twin.last_innovation  # already computed during twin.update()

        # --- Measurement update for p1 ---
        # Linearized sensitivity of G to p1: dG/dp1 ≈ (Gb - G) * dt
        H_p1 = (self.twin.glucose_basal - self.twin.state[0]) * self.twin.dt
        S = H_p1 * p1_var_pred * H_p1 + self.twin.R_noise
        K_p1 = p1_var_pred * H_p1 / S     # Kalman gain for p1

        p1_updated = p1_pred + K_p1 * innovation
        p1_var_updated = (1 - K_p1 * H_p1) * p1_var_pred

        # Enforce physiological bounds:
        self.p1_est = float(np.clip(p1_updated, self.P1_MIN, self.P1_MAX))
        self.p1_var = float(np.clip(p1_var_updated, 1e-10, 1e-3))

        return self.p1_est
```

The `DigitalTwin` requires a thin `set_p1(value: float)` setter that stores the value as an instance attribute to be picked up at the next ODE integration step. This setter must not enter the AC-UKF or RBPF code paths — it only sets a scalar attribute on the `DigitalTwin` instance that the Bergman ODE integration already reads.

### Hard constraints

- **Augment p1 only.** Do not augment `p2`, `p3`, or `SI = p3/p2` in this phase. The research context recommends this explicitly and Caveat 1 makes it mandatory.
- **Do not modify the AC-UKF or RBPF implementations.** The `BergmanParameterAdapter` is a new class that wraps `DigitalTwin` from the outside.
- The identifiability gate is **not optional.** It must be implemented exactly as specified. Without it, `p1` will drift during flat basal periods.
- If `p1` hits either physiological bound for more than 10 consecutive timesteps, log a warning. Do not raise an exception — the bounds are the safety net, not an error condition.
- After this phase, report the estimated `p1` value at the end of the 2-day simulation alongside the RMSE. A converged `p1` estimate higher than the initial 0.028 is expected (the Bergman model self-corrects too fast; a higher `p1` would make it correct even faster, which is wrong — so watch for unexpected convergence direction and report it).

### Gate

Full 43-test suite. R-L2-04 RMSE must decrease. R-INT-04 must remain passing (from Phase 1). No regressions.

---

## Phase 4 — ISA Overlay (Problem 3, Stage 2)

**Source:** research_context.md, Problem 3 / Caveat 3 — Stage 2.  
**Estimated complexity:** Low-medium. Target: new outer-loop function activated by data gate.  
**Prerequisite:** Phase 1 must be complete and R-INT-04 passing. Phase 4 does not depend on Phases 2 or 3.

### What to implement

Add an `InsulinSensitivityAdapter` class that updates `ISF_PATIENT` on a slow outer loop. The static floor from Phase 1 must remain a hard lower bound.

```python
class InsulinSensitivityAdapter:
    """
    Slow outer-loop ISF adaptation. Activates only after sufficient
    meal-response data has been accumulated. The static weight-scaled
    ISF floor (Phase 1) is NEVER replaced — it is the lower bound.
    """

    ACTIVATION_MEAL_RESPONSES: int = 3
    ACTIVATION_HOURS: float = 18.0
    KP_ISF: float = 0.05           # proportional gain for ISF update
    WINDOW_STEPS: int = 12         # steps over which to average ISF error

    def __init__(self, weight_kg: float):
        self.isf_floor = isf_from_weight(weight_kg)  # Phase 1 function
        self.isf_current = self.isf_floor
        self.meal_responses_seen: int = 0
        self.closed_loop_hours: float = 0.0

    @property
    def active(self) -> bool:
        return (
            self.meal_responses_seen >= self.ACTIVATION_MEAL_RESPONSES
            and self.closed_loop_hours >= self.ACTIVATION_HOURS
        )

    def update(self, glucose_history: list[float], u_history: list[float]) -> float:
        """
        Update ISF estimate using glucose-insulin data from a non-meal window.
        Must only be called when meal_active == False.
        Returns the current ISF_PATIENT value.
        """
        if not self.active:
            return self.isf_current

        # Simple proportional ISF correction from glucose trend:
        mean_glucose = np.mean(glucose_history[-self.WINDOW_STEPS:])
        glucose_error = mean_glucose - 120.0     # deviation from target
        isf_correction = self.KP_ISF * glucose_error

        isf_candidate = self.isf_current + isf_correction

        # Hard lower bound: static weight-scaled floor always wins:
        self.isf_current = max(isf_candidate, self.isf_floor)
        return self.isf_current
```

### Hard constraints

- **The static ISF floor (`isf_from_weight`) is a hard lower bound.** If the ISA update produces a value below the floor, the floor value is used. No exception.
- `update()` must **only be called when `meal_active == False`** and the closed-loop session has been running for at least `ACTIVATION_HOURS`. Gate this in the calling code, not inside the method.
- Do not change Phase 1's `isf_from_weight` function. Phase 4 imports and calls it.

### Gate

Full 43-test suite. All previously passing tests must remain passing. R-INT-04 must remain passing. Report final TIR and TBR for the 2-day single-patient simulation (adult, seed=42) and for the multi-patient pediatric test.

---

## Phase 5 — GCD Layer 1 (Problem 4) — Deferred, Requires Separate Session

Phase 5 (Grammar-Constrained Decoding / Flan-T5 replacement of the Layer 1 stub) is scoped for a separate implementation session. Do not begin Phase 5 in this session.

Before Phase 5 begins, the following preparation must be completed outside of code:
1. Enumerate the fixed, closed concept ontology from the Hovorka simulator's causal proxy generation code.
2. Prepare a labeled training corpus of at least 50 (target: 200) patient note examples with ground-truth concept annotations drawn from that ontology.
3. Define the exact JSON schema that the Layer 1 output must conform to, including the `semantic_entropy` field range and units.
4. Confirm the `L1_FIREWALL` remains active throughout Phases 1–4. It must not be touched until Phase 5 is validated.

If these preparation items are not complete, Phase 5 must not begin regardless of the state of Phases 1–4.

---

## General Hard Constraints (Apply to All Phases)

These constraints are drawn directly from `research_context.md` Section 7 and the post-review caveats. They are non-negotiable.

1. **Do not modify `core/layer2_digital_twin.py`'s AC-UKF or RBPF implementations.** External wrappers and thin setters/getters are permitted. Internal modifications to filter update equations, covariance propagation, or switching logic are not.
2. **Do not modify `core/layer3_causal.py`.** The causal DAG direction, G-estimation formulation, and Anytime-Valid Confidence Sequences are architecturally correct and stable. They are not inputs to any problem in this session.
3. **Do not modify the Hovorka simulator** (`simulator/patient.py`). No parameter values from the simulator may be read and transferred into the pipeline. This would constitute a cheat-sheet violation.
4. **Do not remove `L1_FIREWALL` or set `STUB_ACTIVE = False`** in any phase of this session.
5. **After each phase, run the full 43-test suite and report the complete pass/fail breakdown** — not just the target tests. Partial reporting is not acceptable.
6. **Report all changes as a diff or explicit file-by-file list.** Do not summarize changes in prose without also providing the actual code.

---

## Expected Outcomes Summary

| Phase | Target test | Expected result |
|-------|-------------|-----------------|
| Pre-0 | R-L2-04 | Verify pass/fail status only |
| 1 | R-INT-04 | FAIL → PASS (pediatric hypoglycemia fixed) |
| 2 | R-L2-04 | RMSE unchanged or lower |
| 3 | R-L2-04 | RMSE lower than Phase 2 baseline |
| 4 | R-INT-04 | Remains PASS; ISA overlay active after Day 1 |
| 5 | R-L1-* | Deferred — separate session |
