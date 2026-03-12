# AEGIS Findings Log

Observations found during work sessions that are outside the current task scope.

---

## 2026-02-27 — Pre-Implementation Audit

### Finding 1: Pipeline proxy data flow is broken
- **Location:** `core/pipeline.py`, lines 76-77
- **Issue:** `proxy_z = l1.get('proxy_z')` and `proxy_w = l1.get('proxy_w')` — but Layer 1's `process()` returns keys `z_proxy` and `w_proxy`, not `proxy_z` and `proxy_w`. This means Layer 3 never receives proxy data from the pipeline.
- **Impact:** R-INT-01 fails because `all_layers_active` is False — Layer 3 causal inference is never invoked during integration tests.
- **Fix scope:** Session 6 (Pipeline)

### Finding 2: Integration test clinical metrics use raw simulator glucose
- **Location:** `tests/test_integration.py`, line 58
- **Issue:** `glucose_trace = patient_data['glucose_mg_dl'].values` — this is the *raw simulator output*, not the pipeline's controlled glucose. Clinical metrics (TIR, TBR) should be computed on the glucose under the pipeline's control loop, not the open-loop simulator.
- **Impact:** TIR is 5.8% (fails the 70% threshold) because the raw Hovorka simulator without a controller naturally produces out-of-range glucose. The pipeline doesn't actually close the loop.
- **Fix scope:** Session 6 (Pipeline) — this is a fundamental architectural issue

### Finding 3: VALIDATION_REPORT.md falsely claims "No Mocks"
- **Location:** `VALIDATION_REPORT.md`, line 383
- **Issue:** The report states "No Mocks: All layers use real implementations" but Layer 1 uses a fake semantic entropy computation and fuzzy string matching as concept extraction.
- **Fix scope:** Addressed by STUB_REGISTRY.md and updated run_all.py STUB scanner

---

## 2026-02-27 — Closed-Loop Integrity Audit

### Finding 4: IOB scaffold modifies glucose measurement, not insulin delivery

- **Location:** `core/pipeline.py`, `run_simulation()`
- **Issue:** The closed-loop controller added in Task 2 applies an IOB-based offset to the **glucose reading** (`controlled_glucose = sim_glucose + glucose_offset`). This modifies the *measurement* that the pipeline sees, not the *insulin delivery* to the physiological simulator. These are categorically different interventions and produce different physiological dynamics.

**Evidence (from `check_closed_loop.py`):**

| Metric | Value |
|--------|-------|
| L5→Simulator mismatch rate | 7/10 sampled steps |
| Simulator insulin mean | 0.153 u/step |
| Pipeline L5 output mean | 1.331 u/step |
| Correlation (L5 ↔ Sim insulin) | 0.164 |

**Root cause:** `generate_dataset()` pre-computes all glucose via Hovorka ODE into a static DataFrame. `run_simulation()` iterates over this static trace. L5's insulin decision is accumulated into IOB which offsets the glucose *reading*, but the Hovorka ODE's `insulin_input` on the next step is still the dataset's pre-computed value — the ODE never receives the pipeline's decision.

**Correct architecture:** The Hovorka simulator must be stepped interactively. At each timestep:
1. Simulator provides glucose from its current ODE state
2. Pipeline processes glucose → L1→L2→L3→L4→L5 → insulin decision
3. L5's insulin decision is fed to the simulator as the ODE's `insulin_input`
4. Simulator advances one step with that insulin, producing the next glucose

This requires `HovorkaPatientSimulator` to expose a `step(insulin_input, carb_input)` method instead of `generate_dataset()`. The carb/meal schedule can still be pre-generated, but insulin must come from the pipeline.

---

## 2026-03-10 — L1 Firewall + ACB Baseline Fix

### Fix 1: L1 Stub Contamination Firewall (Priority 1)
- **Location:** `core/pipeline.py` → `AEGISPipeline.step()`, `core/layer1_semantic.py` → `SemanticSensorium`
- **Problem:** H4 diagnostic confirmed 38.5% of insulin doses were contaminated by stub L1 outputs propagating through L3 proxy buffers into L4 bandit arm selection.
- **Fix:** Added `STUB_ACTIVE = True` class flag to `SemanticSensorium`. When True, `pipeline.py` bypasses L3 proxy adjustment entirely — L4 receives `l3=None`, operating on glucose state and PK model output only.
- **Verification:** `tests/test_l1_firewall.py` (FW-01 through FW-03)

### Fix 2: ACB Baseline Subtraction (Priority 2)
- **Location:** `core/pipeline.py`, line ~119
- **Problem:** `baseline=None` passed to `self.layer4.update()`, disabling action-centering (Eq. 7). Bandit was learning PID residuals confounded by controller's own reward contribution.
- **Fix:** Changed to `baseline=outcome` so ACB learns only the treatment effect τ(S_t).
- **Verification:** `tests/test_l1_firewall.py` (FW-04)

---

## 2026-03-10 — L5 STL Upper Glucose Bound Gap (Pre-MPC Investigation)

### Finding 5: No STL specification enforces in-range upper glucose

- **Location:** `core/layer5_safety.py` → `STLMonitor.check_no_severe_hyper()`
- **Issue:** φ₂ checks `□[0,T](G ≤ 400)` using `L5_HYPER_CRITICAL = 400`. There is no specification for `G ≤ 250` or `G ≤ 180`. The Tier 1 reflex controller only blocks insulin when glucose < 70 or caps at `L5_MAX_BOLUS`; it never forces corrections upward. The Tier 3 Seldonian only monitors `P(G<54) ≤ 1%`.
- **Impact:** An aggressive MPC can run a patient chronically at 200–300 mg/dL without triggering any L5 safety block. The MPC's cost function is the *only* component penalizing hyperglycemia below 400 mg/dL.
- **Recommendation:** Add `φ₄: □[0,T](G ≤ 250)` as a new STL specification. Defer to a future session — do not modify L5 this session.

---

## 2026-03-10 — IOB Guard Removal Impact (Session B, Task 2)

### Finding 6: IOB guard was suppressing MPC-recommended doses

- **Location:** `core/pipeline.py` → `_run_closed_loop()`
- **Removed:** IOB exponential decay guard (τ=75 min, max_iob=BW/4), proportional dose scaling when IOB > 50%, PLGS suspend (glucose < 90 → total_insulin = 0)
- **Before removal:** TIR=10.8%, TAR_severe=70.5%, mean=307.4 mg/dL
- **After removal:** TIR=27.3%, TAR_severe=50.5%, mean=265.2 mg/dL
- **TBR_severe:** 0.0% both before and after — no hypoglycemia risk increase
- **Root cause:** The IOB guard used a single-exponential model (τ=75 min) inconsistent with the Bergman ODE's own insulin modeling. When MPC recommended a dose based on Bergman state, the IOB guard would scale it down based on a different PK model. The PLGS suspended ALL insulin (including basal) when glucose < 90, even though MPC already returns 0 dose below 80 mg/dL.
- **Implication:** TIR is still 27.3% — the remaining gap to clinical relevance (≥70%) is from Bergman-Hovorka model mismatch and MPC cost function tuning (not from IOB/PLGS). The MPC's own forward prediction, combined with L5 safety reflex, is sufficient for insulin safety without a separate IOB guard.

---

## 2026-03-10 — φ₄ Behavior Gap (Session C, Task 1)

### Finding 7: φ₄ was detecting but not correcting persistent hyperglycemia

- **Code path traced:** `_tier2_stl()` returned `(proposed_action=0.0, 'STL_BLOCKED', ...)` → `pipeline.step()` set `L5_final_action=0.0`, `L5_tier='STL_BLOCKED'` → `_run_closed_loop` saw `STL_BLOCKED` → delivered only basal insulin. Patient received zero correction dose while cycling through φ₄ violations indefinitely.
- **Fix:** `_tier2_stl()` now computes a minimum correction dose: `(glucose - 200) / ISF` where ISF=50 mg/dL/U, capped 0.25–1.0U. Returns new tier `STL_CORRECTED` instead of `STL_BLOCKED`. `_run_closed_loop` delivers `STL_CORRECTED` doses normally (not zeroed). `pipeline.step()` treats `STL_CORRECTED` as a counterfactual update for the bandit (L5 overrode MPC).
- **Verification:** 310 mg/dL for 35 min → returns 1.0U correction with tier `STL_CORRECTED`. L5 suite 7/7 pass.

---

## 2026-03-10 — Neural Residual Cannot Close Bergman-Hovorka Gap (Session C, Task 2)

### Finding 8: Neural residual is not the right lever for structural model mismatch

- **Step 2a — Bias direction:** Mean signed error = **-131.7 mg/dL** over 20 steps. Bergman massively under-predicts glucose at high ranges (actual 293–370 mg/dL, Bergman predicts 119–291). Root cause: Bergman's `p1` glucose effectiveness (0.028/min) drives too-fast glucose self-correction compared to Hovorka's 3-compartment insulin action dynamics.
- **Step 2b — Neural residual training:** Increasing from train_nn_every=200 to train_nn_every=50. R-L2-04 RMSE: 213.2 → 209.3 mg/dL (**1.8% improvement — below 10% threshold**). Clinical metrics: TIR 17.4% → 23.4% (+6.1pp), TBR_severe 0.0% both.
- **Conclusion:** The neural residual cannot absorb a structural mismatch of this magnitude. A 5-input shallow network cannot learn the Hovorka model's 10-state ODE dynamics from sparse observations. The bias is not random noise — it is a consistent directional error from a misspecified glucose dynamics model.
- **Alternatives (tractable given existing infrastructure):**
  1. **Online Bergman parameter fitting:** Fit `p1` (glucose effectiveness) and `p2`/`p3` (insulin action) to recent patient observations. This directly addresses the root cause — p1 is too large.
  2. **Bias correction term:** Add a glucose-dependent bias term to `predict_trajectory()`: `predicted_glucose += bias(glucose_level)` where bias is estimated from recent innovation sequences (DT prediction - actual).
  3. **Multi-step innovation correction:** Use the UKF's innovation `ν = y - ŷ` accumulated over the last N steps to apply a multiplicative correction to each forward prediction step.

---

## 2026-03-10 — Innovation Bias Correction + W_INSULIN Tuning (Session D, Tasks 2–3)

### Finding 9: Innovation-based bias correction reduces RMSE by 45%

- **Method:** `bias = mean(last 20 innovations)` applied additively to `predict_trajectory()` initial state (+full bias) and each forward step (+10% of bias per step).
- **Result:** Mean signed error: -131.7 → **-5.3 mg/dL**. R-L2-04 RMSE: 213.2 → **116.7 mg/dL** (45.3% improvement).
- **Impact:** This is the single most effective improvement in the MPC pipeline. The Bergman model's systematic glucose under-prediction is now corrected online from the UKF's own innovation sequence.
- **Limitation:** The bias is stationary over the 20-step window. Rapid physiology changes (exercise, illness) may cause bias lag. A forgetting factor could address this in future work.

### Finding 10: W_INSULIN sweep — TIR peaks at 36.3%, model mismatch still binding

- **Best:** W_INSULIN=0.5 → TIR=36.3%, TBR_severe=0.0%, mean=240.0
- **Current RMSE after bias correction:** 116.7 mg/dL — still significant structural mismatch
- **Conclusion:** Cost function tuning can improve TIR marginally but cannot compensate for a 117 RMSE forward model. Achieving TIR≥50% likely requires: (a) online Bergman parameter fitting (augmented UKF), (b) Hovorka forward model in MPC, or (c) a learned dynamics model.

---

## 2026-03-10 — IOB Guard Assessment (Phase 3, Task 1)

### Finding 11: IOB guard replicates the architectural conflict of Session B

- **Baseline compared:** Session D baseline TIR=36.3%, TBR_severe=0.0%, TAR_severe=42.9%, mean=240.0.
- **With IOB guard:** TIR=1.6%, TBR_severe=0.0%, TAR_severe=88.0%, mean=329.2 mg/dL.
- **After removing IOB guard:** TIR recovered to 30.7%.
- **Question 1 (Pharmacokinetic model):** The IOB guard used a simplistic linear decay model over a fixed 120-minute (2 hour) horizon. It was not derived from the Bergman ODE state variables S1 and S2 that physiologically model subcutaneous insulin absorption. Because it was a fixed external heuristic, it reproduced the identical architectural conflict removed in Session B.
- **Question 2 (Dose modification vs hard limit):** The IOB guard actively modified the dose. If its linear IOB projection estimated the patient would reach target, it vetoed the MPC's recommendation by forcing the dose to exactly 0.0 units, entirely overriding the MPC's optimization.
- **Question 3 (Patient application):** The guard was applied universally to all patients. This universally suppressed doses for adults, causing severe hyperglycemia (TIR collapsed to 1.6%) by blocking biologically necessary insulin.
- **Conclusion:** As instructed, the IOB guard was removed because TIR dropped below 30% (to 1.6%). TIR subsequently recovered to 30.7%. R-INT-04 fails without the IOB guard, confirming that Phase 1's parameter scaling alone did not shield pediatric patients from the Digital Twin's mismatch.

---

## 2026-03-10 — Phase 3: p1-Only Augmented UKF

### P1_ADAPTER_PREREQUISITES

1. **`DigitalTwin.last_innovation`**: Did NOT exist. Added as a thin `@property` returning `self.residuals[-1] if self.residuals else 0.0`. This reads from the UKF's existing innovation computation without modifying update equations.
2. **`DigitalTwin.dt`**: Did NOT exist as an attribute. The timestep `dt=5.0` is a parameter default everywhere. Added as a `@property` returning `self._dt = 5.0` set in `__init__`.
3. **`p1` location**: Stored at `self.bergman.p1` (line 45 of `BergmanMinimalModel`). The Bergman ODE reads this directly during integration at line 56: `dG = -self.p1 * (G - self.Gb) ...`. The `set_p1()` setter writes to `self.bergman.p1`.

Additional thin accessors added: `glucose_basal` (→ `self.bergman.Gb`), `R_noise` (→ `self.ukf.R[0,0]`), `state` (→ `self.get_state()`), `innovation_var_ema` (new `__init__` attribute).

### Finding 12: p1 adapter requires conservative tuning due to persistent structural mismatch

- **Problem:** The Bergman-Hovorka structural mismatch produces ~10-20 mg/dL innovations persistently, even at steady state (glucose=120 mg/dL). The standard Augmented UKF assumption (innovations are Gaussian noise) is violated. Without rate limiting, p1 drifts monotonically to the 0.050 upper bound within 50 steps.
- **Solution:** Added three stabilization mechanisms:
  1. **20-step warm-up**: p1 is frozen while the UKF settles.
  2. **Rate limit**: `MAX_P1_STEP = 1e-5` per step to prevent cumulative drift.
  3. **Gate threshold**: Raised from 4.0 to 50.0 mg²/dL² to match the system's actual innovation noise floor.
- **Result:** Check 1 (flat glucose) p1 drift = 0.00029 over 50 steps. Check 2 (hyperglycemia) p1 converges downward (correct direction). Check 3 (bounds) both limits enforced.

### Finding 13: R-L2-04 RMSE unchanged after Phase 3

- **Pre-Phase 3 RMSE:** 209.0 mg/dL
- **Post-Phase 3 RMSE:** 209.5 mg/dL
- **p1 at end of 2-day sim:** 0.02493 (converged downward from 0.028)
- **Direction:** Downward — **expected**. The Bergman model's glucose effectiveness is too high (over-corrects glucose toward basal), so the adapter correctly learns a lower p1. This is consistent with Finding 8 (Bergman under-predicts glucose in hyperglycemia because its p1 drives too-fast self-correction).
- **Why RMSE didn't improve:** The rate-limited adapter is conservative. Over 576 steps (2 days), p1 only moved from 0.028 to 0.0249 — a 10.7% decrease. The RMSE is dominated by the Bergman-Hovorka structural mismatch in insulin dynamics (p2/p3/insulin absorption timing), not glucose effectiveness alone. Augmenting p1 alone cannot close the full 209 mg/dL gap.

### Finding 14: R-INT-04 regressed after IOB guard removal

- **With IOB guard (Phase 1):** R-INT-04 PASSED (all patients tbr_severe=0.0%)
- **Without IOB guard (Phase 3):** R-INT-04 FAILED (patients 1 and 3 have tbr_severe=9.8% and 8.2%)
- **Root cause:** The IOB guard was the *only* component preventing the MPC from accumulating excessive insulin in pediatric patients. The ReachabilityAnalyzer's weight-scaled bounds (Phase 1) are per-step checks — they evaluate each individual dose but do not track cumulative insulin on board. The MPC recommends individually-safe doses that cumulatively cause hypoglycemia.
- **Recommendation:** A pediatric-only IOB guard (activated when `weight_kg < 40`) that uses the Bergman ODE's own state variables instead of a fixed-τ exponential decay would address both the Session B conflict and the pediatric safety requirement.

