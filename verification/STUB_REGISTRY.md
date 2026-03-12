# AEGIS STUB Registry

All placeholder implementations are registered here with descriptions of the correct implementation.

---

## Active STUBs

### STUB_ConceptExtractor
- **Location:** `core/layer1_semantic.py` → `SemanticSensorium.extract_concepts()`
- **Current:** Fuzzy Levenshtein matching against a hardcoded synonym dictionary (`CONCEPT_ONTOLOGY`)
- **Correct:** SLM with constrained decoding producing SNOMED-CT concept identifiers. The model should output structured JSON with SNOMED-CT codes, not freeform strings.
- **SNOMED-CT codes required:** Stress (73595000), Fatigue (84229001), Exercise (256235009), Meal (129007004), Hypoglycemia symptoms (267384006), Hyperglycemia symptoms (80394007), Sleep quality (258158006)

### STUB_SemanticEntropy
- **Location:** `core/layer1_semantic.py` → `SemanticSensorium.compute_semantic_entropy()`
- **Current:** Random-integer jitter of fuzzy matching threshold, Shannon entropy over string-match counts
- **Correct:** Multi-temperature sampling from SLM at T ∈ {0.1, 0.4, 0.7, 1.0, 1.4}, embed K=5 outputs via sentence transformer (e.g., all-MiniLM-L6-v2), cluster in embedding space, compute H_sem = −Σ p(c) log p(c) over cluster distribution

### L1_FIREWALL
- **Location:** `core/pipeline.py` → `AEGISPipeline.step()`, lines 78–106
- **Mechanism:** When `SemanticSensorium.STUB_ACTIVE` is `True`, Layer 3 proxy adjustment is bypassed entirely. L4 receives `l3=None` (glucose state + PK model output only). No input originating from STUB_ConceptExtractor or STUB_SemanticEntropy reaches the bandit.
- **Rationale:** H4 diagnostic confirmed 38.5% dose contamination from stub L1 outputs propagating through L3 proxies into L4 arm selection.
- **Removal condition:** Set `SemanticSensorium.STUB_ACTIVE = False` when real SLM replaces fuzzy matching + perturbation-based entropy.
- **Verified by:** `tests/test_l1_firewall.py` (FW-01 through FW-03)

### ACB_BASELINE_FIX ✅ RESOLVED
- **Location:** `core/pipeline.py` → `AEGISPipeline.step()`, line ~119
- **Was:** `self.layer4.update(action_idx, reward, baseline=None)` — disabled action-centering
- **Fixed:** `self.layer4.update(action_idx, reward, baseline=baseline)` — ACB now learns τ(S_t) not PID-confounded residuals (Eq. 7)
- **Verified by:** `tests/test_l1_firewall.py` (FW-04)

### MPC_CONTROLLER
- **Location:** `core/layer4_decision.py` → `MPCController`, `DecisionEngine`
- **Forward model:** Uses Bergman Minimal Model (L2 Digital Twin), NOT the Hovorka ODE used by the simulator. Structural mismatch: 219 RMSE (R-L2-04).
- **Optimization:** Grid search over 8 discrete dose candidates (not gradient-based). Coarse grid resolution (0.25U min).
- **Bandit arms:** Redefined as MPC confidence multipliers [0.5, 0.75, 1.0, 1.25] instead of PID micro-adjustments.
- **Known limitation:** IOB guard and PLGS suspend still active in `_run_closed_loop` — redundant with MPC's forward prediction but not yet removed (Session B).
- **Removal condition:** Replace Bergman forward model with patient-specific Hovorka model when available.

### STUB_REACHABILITY (Potential)
- **Location:** `core/layer5_safety.py` → `ReachabilityAnalyzer`
- **Current:** Hardcoded `max_glucose_drop_per_min = 1.5`
- **Correct:** Sample N_pop patient parameter vectors from Bergman Minimal Model parameter distribution (UVA/Padova literature), simulate one step forward per vector, use 99th percentile of max glucose change as bound
- **Status:** Will be assessed in Session 5

### ~~STUB_LAMBDA_CALIBRATION~~ ✅ RESOLVED

**File:** `core/layer4_decision.py` — `LambdaCalibrator` class

**Resolution:** Added `LambdaCalibrator` using `sklearn.isotonic.IsotonicRegression`.
- Collects `(DT_prediction, actual_glucose)` pairs from pipeline
- After 30 observations, fits isotonic regression: `|error| → λ` (monotone decreasing)
- Low error → high λ (trust DT), high error → low λ
- Re-fits every 20 new observations
- Falls back to L4_CF_LAMBDA_DIRECT until calibrated

**Wiring:**
- `pipeline.py:step()` records `(pred_state[0], glucose)` into calibrator
- `CTS.counterfactual_update()` queries calibrator for λ instead of hardcoded constants (Potential)
- **Location:** `core/layer4_decision.py` → `CounterfactualThompsonSampling.counterfactual_update()`
- **Current:** `λ` is hardcoded (`L4_CF_LAMBDA_POPULATION=0.3`, `L4_CF_LAMBDA_DIRECT=0.5`)
- **Correct:** `λ` must come from calibrated isotonic regression on historical prediction accuracy. `Ŷ` must come from Digital Twin forward simulation.
- **Status:** Will be assessed in Session 4
