# AEGIS 3.0 Robust Validation Report

**Generated:** 2026-03-10T18:16:06.023669
**Runtime:** 436.0 seconds
**Suite Version:** Robust Validation v1.0

---

## Summary

| Category | Passed | Total | Rate |
|----------|--------|-------|------|
| **Nominal** | 19 | 20 | 95% |
| **Robustness** | 17 | 19 | 89% |
| **Adversarial** | 4 | 4 | 100% |
| **Overall** | **40** | **43** | **93.0%** |

---

## Per-Layer Results

### Layer 1: Semantic Sensorium (6/6)

| ID | Name | Type | Status |
|-----|------|------|--------|
| R-L1-01 | Concept Extraction (Nominal) | nominal | ✅ PASS |
| R-L1-02 | Semantic Entropy Calibration | nominal | ✅ PASS |
| R-L1-03 | Proxy Classification | nominal | ✅ PASS |
| R-L1-04 | Noisy/Informal Text | robust | ✅ PASS |
| R-L1-05 | Out-of-Vocabulary Concepts | robust | ✅ PASS |
| R-L1-06 | Ambiguous Proxy Boundaries | robust | ✅ PASS |

### Layer 2: Digital Twin (5/6)

| ID | Name | Type | Status |
|-----|------|------|--------|
| R-L2-01 | State Estimation Quality | nominal | ✅ PASS |
| R-L2-02 | Constraint Satisfaction | nominal | ✅ PASS |
| R-L2-03 | UKF Q Adaptation Direction | nominal | ✅ PASS |
| R-L2-04 | Model Mismatch (Hovorka vs Bergman) | robust | ❌ FAIL |
| R-L2-05 | Unannounced Meal Detection (Innovation) | robust | ✅ PASS |
| R-L2-06 | Sensor Noise Filtering | robust | ✅ PASS |

### Layer 3: Causal Inference (8/8)

| ID | Name | Type | Status |
|-----|------|------|--------|
| R-L3-01 | Harmonic Effect Recovery | nominal | ✅ PASS |
| R-L3-02 | Double Robustness (AIPW) | nominal | ✅ PASS |
| R-L3-03 | Proximal Bias Reduction | nominal | ✅ PASS |
| R-L3-04 | Anytime Validity (Confidence Sequences) | nominal | ✅ PASS |
| R-L3-05 | Non-Linear Confounding | robust | ✅ PASS |
| R-L3-06 | Proxy Violation Detection | robust | ✅ PASS |
| R-L3-07 | Time-Varying Confounding | robust | ✅ PASS |
| R-L3-08 | Small Sample (T=200) | robust | ✅ PASS |

### Layer 4: Decision Engine (6/6)

| ID | Name | Type | Status |
|-----|------|------|--------|
| R-L4-01 | ACB Variance Reduction | nominal | ✅ PASS |
| R-L4-02 | Regret Scaling | nominal | ✅ PASS |
| R-L4-03 | Posterior Collapse Prevention (CTS) | nominal | ✅ PASS |
| R-L4-04 | Biased Digital Twin | robust | ✅ PASS |
| R-L4-05 | Non-Stationary Rewards | robust | ✅ PASS |
| R-L4-06 | High Blocking Rate (>50%) | robust | ✅ PASS |

### Layer 5: Safety Supervisor (7/7)

| ID | Name | Type | Status |
|-----|------|------|--------|
| R-L5-01 | Tier Priority Classification | nominal | ✅ PASS |
| R-L5-02 | Reflex Response Time | nominal | ✅ PASS |
| R-L5-03 | STL Specification Satisfaction | nominal | ✅ PASS |
| R-L5-04 | Cold Start Relaxation Schedule | nominal | ✅ PASS |
| R-L5-05 | Sensor Dropout Safety | robust | ✅ PASS |
| R-L5-06 | Cascading Layer Failures | robust | ✅ PASS |
| R-L5-07 | Rapid Glucose Transition | robust | ✅ PASS |

### Integration Tests (4/6)

| ID | Name | Type | Status |
|-----|------|------|--------|
| R-INT-01 | 7-Day Pipeline Execution | nominal | ✅ PASS |
| R-INT-02 | Clinical Metrics (TIR/TBR) | nominal | ❌ FAIL |
| R-INT-03 | Cross-Layer Communication | nominal | ✅ PASS |
| R-INT-04 | Multi-Patient Cohort (5 patients) | robust | ❌ FAIL |
| R-INT-05 | Recovery from Safety Events | robust | ✅ PASS |
| R-INT-06 | Ablation — Each Layer Matters | robust | ✅ PASS |

### Adversarial Tests (4/4)

| ID | Name | Type | Status |
|-----|------|------|--------|
| R-ADV-01 | Worst-Case Meal Timing | adversarial | ✅ PASS |
| R-ADV-02 | Sensor Bias Attack (+50 mg/dL) | adversarial | ✅ PASS |
| R-ADV-03 | Rapid-Fire State Swings | adversarial | ✅ PASS |
| R-ADV-04 | Maximum Stress Scenario | adversarial | ✅ PASS |

---

## Detailed Metrics

### Layer 1: Semantic Sensorium

**R-L1-01: Concept Extraction (Nominal)** ✅

- precision: `0.8125`
- recall: `0.9286`
- f1: `0.8667`

**R-L1-02: Semantic Entropy Calibration** ✅

- mean_ambiguous_entropy: `0.4420`
- mean_clear_entropy: `0.4072`
- entropy_difference: `0.0347`

**R-L1-03: Proxy Classification** ✅

- z_precision: `1.0000`
- z_recall: `1.0000`
- w_recall: `1.0000`

**R-L1-04: Noisy/Informal Text** ✅

- precision: `0.8571`
- recall: `0.8571`
- f1: `0.8571`

**R-L1-05: Out-of-Vocabulary Concepts** ✅

- known_concepts: `4`
- unknown_concepts: `0`
- entropy_known: `0.8979`
- entropy_unknown: `-0.0000`

**R-L1-06: Ambiguous Proxy Boundaries** ✅

- all_processed: `True`
- valid_proxy_types: `True`
- any_concepts_found: `True`
- proxy_trace: `[{'text': 'Feeling a bit tense, maybe from the coffee', 'z_proxy': True, 'w_proxy': 0.0, 'concepts': ['stress', 'mood_positive'], 'entropy': -1.000088900581841e-12}, {'text': "Didn't sleep great, busy day ahead", 'z_proxy': False, 'w_proxy': 1.0, 'concepts': ['fatigue', 'sleep_quality', 'mood_positive'], 'entropy': -1.000088900581841e-12}, {'text': 'Tired from the workout yesterday', 'z_proxy': True, 'w_proxy': 0.5, 'concepts': ['stress', 'fatigue', 'exercise', 'work_stressor'], 'entropy': 0.5004024235361879}]`

### Layer 2: Digital Twin

**R-L2-01: State Estimation Quality** ✅

- estimate_rmse: `9.8761`
- raw_rmse: `8.0893`
- improvement_ratio: `1.2209`

**R-L2-02: Constraint Satisfaction** ✅

- violations: `0`
- total: `500`
- violation_rate: `0.0000`

**R-L2-03: UKF Q Adaptation Direction** ✅

- q_base: `25.0000`
- q_stable: `0.0001`
- q_after_mismatch: `250.0000`
- q_increased: `True`
- q_diag_full: `[249.999999996691, 0.0005687327489880722, 2.0859597229237616]`
- adaptation_steps: `91`
- innovation_count: `100`
- multi_step_adaptation: `True`

**R-L2-04: Model Mismatch (Hovorka vs Bergman)** ❌

- mismatch_rmse: `209.5375`

**R-L2-05: Unannounced Meal Detection (Innovation)** ✅

- baseline_innovation: `20.8497`
- post_meal_innovation: `36.8365`
- innovation_ratio: `1.7668`
- detected: `True`

**R-L2-06: Sensor Noise Filtering** ✅

- filtered_rmse: `19.6710`
- raw_noisy_rmse: `21.9984`
- filter_helps: `True`

### Layer 3: Causal Inference

**R-L3-01: Harmonic Effect Recovery** ✅

- psi0_mean: `0.5435`
- psi0_rmse: `0.0489`
- harmonic_rmse: `0.0517`

**R-L3-02: Double Robustness (AIPW)** ✅

- both_correct:
  - mean: `0.5020`
  - bias: `0.0020`
- outcome_only:
  - mean: `0.5017`
  - bias: `0.0017`
- propensity_only:
  - mean: `0.5042`
  - bias: `0.0042`
- both_wrong:
  - mean: `0.7067`
  - bias: `0.2067`

**R-L3-03: Proximal Bias Reduction** ✅

- mean_bias_reduction: `0.0161`
- pct_positive: `0.7800`

**R-L3-04: Anytime Validity (Confidence Sequences)** ✅

- coverage_rate: `1.0000`

**R-L3-05: Non-Linear Confounding** ✅

- mean_bias: `0.0346`

**R-L3-06: Proxy Violation Detection** ✅

- detection_rate: `0.8333`

**R-L3-07: Time-Varying Confounding** ✅

- mean_bias: `0.0421`

**R-L3-08: Small Sample (T=200)** ✅

- rmse: `0.0947`

### Layer 4: Decision Engine

**R-L4-01: ACB Variance Reduction** ✅

- mean_variance_ratio: `0.0103`

**R-L4-02: Regret Scaling** ✅

- log_log_slope: `0.3490`
- final_regret: `187.8000`

**R-L4-03: Posterior Collapse Prevention (CTS)** ✅

- initial_var: `1.0000`
- final_var: `0.0323`
- var_ratio: `0.0323`

**R-L4-04: Biased Digital Twin** ✅

- estimated_mean: `0.0000`
- true_mean: `0.6000`
- mean_error: `0.6000`
- post_var: `0.0164`

**R-L4-05: Non-Stationary Rewards** ✅

- best_arm_rate_late: `0.4040`

**R-L4-06: High Blocking Rate (>50%)** ✅

- blocking_rate: `0.6130`
- posteriors_narrowed: `True`

### Layer 5: Safety Supervisor

**R-L5-01: Tier Priority Classification** ✅

- accuracy: `1.0000`
- correct: `6`
- total: `6`

**R-L5-02: Reflex Response Time** ✅

- mean_ms: `0.0426`
- p99_ms: `0.0682`
- max_ms: `0.8660`

**R-L5-03: STL Specification Satisfaction** ✅

- satisfaction_rate: `1.0000`

**R-L5-04: Cold Start Relaxation Schedule** ✅

- alpha_day0: `0.0100`
- alpha_day14: `0.0353`
- alpha_day30: `0.0450`
- alpha_day44: `0.0483`
- day1_correct: `True`
- day30_correct: `True`
- monotonic_trend: `True`

**R-L5-05: Sensor Dropout Safety** ✅

- safety_maintained: `True`
- recovery_after_dropout: `True`

**R-L5-06: Cascading Layer Failures** ✅

- all_safe: `True`

**R-L5-07: Rapid Glucose Transition** ✅

- all_safe: `True`

### Integration Tests

**R-INT-01: 7-Day Pipeline Execution** ✅

- steps_executed: `2016`
- steps_expected: `2016`
- completion_rate: `1.0000`
- days_covered: `1`
- all_layers_active: `True`

**R-INT-02: Clinical Metrics (TIR/TBR)** ❌

- tir: `41.2698`
- tbr: `0.7937`
- tbr_severe: `0.0000`
- tar: `57.9365`
- tar_severe: `35.7143`
- mean: `224.0034`
- std: `103.6360`
- cv: `46.2654`

**R-INT-03: Cross-Layer Communication** ✅

- l2_predictions_complete: `True`
- l5_blocks_correct: `True`
- l5_total_checks: `2016`
- l5_tier_counts:
  - 1: `0`
  - 2: `38`
  - 3: `11`
  - 4: `1967`

**R-INT-04: Multi-Patient Cohort (5 patients)** ❌

- all_safe: `False`
- patient_summaries: `[{'patient_id': 0, 'tir': 36.80555555555556, 'tbr_severe': 0.0}, {'patient_id': 1, 'tir': 44.21296296296296, 'tbr_severe': 9.837962962962964}, {'patient_id': 2, 'tir': 30.208333333333332, 'tbr_severe': 0.0}, {'patient_id': 3, 'tir': 35.06944444444444, 'tbr_severe': 8.217592592592593}, {'patient_id': 4, 'tir': 42.592592592592595, 'tbr_severe': 0.0}]`

**R-INT-05: Recovery from Safety Events** ✅

- emergency_triggered: `True`
- recovered: `True`
- post_recovery_tiers: `['OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK']`

**R-INT-06: Ablation — Each Layer Matters** ✅

- L1_concepts_extracted: `True`
- L2_predictions_complete: `True`
- L4_unique_actions: `10`
- L5_total_checks: `576`

### Adversarial Tests

**R-ADV-01: Worst-Case Meal Timing** ✅

- severe_hypo_readings: `5`
- l5_safety_interventions: `5`
- total_steps: `576`

**R-ADV-02: Sensor Bias Attack (+50 mg/dL)** ✅

- safety_violations: `0`
- dose_exceeded_cap: `0`
- total_steps: `288`
- mean_final_action: `0.0269`
- max_final_action: `1.2500`

**R-ADV-03: Rapid-Fire State Swings** ✅

- safety_maintained: `True`

**R-ADV-04: Maximum Stress Scenario** ✅

- safety_maintained: `True`

---

## Validation Notes

1. **Independent Data Generation:** Patient data generated by Hovorka model
   (structurally different from Bergman used by Digital Twin)
2. **Causal Structure:** Proxy variables Z, W satisfy conditional independence
   by construction in the data generating process
3. **Cold Start:** Alpha relaxation verified with real data accumulation
4. **No Mocks:** All layers use real implementations (PyTorch MLP, UKF,
   KernelRidge, etc.)
5. **Reproducibility:** All tests use fixed seeds from config.py

## Test Environment

- Timestamp: 2026-03-10T18:16:06.023669
- Runtime: 436.0s
- Git Hash: ff5ebc3
- Python: 3.13.3
- Platform: macOS-26.2-arm64-arm-64bit-Mach-O
- NumPy: 2.3.0
- PyTorch: 2.9.0
- SciPy: 1.16.2

## Active STUBs

- `layer4_decision.py:9: - Isotonic-regression λ calibration (replaces STUB_LAMBDA_CALIBRATION)`
- `layer4_decision.py:43: Replaces STUB_LAMBDA_CALIBRATION.`
- `layer1_semantic.py:8: (STUB_SemanticEntropy: production version must use true SLM`
- `layer1_semantic.py:9: multi-temperature sampling. See STUB_REGISTRY.md)`
- `layer1_semantic.py:10: - Concept extraction explicitly labeled as STUB_ConceptExtractor.`
- `layer1_semantic.py:18: - SNOMED-CT mapping: Concept IDs documented in STUB_REGISTRY.md`
- `layer1_semantic.py:22: - Presenting fuzzy matching as a semantic model without STUB label`
- `layer1_semantic.py:151: STUB_SemanticEntropy: Uses perturbation-based sampling (word dropout,`
- `layer1_semantic.py:199: STUB_SemanticEntropy: Production version replaces this with`
- `layer1_semantic.py:315: 1. STUB_ConceptExtractor: Ontology-constrained concept extraction`
- `layer1_semantic.py:324: - Concept extraction: Section III-A (STUB — see STUB_REGISTRY.md)`
- `layer1_semantic.py:335: STUB_ACTIVE = True`
- `layer1_semantic.py:368: STUB_ConceptExtractor: Extract SNOMED-CT concepts from patient text`
- `layer1_semantic.py:371: STUB: Production version must use an SLM with constrained decoding`
- `layer1_semantic.py:374: See STUB_REGISTRY.md for the correct implementation specification.`
- `layer1_semantic.py:422: but is not used in the current STUB implementation. Production`
- `layer1_semantic.py:425: STUB_SemanticEntropy: Uses perturbation-based sampling instead`
- `layer1_semantic.py:433: temperatures: ignored (STUB — production uses T ∈ {0.1, 0.4, 0.7, 1.0, 1.4})`
- `layer1_semantic.py:492: 'stub_active': self.STUB_ACTIVE,`


## Failed Tests Detail

### R-L2-04: Model Mismatch (Hovorka vs Bergman)

- Type: robust
- Metrics: {
  "mismatch_rmse": 209.537531533732
}
- Threshold: {
  "rmse_max": 200.0
}

### R-INT-02: Clinical Metrics (TIR/TBR)

- Type: nominal
- Metrics: {
  "tir": 41.269841269841265,
  "tbr": 0.7936507936507936,
  "tbr_severe": 0.0,
  "tar": 57.936507936507944,
  "tar_severe": 35.714285714285715,
  "mean": 224.00337529517196,
  "std": 103.63596313808131,
  "cv": 46.26535783289825
}
- Threshold: {
  "tir_min": 70.0,
  "tbr_severe_max": 0.0
}

### R-INT-04: Multi-Patient Cohort (5 patients)

- Type: robust
- Metrics: {
  "all_safe": false,
  "patient_summaries": [
    {
      "patient_id": 0,
      "tir": 36.80555555555556,
      "tbr_severe": 0.0
    },
    {
      "patient_id": 1,
      "tir": 44.21296296296296,
      "tbr_severe": 9.837962962962964
    },
    {
      "patient_id": 2,
      "tir": 30.208333333333332,
      "tbr_severe": 0.0
    },
    {
      "patient_id": 3,
      "tir": 35.06944444444444,
      "tbr_severe": 8.217592592592593
    },
    {
      "patient_id": 4,
      "tir": 42.592592592592595,
      "tbr_severe": 0.0
    }
  ]
}
- Threshold: {
  "all_safe": true
}

