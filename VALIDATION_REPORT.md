# AEGIS 3.0 Robust Validation Report

**Generated:** 2026-02-25T16:17:41.868477
**Runtime:** 392.3 seconds
**Suite Version:** Robust Validation v1.0

---

## Summary

| Category | Passed | Total | Rate |
|----------|--------|-------|------|
| **Nominal** | 18 | 20 | 90% |
| **Robustness** | 16 | 19 | 84% |
| **Adversarial** | 4 | 4 | 100% |
| **Overall** | **38** | **43** | **88.4%** |

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

### Layer 2: Digital Twin (6/6)

| ID | Name | Type | Status |
|-----|------|------|--------|
| R-L2-01 | State Estimation Quality | nominal | ✅ PASS |
| R-L2-02 | Constraint Satisfaction | nominal | ✅ PASS |
| R-L2-03 | UKF Q Adaptation Direction | nominal | ✅ PASS |
| R-L2-04 | Model Mismatch (Hovorka vs Bergman) | robust | ✅ PASS |
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

### Integration Tests (1/6)

| ID | Name | Type | Status |
|-----|------|------|--------|
| R-INT-01 | 7-Day Pipeline Execution | nominal | ❌ FAIL |
| R-INT-02 | Clinical Metrics (TIR/TBR) | nominal | ❌ FAIL |
| R-INT-03 | Cross-Layer Communication | nominal | ✅ PASS |
| R-INT-04 | Multi-Patient Cohort (5 patients) | robust | ❌ FAIL |
| R-INT-05 | Recovery from Safety Events | robust | ❌ FAIL |
| R-INT-06 | Ablation — Each Layer Matters | robust | ❌ FAIL |

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

- mean_ambiguous_entropy: `0.2942`
- mean_clear_entropy: `0.0816`
- entropy_difference: `0.2125`

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
- entropy_known: `-0.0000`
- entropy_unknown: `-0.0000`

**R-L1-06: Ambiguous Proxy Boundaries** ✅

- all_processed: `True`
- valid_proxy_types: `True`

### Layer 2: Digital Twin

**R-L2-01: State Estimation Quality** ✅

- estimate_rmse: `10.3251`
- raw_rmse: `8.0893`
- improvement_ratio: `1.2764`

**R-L2-02: Constraint Satisfaction** ✅

- violations: `0`
- total: `500`
- violation_rate: `0.0000`

**R-L2-03: UKF Q Adaptation Direction** ✅

- q_base: `25.0000`
- q_stable: `25.0000`
- q_after_mismatch: `128.0704`
- q_increased: `True`

**R-L2-04: Model Mismatch (Hovorka vs Bergman)** ✅

- mismatch_rmse: `110.9060`

**R-L2-05: Unannounced Meal Detection (Innovation)** ✅

- baseline_innovation: `21.2051`
- post_meal_innovation: `37.3212`
- innovation_ratio: `1.7600`
- detected: `True`

**R-L2-06: Sensor Noise Filtering** ✅

- filtered_rmse: `19.9126`
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

- mean_bias_reduction: `0.0160`
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

- mean_ms: `0.0475`
- p99_ms: `0.0665`
- max_ms: `0.3051`

**R-L5-03: STL Specification Satisfaction** ✅

- satisfaction_rate: `1.0000`

**R-L5-04: Cold Start Relaxation Schedule** ✅

- alpha_day0: `0.0122`
- alpha_day14: `0.0362`
- alpha_day30: `0.0452`
- alpha_day44: `0.0484`
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

**R-INT-01: 7-Day Pipeline Execution** ❌

- steps_executed: `2016`
- steps_expected: `2016`
- completion_rate: `1.0000`
- days_covered: `7`
- all_layers_active: `False`

**R-INT-02: Clinical Metrics (TIR/TBR)** ❌

- tir: `5.8036`
- tbr: `2.7778`
- tbr_severe: `1.8353`
- tar: `91.4187`
- tar_severe: `86.5079`
- mean: `348.6294`
- std: `91.3583`
- cv: `26.2050`

**R-INT-03: Cross-Layer Communication** ✅

- l2_predictions_complete: `True`
- l5_blocks_correct: `True`
- l5_total_checks: `2016`
- l5_tier_counts:
  - 1: `37`
  - 2: `32`
  - 3: `18`
  - 4: `1929`

**R-INT-04: Multi-Patient Cohort (5 patients)** ❌

- all_safe: `False`
- patient_summaries: `[{'patient_id': 0, 'tir': 12.847222222222221, 'tbr_severe': 4.282407407407407}, {'patient_id': 1, 'tir': 8.680555555555555, 'tbr_severe': 61.68981481481482}, {'patient_id': 2, 'tir': 22.22222222222222, 'tbr_severe': 26.504629629629626}, {'patient_id': 3, 'tir': 14.583333333333334, 'tbr_severe': 23.958333333333336}, {'patient_id': 4, 'tir': 2.5462962962962963, 'tbr_severe': 24.76851851851852}]`

**R-INT-05: Recovery from Safety Events** ❌

- emergency_triggered: `False`
- recovered: `False`
- post_recovery_tiers: `['OK', 'STL_BLOCKED', 'OK', 'OK', 'OK', 'OK', 'STL_BLOCKED', 'STL_BLOCKED', 'STL_BLOCKED', 'OK']`

**R-INT-06: Ablation — Each Layer Matters** ❌

- L1_concepts_extracted: `True`
- L2_predictions_complete: `True`
- L4_unique_actions: `0`
- L5_total_checks: `576`

### Adversarial Tests

**R-ADV-01: Worst-Case Meal Timing** ✅

- severe_hypo_readings: `5`
- l5_safety_interventions: `5`
- total_steps: `576`

**R-ADV-02: Sensor Bias Attack (+50 mg/dL)** ✅

- safety_violations: `0`
- total_steps: `288`

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

- Timestamp: 2026-02-25T16:17:41.868477
- Runtime: 392.3s

## Failed Tests Detail

### R-INT-01: 7-Day Pipeline Execution

- Type: nominal
- Metrics: {
  "steps_executed": 2016,
  "steps_expected": 2016,
  "completion_rate": 1.0,
  "days_covered": 7,
  "all_layers_active": false
}
- Threshold: {
  "completion": 1.0
}

### R-INT-02: Clinical Metrics (TIR/TBR)

- Type: nominal
- Metrics: {
  "tir": 5.803571428571429,
  "tbr": 2.7777777777777777,
  "tbr_severe": 1.8353174603174605,
  "tar": 91.41865079365078,
  "tar_severe": 86.5079365079365,
  "mean": 348.6294046950156,
  "std": 91.3582904836815,
  "cv": 26.204987087535724
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
      "tir": 12.847222222222221,
      "tbr_severe": 4.282407407407407
    },
    {
      "patient_id": 1,
      "tir": 8.680555555555555,
      "tbr_severe": 61.68981481481482
    },
    {
      "patient_id": 2,
      "tir": 22.22222222222222,
      "tbr_severe": 26.504629629629626
    },
    {
      "patient_id": 3,
      "tir": 14.583333333333334,
      "tbr_severe": 23.958333333333336
    },
    {
      "patient_id": 4,
      "tir": 2.5462962962962963,
      "tbr_severe": 24.76851851851852
    }
  ]
}
- Threshold: {
  "all_safe": true
}

### R-INT-05: Recovery from Safety Events

- Type: robust
- Metrics: {
  "emergency_triggered": false,
  "recovered": false,
  "post_recovery_tiers": [
    "OK",
    "STL_BLOCKED",
    "OK",
    "OK",
    "OK",
    "OK",
    "STL_BLOCKED",
    "STL_BLOCKED",
    "STL_BLOCKED",
    "OK"
  ]
}
- Threshold: {
  "recovery": true
}

### R-INT-06: Ablation — Each Layer Matters

- Type: robust
- Metrics: {
  "L1_concepts_extracted": true,
  "L2_predictions_complete": true,
  "L4_unique_actions": 0,
  "L5_total_checks": 576
}
- Threshold: {
  "all_layers_active": true
}

