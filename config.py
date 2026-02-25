"""
AEGIS 3.0 Robust Validation Suite — Single Source of Truth Configuration.

All thresholds, seeds, and parameters defined here.
Referenced by both tests and report generator — no duplication.
"""

import numpy as np

# ============================================================
# Global
# ============================================================
RANDOM_SEED = 42
N_MONTE_CARLO = 100          # Standard Monte Carlo iterations
N_MC_SMALL = 50              # For computationally expensive tests
CONFIDENCE_LEVEL = 0.95

# ============================================================
# Simulator Parameters (Hovorka Model — independent from Bergman)
# ============================================================
SIM_DT_MINUTES = 5           # 5-minute sampling interval
SIM_DAYS = 7                 # 7-day simulation duration
SIM_STEPS_PER_DAY = 288      # 24*60/5
SIM_TOTAL_STEPS = SIM_DAYS * SIM_STEPS_PER_DAY  # 2016

# Physiological bounds (shared across simulator and core)
GLUCOSE_MIN = 20.0           # mg/dL — absolute physiological floor
GLUCOSE_MAX = 600.0          # mg/dL — absolute physiological ceiling
INSULIN_MIN = 0.0            # mU/L
INSULIN_MAX = 500.0          # mU/L
REMOTE_INSULIN_MIN = 0.0
REMOTE_INSULIN_MAX = 0.1     # 1/min

# ============================================================
# Layer 1: Semantic Sensorium Thresholds
# ============================================================
L1_NOMINAL = {
    'extraction_f1_min': 0.80,
    'extraction_precision_min': 0.80,
    'extraction_recall_min': 0.75,
    'entropy_spearman_min': 0.60,
    'entropy_auc_min': 0.80,
    'proxy_precision_min': 0.80,
    'proxy_recall_min': 0.75,
}
L1_ROBUST = {
    'noisy_f1_min': 0.65,
    'oov_entropy_increase_min': 0.3,   # Entropy should rise by ≥0.3 for OOV
    'ambiguous_proxy_f1_min': 0.55,
}
L1_ENTROPY_THRESHOLD = 0.5          # HITL trigger threshold

# ============================================================
# Layer 2: Digital Twin Thresholds
# ============================================================
L2_NOMINAL = {
    'ude_rmse_ratio_max': 1.30,      # UDE ≤ 130% of mechanistic
    'constraint_violation_rate': 0.02,# ≤2% (physiological clamping may lag)
    'ukf_q_adapt_min': 1.5,          # Q must adapt ≥1.5x
}
L2_ROBUST = {
    'mismatch_rmse_max': 200.0,      # mg/dL — Hovorka→Bergman structural mismatch
    'unannounced_adapt_minutes': 30,  # Must adapt within 30 min
    'noisy_state_rmse_max': 80.0,    # Under ±20 mg/dL sensor noise (filter helps)
}

# Neural residual (real PyTorch MLP)
L2_NN_HIDDEN = 64
L2_NN_LAYERS = 2
L2_NN_LR = 1e-3
L2_NN_EPOCHS = 300

# UKF parameters
L2_UKF_ALPHA = 1.0
L2_UKF_BETA = 2.0
L2_UKF_KAPPA = 0.0
L2_UKF_Q_ADAPT_RATE = 0.1
L2_UKF_Q_MAX_RATIO = 10.0

# RBPF parameters
L2_RBPF_N_PARTICLES = 500
L2_RBPF_RESAMPLE_THRESHOLD = 0.5    # ESS/N threshold

# Filter switching
L2_SWITCH_SHAPIRO_ALPHA = 0.05
L2_SWITCH_BIMODALITY_THRESHOLD = 0.555

# ============================================================
# Layer 3: Causal Inference Thresholds
# ============================================================
L3_NOMINAL = {
    'gest_psi0_rmse_max': 0.10,
    'gest_harmonic_rmse_max': 0.15,
    'gest_peak_error_hours_max': 1.0,
    'dr_bias_both_correct_max': 0.05,
    'dr_bias_one_correct_max': 0.10,
    'proximal_bias_reduction_min': 0.30,  # ≥30%
    'anytime_coverage_min': 0.93,
}
L3_ROBUST = {
    'nonlinear_bias_max': 0.20,           # Graceful under nonlinear confounding
    'proxy_violation_detection_rate': 0.70, # Should detect ≥70% of violations
    'timevarying_bias_max': 0.15,
    'small_sample_rmse_max': 0.25,        # T=200 → relaxed
}
L3_N_HARMONICS = 3
L3_RANDOMIZATION_RANGE = (0.3, 0.7)

# ============================================================
# Layer 4: Decision Engine Thresholds
# ============================================================
L4_NOMINAL = {
    'acb_variance_ratio_max': 1.0,
    'regret_slope_min': 0.3,
    'regret_slope_max': 0.7,
    'cts_posterior_ratio_max': 1.0,
}
L4_ROBUST = {
    'biased_twin_regret_ratio_max': 2.0,  # ≤2x optimal regret
    'nonstationary_adaptation_steps': 200, # Must adapt within 200 steps
    'high_block_functionality_min': 0.80,  # ≥80% actions still productive
}
L4_EPSILON_INIT = 1.0
L4_EPSILON_DECAY = 0.995
L4_EPSILON_MIN = 0.05
L4_PRIOR_VAR = 1.0
L4_NOISE_VAR = 1.0
L4_CF_LAMBDA_POPULATION = 0.3
L4_CF_LAMBDA_DIRECT = 0.5

# ============================================================
# Layer 5: Safety Supervisor Thresholds
# ============================================================
L5_NOMINAL = {
    'tier_accuracy': 1.0,
    'reflex_latency_ms_max': 100.0,
    'stl_satisfaction_min': 0.95,
    'cold_start_tolerance': 0.005,
}
L5_ROBUST = {
    'dropout_safety_maintained': True,
    'cascade_failure_safety': True,
    'rapid_change_handled': True,
}

# Safety tiers
L5_HYPO_SEVERE = 54.0      # mg/dL
L5_HYPO_MILD = 70.0
L5_HYPER_MILD = 180.0
L5_HYPER_SEVERE = 250.0
L5_HYPER_CRITICAL = 400.0
L5_MAX_BOLUS = 10.0         # units

# STL specifications
L5_STL_RECOVERY_WINDOW_MIN = 30  # minutes
L5_STL_RECOVERY_THRESHOLD = 80   # mg/dL

# Cold start
L5_COLD_ALPHA_STRICT = 0.01
L5_COLD_ALPHA_STANDARD = 0.05
L5_COLD_TAU_DAYS = 14

# Seldonian
L5_SELDONIAN_DELTA = 0.01   # P(G<54) ≤ 1%

# ============================================================
# Integration Thresholds
# ============================================================
INT_NOMINAL = {
    'tir_min': 70.0,           # % in 70–180 mg/dL
    'tbr_severe_max': 0.0,     # % < 54 mg/dL — must be zero
    'tar_severe_max': 5.0,     # % > 250 mg/dL
    'seldonian_violations_max': 0.01,
}
INT_ROBUST = {
    'multi_patient_all_safe': True,
    'recovery_within_steps': 6, # 30 min
}

# Adversarial — safety is the ONLY criterion
ADV_SAFETY = {
    'tbr_severe_max': 0.0,     # Zero severe hypos, always
}
