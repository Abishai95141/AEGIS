# AEGIS 3.0: Research Context & System State

This document serves as a self-contained briefing for a researcher to understand the architecture, current implementation state, identified problems, and the bounded scope of the AEGIS N-of-1 precision medicine pipeline. **The explicit constraint is to accurately document the current system state, performance metrics, and architecture without proposing solutions.**

---

## 1. Introduction & Project Scope

AEGIS is a five-layer pipeline designed for safe causal N-of-1 precision medicine, specifically evaluated on a simulated Type 1 Diabetes (T1D) task. The core philosophy is to safely personalize insulin delivery by estimating individualized causal treatment effects dynamically over time, supervised by a multi-tier formal safety layer.

The original claim of achieving 73.1% Time In Range (TIR) with zero severe hypoglycemia was found irreproducible with the correct formal architecture. The project is currently operating under a revised scope focused on rigorous diagnostics, honest capability measurement, and targeted stabilization of the architectural components before returning to optimization.

## 2. Architecture Overview

### Simulation Environment (The Ground Truth)
The project utilizes the **Hovorka Patient Simulator** (`simulator/patient.py`) as the ground truth environment. This presents a deliberate structural difference from the pipeline’s internal awareness:
*   The Hovorka simulator uses a complex **10-compartment ODE** (2 subcutaneous insulin, 3 insulin action, 2 blood glucose, 2 gut absorption, 1 plasma insulin).
*   It also generates causal text proxies (e.g., stress and fatigue symptoms) corresponding directly to the underlying hidden state variables.

### The 5-Layer Pipeline
The pipeline (`core/pipeline.py`) processes glucose, carb inputs, and natural language text (patient notes) sequentially through 5 modular layers:

1.  **Layer 1: Semantic Sensorium (`core/layer1_semantic.py`)** 
    *   **Purpose:** Extract concepts and semantic entropy from natural language text.
    *   **Current State:** Primarily acting as a **STUB**. It utilizes fuzzy Levenshtein matching against a dictionary instead of an SLM, and a random-integer jitter for entropy. A firewall currently protects downstream bandit processing from STUB outputs to prevent contamination.
2.  **Layer 2: Digital Twin (`core/layer2_digital_twin.py`)**
    *   **Purpose:** A forward-predictive patient model utilizing the **Bergman Minimal Model** (3-compartment ODE: G, X, I).
    *   **Current State:** Uses an Adaptive Constrained Unscented Kalman Filter (AC-UKF) and a Rao-Blackwellized Particle Filter (RBPF) for state estimation. A PyTorch-based Multilayer Perceptron (MLP) acts as a neural residual predictor. The structural mismatch between this Bergman ODE and the true Hovorka ODE is the primary system bottleneck.
3.  **Layer 3: Causal Inference Engine (`core/layer3_causal.py`)**
    *   **Purpose:** Estimates the individualized treatment effect via Proximal/Harmonic G-estimation and limits the scope of randomized exploration mathematically.
    *   **Current State:** Highly stable. Uses kernel ridge regression for bridge functions and Anytime-Valid Confidence Sequences for tight estimation bounds. Validated mathematically correct.
4.  **Layer 4: Decision Engine (`core/layer4_decision.py`)**
    *   **Purpose:** Optimizes the insulin dose decision.
    *   **Current State:** Substituted previous PID+IOB implementations with a **Model Predictive Controller (MPC)** performing a discrete 8-candidate Grid Search utilizing Layer 2's Bergman ODE as the forward dynamics model. Relies on Action-Centered Bandits (ACB) and Counterfactual Thompson Sampling (CTS) with isotonic-regression calibrating the counterfactual weight parameter $\lambda$.
5.  **Layer 5: Safety Supervisor (`core/layer5_safety.py`)**
    *   **Purpose:** A 3-Tier Simplex safety boundary that has final veto power over all doses.
    *   **Current State:** Tier 1 (Reflex threshold logic), Tier 2 (Signal Temporal Logic - STL and conservative reachability utilizing population parameters), and Tier 3 (Seldonian UCB probabilistic bounds).

---

## 3. Current Evaluation Results

### System CI/CD & Testing
A 43-test integration, robustness, and adversarial suite (`run_all.py`) is used to validate system integrity. The most recent diagnostic run yielded **39/43 passing tests (90.7%)**, taking ~490 seconds.
*   **Layer 3, Layer 4, Layer 5, and Adversarial** tests pass 100%.
*   **Failures (4):** The failures highlight the core unresolved architectural blocks (L1 stub entropy discrepancy, L2 Model RMSE mismatch, Integration Clinical Metrics, and Integration Multi-Patient Safety bounds).

### Clinical Baseline Metrics
The most recent 2-day single adult closed-loop test (`seed=42`) using the MPC yields the following:
*   **Time In Range (TIR, 70-180 mg/dL):** 34.7% 
*   **Time Below Range Severe (TBR, <54 mg/dL):** 0.0%
*   **Time Above Range Severe (TAR, >250 mg/dL):** 41.7%
*   **Mean Glucose:** 242.3 mg/dL
*   **Systemic Observation:** The pipeline operates with a strong safety factor against hypoglycemia (0.0% severe) but struggles severely with chronic hyperglycemia (Mean: ~242) and fails to achieve a clinically relevant TIR boundary ($\ge 70\%$).

---

## 4. Key Identified Problems (Blockers)

### 1. Structural Forward Model Mismatch
This is the most critical constraint in the system. Layer 2 Digital Twin predicts glucose using the 3-state Bergman Minimal Model, but it is controlling a 10-state Hovorka patient.
*   **Symptom:** The Bergman ODE's glucose effectiveness parameter `p1` (0.028/min) drives too-fast glucose self-correction. The Bergman model persistently underpredicts true high-range glucose relative to the Hovorka outcome.
*   **Impact:** The MPC optimizes doses against a profoundly inaccurate prediction horizon (initially 219 RMSE).
*   **Neural Residual Limits:** Training the neural residual proved mathematically insufficient to bridge a structural ODE discrepancy of this magnitude (only improved RMSE by 1.8%).

### 2. Generalization Across Patient Cohorts (R-INT-04 Failure)
Currently, in a multi-patient batch encompassing Adult, Adolescent, and Child parameter boundaries, the system is failing safety thresholds for the pediatric group.
*   **Symptom:** Child patients in `test_integration.py` (`pid=1`, `pid=3`) experience ~9.4% to 13.0% Severe TBR (hypoglycemia), severely violating the 0.0% goal.
*   **Impact:** Generalization and body-weight scaling mechanisms in the MPC dosing or L5 reachability parameters have not entirely normalized risk for individuals with low Body Weight (e.g. 30kg children).

---

## 5. Implementation Specifics & Recent Attempts at Solutions

### The MPC Controller
*   Optimizes a custom cost function penalizing excursions from 120 mg/dL, utilizing a grid search of scaling candidates `[0.1, 0.25, ..., 1.5]` to a heuristically generated base dose.
*   **Tuning Sweep:** The cost function weight for insulin (`W_INSULIN`) was swept; `W_INSULIN=0.5` provided the peak TIR limit of 36.3%. No value reaches 50% due to the binding model mismatch.
*   The previous IOB guard (exponential decay model) and PLGS suspend layers have been structurally removed to allow the MPC uninhibited forward prediction capacity.

### Bias Correction Implementation
Due to the mathematical impossibility of the neural residual capturing the Hovorka ODE complexity, a direct **Innovation Bias Correction term** was applied in `DigitalTwin.predict_trajectory`.
*   **Approach:** Aggregates a running mean of the last 20 UKF residual errors (innovations) and injects this additively to the initial state (+full bias) and recursively over the forward horizon (+10% per step).
*   **Result:** Mean signed error drastically dropped from -131.7 mg/dL to -5.3 mg/dL. RMSE against Hovorka drops from 213.2 to 116.7 mg/dL (a massive 45.3% improvement). 

### Layer 5 Modifications
An analysis of the Signal Temporal Logic behavior revealed an intrinsic gap in the $\phi_4$ specification (Persistent Hyperglycemia $>300$ for $>30\text{min}$). 
*   **Issue:** The L5 supervisor was correctly identifying persistent hyperglycemia and vetoing the MPC dose (returning `STL_BLOCKED`), but `BLOCKED` universally returned 0.0 units, starving the patient of a desperately needed correction dose over prolonged windows.
*   **Correction:** Added a weight-scaled minimum correction floor during $\phi_4$ violations ($`\text{Dose} = (\text{Glucose} - 200) / \text{ISF}`$) yielding `STL_CORRECTED`.

---

## 6. Open Research Questions

To advance the AEGIS 3.0 pipeline beyond the 35% TIR threshold without abandoning the formal causal architecture:
1.  **Online Parameter Fitting:** Could augmenting the UKF state vector to learn the individual's `p1` (glucose effectiveness), `p2`, or `p3` parameters online close the Bergman-Hovorka mismatch gap permanently?
2.  **Adaptive Innovation Forgetting:** The 20-step rectangular window for bias correction lags during rapid physiological changes (meals/exercise). Will an exponential forgetting factor for the bias correction improve responsiveness?
3.  **Pediatric Cohort Safety:** How can the MPC dose-scaling cap and the ISF approximations be structurally adjusted so that the 30kg child patient reaches the identical Seldonian safety boundary ($\le 1\%$ Hypoglycemia) as the 75kg adult patient?
4.  **SLM Semantic Replacement:** What exact structural integration is required to fully remove `L1_FIREWALL` and `STUB_ACTIVE` from the semantic context engine, using constrained SLM decoding?

## 7. System Constraints & Invariants
*   **Layer 2 (Digital Twin) Structural Code**: The AC-UKF + RBPF implementation computing the full outer product Q-update and RBPF switching logic is fundamentally sound and must **NOT** be touched.
*   **Causal Integrity:** Changes mapping inputs backwards against the directional flow of the DAG (e.g. attempting to optimize over proxies instead of optimizing through states to outcomes) are mathematically prohibited. 
*   **Hovorka Simulator Structure:** The simulator parameter mappings must remain fixed to enforce rigid unobservable testing fidelity. No cheat sheets transferring parameters from the simulation to the twin are permitted.
