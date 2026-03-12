# AEGIS 3.0 — Paper Claims Registry

**Purpose:** This document constrains what the paper can and cannot claim,
based on actual empirical evidence from the causal evaluation experiment.
Every claim must cite a specific test result or metric. No claim may exceed
what the evidence supports. This document must be consulted before writing
any results section, abstract, or conclusion.

**Last updated:** 2026-03-12
**Evidence base:** 50-patient experiment (seed=42), 288 obs/patient,
semi-synthetic DGP with Hovorka-derived ground truth τ_i.

---

## Supported Claims (evidence exists)

### Claim 1: Proximal G-estimation reduces MAE vs. naive OLS under strong proxies

**Evidence:** MAE_naive = 4.04, MAE_proximal_strong = 2.12.
Reduction = 47.5%. Passes the revised 20% threshold.

**Exact wording permitted:**
> "Under strong proxy conditions (β=0.8, σ=0.2), the proximal
> G-estimator achieved 47.5% lower MAE than naive OLS (2.12 vs 4.04
> mg/dL/U) across 50 simulated patients."

**Not permitted:**
- Claiming this holds for real patient data
- Claiming "clinical significance" — only statistical improvement is shown

### Claim 2: Proxy quality matters — strong proxies outperform weak proxies

**Evidence:** MAE_proximal_strong = 2.12, MAE_proximal_weak = 3.54.
Strong < weak. Criterion 3 passes.

**Exact wording permitted:**
> "Proxy quality significantly affects estimation accuracy: strong proxies
> (β=0.8) yielded MAE = 2.12 vs. weak proxies (β=0.3) MAE = 3.54,
> confirming that the estimator genuinely uses proxy information."

### Claim 3: The proximal estimator beats all alternatives under strong proxies

**Evidence:** Proximal strong (2.12) < Naive (4.04), Population AIPW (6.08),
Standard G-est (4.04).

**Exact wording permitted:**
> "The proximal G-estimator outperformed all three comparison estimators
> under strong proxy conditions."

### Claim 4: Ground truth τ_i is heterogeneous across patients

**Evidence:** τ range = [-25.03, -0.21], span = 24.82 mg/dL/U.

**Exact wording permitted:**
> "The Hovorka simulator produced heterogeneous individual treatment
> effects spanning 24.82 mg/dL/U across 50 patients."

### Claim 5: The control function approach identifies τ without known propensity

**Evidence:** The DGP generates continuous treatment A via a confounded
equation with no known assignment mechanism. The proximal estimator
achieves 47.5% MAE reduction using only proxy variables Z and W, without
using the propensity field. Standard G-estimation (which has no proxy
adjustment and no valid propensity) performs identically to naive OLS.

**Exact wording permitted:**
> "Identification is achieved via the proximal bridge function
> (Cui et al. 2024), not via known treatment propensity. The proxy
> independence conditions provide the identification mechanism."

### Claim 6: Results replicate at scale (50 patients)

**Evidence:** All five claims originally established on 10 patients
hold at 50 patients with consistent effect sizes. MAE reduction
under strong proxies: 57% (n=10) → 47.5% (n=50). The smaller
reduction at scale is expected: the 50-patient cohort includes
more diverse patients (τ span 24.82 vs 16.89 mg/dL/U), making
estimation harder. The proximal estimator still dominates all
alternatives by a wide margin.

**Exact wording permitted:**
> "Results replicate at 50 patients. The proximal G-estimator
> achieved 47.5% MAE reduction vs. naive OLS, consistent with
> the 57% reduction observed at 10 patients. The wider patient
> heterogeneity at n=50 (τ span = 24.82 mg/dL/U) makes the
> estimation problem harder, yet the proximal estimator remains
> the best-performing method."

---

## Unsupported Claims (evidence does NOT exist — do NOT make these)

### ✗ "Confidence sequences achieve 95% coverage"

**Reality:** The reported 96% coverage (strong and weak) is achieved
using sensitivity-adjusted intervals with:
1. Dividing effective sample size by 5 (`effective_n = n/5`), which
   inflates CI width by √5 ≈ 2.24×
2. Adding sensitivity padding (`gamma=0.15 × |τ̂|`)

These are hand-tuned constants with post-hoc justification. The honest
coverage numbers (without inflation) should be reported alongside the
inflated ones. Any paper that reports only the 96% figure without
disclosing the inflation mechanism is misleading.

**What the paper CAN say:**
> "Sensitivity-adjusted intervals (effective_n = n/5, γ=0.15) achieve
> 96% coverage under both strong and weak proxy conditions across 50
> patients. However, these intervals use a hand-tuned sensitivity
> parameter. Honest CLT-based confidence intervals (without inflation)
> have lower coverage, consistent with residual confounding bias in
> the point estimates."

### ✗ "The method works for real patient text data"

**Reality:** All proxies are synthetic, generated from known latent
variables with exact satisfaction of independence conditions. No real
text has been processed by the proximal estimator.

### ✗ "Standard G-estimation provides a meaningful comparison"

**Reality:** After removing propensity weighting (Fix 1 from forensic
audit), Standard G-estimation produces identical results to Naive OLS
(MAE = 4.04 for both). In this DGP, without valid propensity scores
or proxy variables, per-patient OLS IS naive OLS. The estimator should
be reported as "Per-patient OLS (no confounding adjustment)" or removed
from the comparison table to avoid implying it is a distinct method.

### ✗ "The 30% MAE reduction threshold is clinically grounded"

**Reality:** The 30% threshold was set in system_design.md without
clinical justification. It has been revised to 20% based on the
finite-sample literature (Kompa et al. 2022). The 47.5% actual reduction
exceeds both thresholds, but the threshold itself is arbitrary.

### ✗ "The framework is ready for clinical deployment"

**Reality:** See system_design.md §8 for the full list of limitations.
The experiment uses a synthetic simulator, synthetic proxies, and a
50-patient cohort. Clinical validation requires real patient data,
IRB approval, and safety monitoring.

---

## Coverage Reporting Requirements

Any results table in the paper MUST report coverage with full transparency:

| Method | Strong (n=50) | Weak (n=50) | Description |
|--------|---------------|-------------|-------------|
| Sensitivity-adjusted (n/5 + γ=0.15) | 96% | 96% | Full inflation |

The paper must state: "The 96% coverage is achieved using sensitivity-
adjusted intervals with effective sample size n/5 and sensitivity
parameter γ=0.15. These constants account for bridge function estimation
error and residual unmeasured confounding, but are hand-tuned rather
than derived from theory. The equal coverage under strong and weak
proxies (both 96%) reflects the sensitivity padding dominating the
interval width, rather than the proxy quality driving the uncertainty
quantification."

---

## Per-Patient Diagnostics Summary (50 patients, strong proxies)

The per-patient diagnostics reveal the mechanism behind the proximal
estimator's performance:

- **First-stage F-statistics** range from 0.45 to 31.9 (median ~6.4)
  - Only ~30% of patients exceed the Stock & Yogo (2005) F=10 threshold
  - The relevance weighting mechanism is actively engaged for most patients
- **Relevance weights** range from 0.04 to 0.96 (median ~0.47)
  - Most patients receive a blend of control function and naive estimates
  - The estimator is NOT applying full proximal correction for most patients
- **tau_cf** (raw control function) is consistently less negative than
  **tau_naive**, confirming the control function corrects toward zero
  (reducing the positive confounding bias from U→A→Y)

This diagnostic transparency is important: the proximal estimator's
advantage comes partly from the control function correction and partly
from the relevance weighting acting as regularization. Both mechanisms
contribute to the 47.5% MAE reduction.

---

## Full Results Table (50 patients, seed=42, 288 obs/patient)

```
Estimator                 Proxy       MAE    P95 Err   Coverage    Bias
Naive OLS                 strong    4.0379    6.6388      N/A    -4.0379
Population AIPW           strong    6.0841    8.8077      N/A    -3.4354
Standard G-estimation     strong    4.0379    6.6388      N/A    -4.0379
Proximal G-estimation     strong    2.1195    3.6574    0.960    -1.7492
Naive OLS                 weak      4.0379    6.6388      N/A    -4.0379
Population AIPW           weak      6.0841    8.8077      N/A    -3.4354
Standard G-estimation     weak      4.0379    6.6388      N/A    -4.0379
Proximal G-estimation     weak      3.5420    6.1871    0.960    -3.5108
```

---

## Revision History

- 2026-03-11: Initial creation based on 10-patient experiment results
  and forensic audit findings.
- 2026-03-12: Updated with 50-patient experiment results. All five
  original claims confirmed at scale. Added Claim 6 (replication at
  scale). Updated all numeric values. Added per-patient diagnostics
  summary and full results table. Coverage section updated with 50-patient
  numbers (96% strong, 96% weak under sensitivity adjustment).
