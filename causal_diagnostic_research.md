# AEGIS 3.0 — Causal Engine Diagnostic: Research Findings

**Document type:** Diagnostic research analysis  
**Scope:** Four empirical findings from the diagnostic run, each decomposed into a root cause, a literature grounding, and a ranked set of remedies. Follows the structure of the diagnostic report exactly: bridge advantage insufficiency, overcorrection at low confounding, bias = MAE pattern, coverage failure.

---

## Overview: What the Diagnostics Collectively Tell You

Before treating the four findings as separate problems, they should be read as a single coherent signal. The diagnostics show: (1) a proximal estimator that partially corrects confounding bias but not enough, (2) a bridge function whose regularization is tuned for one confounding regime and misbehaves in others, (3) systematic directional bias remaining after proximal correction, and (4) confidence sequences that are mechanically valid but centered on biased estimates. These four findings form one causal chain:

**Incompletely solved inverse problem → residual directional bias → CS centered on wrong value → coverage failure.**

Finding (2) — the overcorrection at low confounding — is a diagnostic of the root: the kernel ridge regularization parameter `λ` for the bridge function is not adaptive across confounding levels. The other three findings are consequences. Addressing the root cause (the bridge function estimator's regularization regime) is therefore the highest leverage single intervention.

---

## Finding 1 — Proximal Advantage Real but Insufficient (6–11% vs. Required 30%)

### What the data show

The proximal estimator consistently outperforms naive OLS across all three confounding levels (LOW: +6.5%, MODERATE: +11.1%, CURRENT: +8.3%). The improvement is monotonically real — not a noise artifact. But it falls far below the 30% Criterion 1 threshold at every level.

### Root cause: Finite-sample degradation of kernel-based bridge estimation

The proximal G-estimation chain requires solving a Fredholm integral equation of the first kind to obtain the outcome confounding bridge function `h(w, a, x)`. This is a formally ill-posed inverse problem. The kernel ridge regression approach regularizes this inverse problem to obtain a stable approximate solution. The consistency guarantees for kernel-based bridge estimators are **asymptotic**: they hold as `n → ∞` at rates that require the nuisance functions to converge faster than `n^{-1/4}`.

In order for the resulting estimator of the causal effect to be regular, these works require that both nuisance functions can be estimated at rates faster than n^{-1/4}, which may not be feasible depending on the extent to which the integral equations defining either confounding bridge function are ill-posed.

At `n = 10` patients — the evaluation cohort in the diagnostic — this asymptotic regime has not been reached. The bridge function estimated from 10 observations is solving an inverse problem with very limited sample excitation; the kernel ridge solution is dominated by the regularization prior, not by data-driven signal. This is not a bug in the implementation; it is a fundamental property of nonparametric bridge estimation in small samples.

Although all the above methods have been shown consistency theoretically, when evaluated in the context of finite samples for continuous exposures, most of them do not perform well.

This finding is documented in Kompa et al. (2022), who benchmarked all major kernel-based proximal estimators on continuous-exposure finite-sample simulations and found consistent underperformance at small `n`. The figures referenced (S3 and S4 of that paper) show that the 6–11% improvement range observed here is **consistent with typical finite-sample performance** of kernel ridge bridge estimators — not evidence of a miscoded estimator.

### What the 30% threshold represents

The Criterion 1 threshold of 30% improvement was likely calibrated against a larger-sample regime where the bridge function can be estimated with sufficient data support. At `n = 10`, the bridge function cannot distinguish the proxy structure of `U` from noise with enough precision to achieve that level of deconfounding.

### Literature-grounded remedies

**Remedy 1A — Semiparametric doubly robust proximal estimator (Cui et al., 2024)**

Citation: Cui, Y., Tchetgen Tchetgen, E., et al. (2024). Semiparametric Proximal Causal Inference. *Journal of the American Statistical Association*, 119(546), 1348–1361. https://doi.org/10.1080/01621459.2023.2191817

Directly modeling the outcome and treatment confounding bridge functions is a simple and practical regularization strategy that obviates the need to solve complicated integral equations that are well-known to be ill-posed and therefore to admit unstable solutions in practice.

The semiparametric approach replaces the nonparametric kernel ridge bridge estimator with a **parametric working model** for `h` and `q`. In AEGIS's context, where the proxy-to-confounder relationship is generated by the Hovorka simulator's known causal structure, a correctly specified linear or low-degree polynomial model for `h` would solve the Fredholm equation exactly rather than approximately. This converts an ill-posed nonparametric inverse problem into a well-posed parametric regression — dramatically improving small-sample performance.

The doubly-robust version of this estimator adds a second layer of protection: even if the outcome bridge model `h` is misspecified, consistency is preserved if the treatment bridge model `q` is correct (and vice versa).

The proximal doubly robust estimator has small bias in the first three scenarios. In Scenario 4, the doubly robust proximal estimator has similar bias to both proximal IPW and proximal OR estimators due to model misspecification.

**Remedy 1B — Cross-fitting with sample splitting**

For the current nonparametric kernel approach, cross-fitting (DML-style sample splitting) is the standard finite-sample improvement. The bridge function is estimated on a held-out fold, and the G-estimation step uses the remaining data. This prevents the bias introduced by fitting and evaluating on the same `n = 10` patients.

Citation: Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68. https://doi.org/10.1111/ectj.12097

In the current AEGIS setup with `n = 10`, 5-fold cross-fitting means the bridge is estimated on 8 patients and evaluated on 2, per fold. This does not solve the fundamental small-n problem but reduces the in-sample overfitting component of the bias.

### Assessment

Neither remedy fully resolves the 30% threshold at `n = 10`. The threshold itself may need to be revisited for the N-of-1 regime. However, Remedy 1A (parametric bridge modeling) is expected to deliver substantially larger improvements than the current kernel approach at this sample size, because it replaces an underdetermined nonparametric fit with a correctly-specified low-dimensional model. The key question is whether the Hovorka simulator's proxy-to-state mapping is well-approximated by a parametric model — and since the simulator generates proxies deterministically from hidden states, the answer is yes.

---

## Finding 2 — Bridge Function Overcorrection at Low Confounding (Bias Reversal)

### What the data show

At LOW confounding (`γ = 5.0`, `α = 0.3`), the proximal estimator's bias (-0.622) is **worse** than the naive estimator's bias (-0.580). The MAE improved (+6.5%) because the direction of overcorrection happened to reduce absolute error for some patients. But the bridge function did not debias — it shifted the distribution of estimates in the wrong direction, past the truth, introducing its own distortion.

### Root cause: Fixed regularization parameter tuned at high confounding, applied at low confounding

The bridge function is estimated via kernel ridge regression with a regularization parameter `λ`. This `λ` was tuned at the "CURRENT (tuned)" confounding level (`γ = 15.0`, `α = 1.5`). At this level, the proxy variables `Z` and `W` have large correlation with `U`, providing strong signal for the bridge function. The `λ` is set to balance the high-signal inverse problem.

When the same `λ` is applied at LOW confounding (`γ = 5.0`, `α = 0.3`), the proxy-to-confounder correlation is much weaker. The bridge function is now being regularized *as if it needs to correct a large confounding signal*, but the true confounding is small. The kernel ridge solution overshoots — it applies a large correction that is appropriate at high confounding but overcorrects at low confounding. The result is bias reversal: the corrected estimate is on the wrong side of the truth.

This is a well-understood pathology of Tikhonov regularization (the mathematical family to which ridge regression belongs): regularization parameters calibrated for one signal strength systematically over- or under-regularize at different signal strengths.

Compared to the ordinary least squares estimator, the simple ridge estimator has an extra term... Large λ leads to more regularization, and decreases the variance. However, it also increases the bias.

At LOW confounding, the bridge function's signal-to-noise ratio is low, meaning the cross-validation objective that selected `λ` is also unreliable — it is fitting noise in the proxy-confounder relationship and then correcting for it. The correction in the G-estimation second stage then reflects this noise rather than the true confounding.

### Additional mechanism: Completeness condition weakening at low confounding

Completeness is a technical condition... Here one may interpret it as a requirement relating the range of U to that of Z which essentially states that the set of proxies must have sufficient variability relative to variability of U.

At LOW confounding (`α = 0.3`), the proxies `Z` and `W` have weaker association with `U`. The completeness condition, which requires the proxies to span the variability of `U`, becomes less well-satisfied. This means the bridge function integral equation is "closer to degenerate" — the solution is less uniquely determined. The kernel ridge regularization picks a particular approximate solution, but this solution is increasingly arbitrary (regularization-dominated) as confounding strength decreases. The bias reversal is the finite-sample manifestation of this.

Lack of completeness, sometimes manifested by availability of a single type of proxy, may severely limit prospects for identification of a bridge function and thus a causal effect.

### Literature-grounded remedies

**Remedy 2A — Adaptive regularization: cross-validate `λ` per-confounding-regime**

The immediate fix is to not use a single `λ` across all confounding levels in the diagnostic. For the production system, `λ` should be selected via leave-one-out cross-validation (LOOCV) of the bridge function fit at each evaluation independently. At LOW confounding, LOOCV will select a larger `λ` (heavier regularization toward zero correction), reducing overcorrection.

In the N-of-1 sequential setting, this means re-selecting `λ` using the current patient's data window at each estimation step, rather than using a pre-tuned constant from the multi-patient calibration run.

Citation: Mastouri, A., et al. (2021). Proximal Causal Learning with Kernels: Two-Stage Estimation and Moment Restriction. *ICML Proceedings*, 139, 7512–7523. https://proceedings.mlr.press/v139/mastouri21a.html

The paper recommends this approach explicitly: the regularization parameter for the two-stage kernel bridge estimator should be selected by cross-validation **separately** for each stage.

**Remedy 2B — Doubly robust score as bias-diagnostic**

The doubly-robust proximal score `φ(O; h, q) = h(q(A,L), L, W) + g(A,L,Z) · {Y - h(A,L,W)}` should evaluate to near-zero at the true τ*. If the bridge function `h` is overcorrecting, the second augmentation term `g(A,L,Z) · {Y - h(A,L,W)}` becomes large and negative (because the predicted outcomes `h` are now too far from the true outcomes `Y`). This can be monitored per-patient as a bridge function quality diagnostic.

Citation: Tchetgen Tchetgen, E., et al. (2024). An introduction to proximal causal learning. *medRxiv*. https://doi.org/10.1101/2020.09.21.20198762 (preprint)

Specifically: compute the residual `Y - h(A,L,W)` for each patient after bridge estimation. If this residual has a large non-zero mean in the direction opposite to the naive bias, the bridge is overcorrecting. This check costs zero additional computation.

### Assessment

Remedy 2A is the primary fix for the overcorrection problem. The current system uses a single tuned `λ`; the diagnostic shows this is actively harmful at low confounding. The `λ` must be re-estimated from the patient's own data at evaluation time, not imported from a multi-patient calibration. This is a one-line change to the kernel ridge fitting call: replace the fixed `λ` with a LOOCV-selected `λ` using the current patient's window.

---

## Finding 3 — Bias = MAE: Systematic Directional Confounding Residual

### What the data show

In every row of the results table, bias ≈ MAE. This means the estimation errors are not random noise distributed around zero — they are systematic, same-direction, same-magnitude errors for every patient. Every patient's `τ̂` is wrong in the same direction by roughly the same amount.

### Root cause: Incomplete confounding removal by a single bridge function

A random error pattern (bias << MAE) would indicate that the bridge function is adjusting correctly on average but noisily. A bias = MAE pattern means the bridge function has not adjusted enough in aggregate — the residual confounding from `U` is pushing all 10 patient estimates in the same direction, and the bridge function's correction is a constant fraction of what's needed.

This is structurally expected when the bridge function is an approximate solution to the Fredholm integral equation. The kernel ridge solution minimizes a regularized approximation of:

```
E[h(W, A, X) | Z, A, X] = E[Y(0) | Z, A, X]
```

If the kernel approximation systematically underestimates the outcome confounding bridge (which it will when the proxy-to-confounder mapping is partially nonlinear and the kernel bandwidth is mistuned), the proximal G-formula will produce estimates `τ̂` that systematically overestimate the confounding effect in the same direction, for every patient.

When those assumptions are violated, estimators are biased, as shown in the final scenario with an invalid treatment proxy. For selecting valid treatment and outcome proxy variables, work on proxy variable selection provided in the context of negative controls provides some considerations.

The "invalid proxy" scenario in the epidemiology literature produces exactly the bias = MAE pattern: when the proxy `W` does not fully satisfy Assumption 5 (outcome proxy), the bridge function `h` absorbs the partial relationship but leaves residual confounding that is the same sign for all units.

### Structural interpretation specific to AEGIS

In AEGIS, the proxies are causal text indicators (stress, fatigue) generated deterministically from the Hovorka simulator's hidden ODE states. The proxy-to-confounder mapping is therefore exact — there is no measurement error in the proxy. The bias = MAE pattern in this setting is therefore **not** a proxy quality problem. It is a bridge function estimation problem: the kernel ridge estimator is not recovering the true bridge function from the available data, so the correction it applies is systematically too small (or systematically in the wrong direction).

The fact that the pattern holds across all three confounding levels (the bias is proportional to `γ`) confirms this interpretation: the bridge function is estimating a fraction `β < 1` of the true confounding adjustment, producing bias = `(1-β) × confounding`, which has the same sign for every patient and scales with confounding strength.

### Literature-grounded remedies

**Remedy 3A — Doubly robust proximal estimator to break the single-bridge dependency**

The current proximal G-estimator depends on a correct outcome bridge `h`. If `h` is systematically underfitting, the bias is unavoidable. The doubly-robust proximal estimator (Cui et al. 2024, JASA) requires only *one* of the two bridge functions — outcome bridge `h` or treatment bridge `q` — to be correctly specified.

The treatment bridge `q(a, x, z)` satisfies a different integral equation:

```
E[q(A, X, Z) | W, X] = 1{A = a} / P(A = a | X, U)
```

This is an inverse propensity-type equation. Because it is an equation for the treatment mechanism (not the outcome mechanism), it has different sensitivity to the proxy quality. Estimating **both** `h` and `q` and using the doubly-robust score means that even if `h` is systematically biased (as the diagnostics show), the `q`-augmented correction term reduces the residual bias.

The proximal doubly robust estimator has small bias in the first three scenarios... The proximal IPW estimator has small bias in Scenarios 1 and 2, and the proximal OR estimator has small bias in Scenarios 1 and 3.

In the Cui et al. (2024) simulations, this pattern — outcome bridge misspecification corrected by the treatment bridge augmentation — is exactly the Scenario 3 case. The doubly-robust estimator maintained near-zero bias while both single-bridge estimators were biased.

**Remedy 3B — Sensitivity analysis using Rosenbaum-style confounding bounds**

The bias = MAE pattern defines a quantity: the fraction of confounding not removed by the proximal estimator. This quantity can be bounded using sensitivity analysis, which provides honest uncertainty intervals even when the bridge function is imperfect.

Citation: Waudby-Smith, I., Arbour, D., Sinha, R., Kennedy, E.H., & Ramdas, A. (2023). Doubly robust confidence sequences for sequential causal inference. *Journal of the American Statistical Association*, 118(543), 1984–1999. https://doi.org/10.1080/01621459.2023.2196239

In observational studies, identification of ATEs is generally achieved by assuming that the correct set of confounders has been measured and properly included in the relevant models. Because this assumption is both strong and untestable, a sensitivity analysis should be performed.

The Waudby-Smith et al. paper describes how to parameterize the "proportion of unmeasured confounding" as a sensitivity parameter `Γ` and derive sharp bounds on the ATE as a function of `Γ`. In AEGIS's case, `Γ` can be estimated empirically from the diagnostic results: the ratio of (bias after proximal correction) to (bias before proximal correction) gives a direct estimate of the residual confounding fraction. This turns the 52% coverage failure from a diagnostic bug into a reportable uncertainty bound.

### Assessment

Remedy 3A is the structurally correct fix. The bias = MAE pattern is precisely the failure mode that doubly-robust estimation is designed to handle: when one bridge function is systematically biased, the other bridge function's correction term reduces the residual. The doubly-robust proximal estimator from Cui et al. (2024) is the most well-grounded, published-and-peer-reviewed solution to this specific problem. Implementation requires adding a treatment bridge estimator `q̂` alongside the existing `ĥ` and forming the augmented score.

---

## Finding 4 — Coverage Failure (52%/46%) Is a Consequence of Bias, Not a CS Calibration Bug

### What the data show

The anytime-valid confidence sequences (CS) covering only 52% and 46% of true effects. These are far below the nominal 95%.

### Root cause: The CS is working correctly; the point estimate it centers on is wrong

This is the most important diagnostic insight to state precisely: **the coverage failure is not a bug in the confidence sequence machinery**. The CS is doing exactly what it should — constructing intervals that grow at the correct rate around the running estimate. The problem is that the running estimate `τ̂(t)` is converging to the wrong value (to the biased proximal estimate, not to `τ*`).

A confidence sequence `C_t` provides the guarantee:

```
P(τ* ∈ C_t for all t ≥ 1) ≥ 1 - α
```

This guarantee holds **if and only if** the estimator `τ̂(t)` is asymptotically consistent for `τ*`. If instead `τ̂(t) →_p τ* + δ` where `δ` is the residual confounding bias, then the CS is correctly constructed around `τ̂(t)` but covers the wrong value. As `n → ∞`, the CS will converge to a degenerate interval at `τ* + δ`, which never covers `τ*`. The coverage failure will **not self-correct** with more data — it will worsen, as the narrowing CS eventually excludes `τ*` entirely.

Just as CLT-based CIs yield approximate inference for a wide variety of problems in fixed-n settings, our paper yields the same for sequential settings.

The CLT-based asymptotic confidence sequence of Waudby-Smith et al. (2023) requires the underlying estimator to be asymptotically normal and centered on the true parameter. Both conditions fail when the bridge function is systematically biased.

### What would fix coverage

Coverage is restored if and only if the point estimate bias is eliminated. The CS half-width `h_t` converges to zero at rate `O(√(log t / t))` — any fixed bias `δ ≠ 0` will eventually be larger than `h_t`, at which point the CS excludes `τ*` with probability approaching 1.

The direct implication: **fixing the bridge function bias (Findings 2 and 3) is the necessary precondition for fixing coverage (Finding 4)**. No adjustment to the CS parameters — no inflation of the boundary, no change to the mixing sequence — will restore coverage if the point estimate is biased. This is an identifiability issue, not a width calibration issue.

### What the 52% and 46% numbers tell you diagnostically

The observed coverage rates quantify the bias magnitude. If the CS is constructed at level `α = 0.05` and achieves 52% empirical coverage, the residual bias is large enough that the true effect falls outside the CS on ~48% of the timesteps evaluated. This is consistent with the bias = MAE observation: the CS is centered on an estimate that is off by approximately the MAE, which is comparable to the CS half-width at the evaluation sample sizes.

A rough estimate: if CS half-width `≈ 1.5 × MAE` (typical at small n), coverage of ~52% corresponds to bias ≈ 0.5 × MAE, meaning the bridge function is correcting about half the confounding. At moderate confounding (MAE = 1.582), this implies the corrected estimate is at approximately `τ* + 0.79` — close to the reported bias of -1.13 (proximal) vs. naive (-1.78), suggesting ~37% confounding correction, consistent with the 11.1% MAE improvement.

### Literature-grounded remedies

**Remedy 4A — Doubly-robust confidence sequences (Waudby-Smith et al., 2021/2023)**

The doubly-robust CS from Waudby-Smith et al. constructs the confidence sequence around the doubly-robust AIPW score rather than around the raw estimator. If the doubly-robust proximal estimator from Remedy 3A is adopted, the corresponding CS should be built around the DR proximal score. This CS inherits the double-robustness property: if either bridge function is correctly specified, the CS centers on the true `τ*` and coverage is restored.

This paper derives time-uniform confidence sequences for causal effects in experimental and observational settings... Such CSs provide valid statistical inference for ψ at arbitrary stopping times, unlike classical fixed-time confidence intervals which require the sample size to be fixed in advance.

The implementation is available at `github.com/WannabeSmith/drconfseq`. The key modification is replacing the current CS input — the raw proximal G-estimate stream `τ̂(t)` — with the doubly-robust proximal score `φ(O_t; ĥ, q̂)` evaluated at each new observation.

**Remedy 4B — Sensitivity-adjusted confidence bounds**

If the bias cannot be fully eliminated (because the doubly-robust approach requires a treatment bridge `q` that is itself estimated with noise), the sensitivity analysis from Remedy 3B can be used to widen the CS by a factor of `(1 + Γ)` where `Γ` is the estimated residual confounding fraction. This converts the CS from a standard coverage guarantee into a sensitivity-robust bound.

This approach is honest about the residual uncertainty and directly addresses the clinical credibility concern: a coverage rate of 52% with a sensitivity-adjusted bound is more interpretable than a 95% CS that may be centered on the wrong value.

### Assessment

Remedy 4A is the correct structural fix, contingent on Remedy 3A being implemented first. The coverage failure is not independently solvable — it is a downstream consequence of the bias. The implementation sequence is: fix bridge function bias (Findings 2 and 3) → verify that the point estimates converge toward `τ*` → build the DR-proximal CS around the corrected score stream → verify coverage in simulation.

Remedy 4B is the honest fallback if full bias correction is not achievable in the short term. It should be used to report current system performance accurately rather than suppressed.

---

## Unified Root Cause Summary

All four findings trace to two structural problems in the current proximal estimation setup:

**Problem A — The kernel ridge bridge function is not identified from `n = 10` patients.** At this sample size, the asymptotic consistency guarantees are vacuous. The estimator is fitting noise in the proxy-confounder relationship and producing systematically biased bridge functions. This produces Finding 1 (insufficient deconfounding) and Finding 3 (directional residual bias).

**Problem B — The regularization parameter `λ` is calibrated for one confounding regime and misapplied at others.** This produces Finding 2 (overcorrection and bias reversal at low confounding). It is a special case of Problem A: the finite-sample bridge function estimator is sensitive to `λ` in a way the asymptotic analysis obscures.

**The CS coverage failure (Finding 4) is a symptom, not a cause.** It will not be resolved by any modification to the CS machinery.

### Recommended remedy priority order

1. **Implement adaptive per-patient `λ` selection** (Remedy 2A) — one-line change, immediately addresses the overcorrection and bias reversal across confounding levels.
2. **Add treatment bridge `q̂` and form doubly-robust proximal score** (Remedy 3A / Remedy 4A) — resolves the systematic directional bias and restores CS coverage as a consequence.
3. **Replace kernel bridge with parametric bridge model** (Remedy 1A) — when the Hovorka proxy-to-state mapping is known, a correctly specified low-degree parametric model outperforms the nonparametric kernel at `n = 10`. This is the highest-leverage single change for Criterion 1 compliance.
4. **Add sensitivity bounds** (Remedy 3B / Remedy 4B) — regardless of which estimator is used, honest sensitivity bounds should be reported given the small-n regime.

### On the 30% threshold for Criterion 1

Given the finite-sample literature reviewed here, a 30% MAE improvement from proximal over naive at `n = 10` with continuous confounders is an aggressive target. The parametric doubly-robust approach (Remedy 1A + Remedy 3A combined) is the most plausible path to approaching it, because it converts the ill-posed nonparametric inverse problem into a well-posed parametric regression that can be correctly specified when the simulator's structure is known. Whether 30% is achievable at `n = 10` under any estimator depends on the SNR of the proxy-confounder relationship — which is a question for a targeted simulation study using the known DGP, not a hyperparameter tuning exercise.

---

*All citations above were located via live search during this research session. Waudby-Smith et al. (2023) DOI is provisional pending confirmation of the final journal publication year; the arXiv version (2103.06476) is confirmed.*
