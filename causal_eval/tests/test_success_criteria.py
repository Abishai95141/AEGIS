"""
Formal validation of the success criteria from system_design.md §7.

Runs a 10-patient experiment and asserts:
  Criterion 1 (revised): >20% MAE reduction (Proximal strong vs Naive strong)
      Revised from 30% per forensic audit and Kompa et al. (2022) finite-sample
      literature. See PAPER_CLAIMS.md for justification.
  Criterion 2 (replaced): Honest coverage reporting — THREE numbers reported,
      NO pass/fail assertion. The original ≥90% coverage criterion was
      achievable only via engineered CI inflation (n/5 divisor + γ=0.15
      sensitivity padding). See PAPER_CLAIMS.md §Coverage Reporting Requirements.
  Criterion 3: Strong proxy MAE < Weak proxy MAE for Proximal G-estimation

These are the acceptance criteria for the causal evaluation framework.
"""

import sys
import os
import unittest
import math
import warnings
import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from causal_eval.evaluation.experiment import CausalExperiment
from causal_eval.estimators.proximal_gestimation import ProximalConfidenceSequence


class TestSuccessCriteria(unittest.TestCase):
    """
    End-to-end validation of the success criteria.

    Uses a 10-patient experiment for speed; the full 50-patient experiment
    (run via experiment.py __main__) produces the official results.
    """

    @classmethod
    def setUpClass(cls):
        """Run the experiment once and cache the results."""
        cls.exp = CausalExperiment(n_patients=10, n_obs=288, base_seed=42)
        cls.results, cls.table_str = cls.exp.run()

        # Index results by (estimator, proxy_condition) for easy lookup
        cls.result_map = {}
        for r in cls.results:
            key = (r['estimator'], r['proxy_condition'])
            cls.result_map[key] = r

    def _get(self, estimator, proxy):
        """Helper to look up a result."""
        return self.result_map[(estimator, proxy)]

    # ------------------------------------------------------------------
    # Criterion 1 (REVISED): >20% MAE reduction
    # ------------------------------------------------------------------
    def test_criterion_1_bias_reduction(self):
        """
        Criterion 1 (revised): Proximal G-estimation with strong proxies
        must achieve >20% MAE reduction compared to Naive OLS.

        Revised from 30% to 20% per:
        - Kompa et al. (2022): kernel-based proximal estimators show 6-11%
          improvement at small n; parametric control-function approach does better
        - causal_diagnostic_research.md Finding 1: 30% was calibrated for
          large-sample regime, not n=10 per-patient
        - PAPER_CLAIMS.md: threshold is arbitrary, not clinically grounded

        The actual reduction (57%) far exceeds both thresholds.
        """
        naive_mae = self._get('Naive OLS', 'strong')['mae']
        proximal_mae = self._get('Proximal G-estimation', 'strong')['mae']

        reduction = 1.0 - (proximal_mae / naive_mae)
        self.assertGreater(
            reduction, 0.20,
            f"Criterion 1 FAILED: MAE reduction = {reduction:.1%} "
            f"(Proximal {proximal_mae:.3f} vs Naive {naive_mae:.3f}), "
            f"need >20%"
        )
        print(f"\n✓ Criterion 1 (revised): {reduction:.1%} MAE reduction "
              f"(Proximal {proximal_mae:.3f} vs Naive {naive_mae:.3f})")

    # ------------------------------------------------------------------
    # Criterion 2 (REPLACED): Honest coverage reporting, no assertion
    # ------------------------------------------------------------------
    def test_criterion_2_honest_coverage_report(self):
        """
        Criterion 2 (replaced): Report three coverage numbers honestly.
        NO pass/fail assertion — this is a reporting requirement, not a test.

        The original criterion (coverage ≥ 90%) was achievable only by:
          1. Dividing effective_n by 5 (inflating CI width by √5 ≈ 2.24×)
          2. Adding sensitivity_gamma = 0.15 (padding |τ̂| by 15%)
        Both are hand-tuned constants. The honest coverage (standard CLT)
        is ~50% strong / ~30% weak, reflecting residual confounding bias
        (Finding 4, causal_diagnostic_research.md).

        Three numbers reported per PAPER_CLAIMS.md:
          (a) Honest CLT: 1.96 × SE/√n, no inflation
          (b) Bridge-uncertainty adjusted: effective_n = n/5, no γ
          (c) Sensitivity-adjusted: effective_n = n/5, γ = 0.15
        """
        print("\n--- Criterion 2: Honest Coverage Report ---")
        print("(No pass/fail — reporting requirement only)")
        print("")

        for proxy_cond in ['strong', 'weak']:
            # Get the experiment's proximal estimator for this condition
            # We need to recompute coverage under three different CI methods
            # using the cached experiment data
            Y, A, S, propensity, Z, W, patient_ids = \
                self.exp._build_estimator_inputs(proxy_cond)

            from causal_eval.estimators.proximal_gestimation import (
                ProximalGEstimatorWrapper,
            )
            est = ProximalGEstimatorWrapper()
            tau_est = est.estimate_individual(
                Y, A, S, patient_ids, propensity=propensity, Z=Z, W=W
            )

            ground_truth = self.exp.ground_truth
            coverages = {'honest': 0, 'bridge_adj': 0, 'sensitivity': 0}
            n_patients = len(ground_truth)

            for pid in ground_truth:
                tau_true = ground_truth[pid]
                tau_hat = tau_est.get(pid, float('nan'))
                if math.isnan(tau_hat):
                    continue

                dr_scores = est._individual_dr_scores.get(pid)
                se = est._individual_se.get(pid)

                if dr_scores is None or len(dr_scores) < 2:
                    continue

                n = len(dr_scores)
                score_se = float(np.std(dr_scores, ddof=1))

                # (a) Honest CLT: standard 1.96 × SE/√n, no tricks
                honest_hw = 1.96 * score_se / np.sqrt(n)
                if abs(tau_hat - tau_true) <= honest_hw:
                    coverages['honest'] += 1

                # (b) Bridge-uncertainty adjusted: effective_n = n/5, no γ
                bridge_hw = 1.96 * score_se / np.sqrt(n / 5)
                if abs(tau_hat - tau_true) <= bridge_hw:
                    coverages['bridge_adj'] += 1

                # (c) Sensitivity-adjusted: effective_n = n/5, γ = 0.15
                sens_hw = 1.96 * score_se / np.sqrt(n / 5) + 0.15 * abs(tau_hat)
                if abs(tau_hat - tau_true) <= sens_hw:
                    coverages['sensitivity'] += 1

            cov_honest = coverages['honest'] / n_patients
            cov_bridge = coverages['bridge_adj'] / n_patients
            cov_sens = coverages['sensitivity'] / n_patients

            print(f"  {proxy_cond.upper()} proxies:")
            print(f"    (a) Honest CLT (no inflation):      {cov_honest:.0%} "
                  f"({coverages['honest']}/{n_patients})")
            print(f"    (b) Bridge-adjusted (n/5, no γ):    {cov_bridge:.0%} "
                  f"({coverages['bridge_adj']}/{n_patients})")
            print(f"    (c) Sensitivity-adjusted (n/5+γ):   {cov_sens:.0%} "
                  f"({coverages['sensitivity']}/{n_patients})")
            print("")

        print("  NOTE: Honest coverage reflects residual confounding bias.")
        print("  The inflated intervals use hand-tuned constants (n/5, γ=0.15).")
        print("  See PAPER_CLAIMS.md §Coverage Reporting Requirements.")

    # ------------------------------------------------------------------
    # Criterion 3: Proxy quality sensitivity (unchanged)
    # ------------------------------------------------------------------
    def test_criterion_3_strong_beats_weak(self):
        """
        Criterion 3: Strong proxies must produce lower MAE than weak proxies
        for Proximal G-estimation.

        This validates that proxy quality matters: better proxies → better
        confounding adjustment → lower estimation error. This is a negative
        result that strengthens the paper — it demonstrates the estimator
        genuinely uses proxy information.
        """
        strong_mae = self._get('Proximal G-estimation', 'strong')['mae']
        weak_mae = self._get('Proximal G-estimation', 'weak')['mae']

        self.assertLess(
            strong_mae, weak_mae,
            f"Criterion 3 FAILED: strong MAE ({strong_mae:.3f}) ≥ "
            f"weak MAE ({weak_mae:.3f})"
        )
        print(f"\n✓ Criterion 3: strong MAE {strong_mae:.3f} < "
              f"weak MAE {weak_mae:.3f}")

    # ------------------------------------------------------------------
    # Supplementary: Proximal beats all alternatives
    # ------------------------------------------------------------------
    def test_proximal_beats_all_alternatives(self):
        """
        Supplementary check: Proximal G-estimation (strong) should beat
        ALL alternative estimators under strong proxies.
        """
        proximal_mae = self._get('Proximal G-estimation', 'strong')['mae']

        for alt in ['Naive OLS', 'Population AIPW', 'Standard G-estimation']:
            alt_mae = self._get(alt, 'strong')['mae']
            self.assertLess(
                proximal_mae, alt_mae,
                f"Proximal MAE ({proximal_mae:.3f}) should beat "
                f"{alt} MAE ({alt_mae:.3f})"
            )
            reduction = 1.0 - (proximal_mae / alt_mae)
            print(f"✓ Proximal ({proximal_mae:.3f}) beats {alt} "
                  f"({alt_mae:.3f}) by {reduction:.1%}")

    # ------------------------------------------------------------------
    # Diagnostic: Standard G-estimation = Naive OLS identity
    # ------------------------------------------------------------------
    def test_standard_gest_equals_naive_ols(self):
        """
        Diagnostic check: After removing propensity weighting (Fix 1),
        Standard G-estimation should produce results very close to Naive OLS.

        In this DGP, treatment A is continuous and confounded. Without
        proxy variables or valid propensity scores, per-patient partialling-out
        OLS reduces to ordinary OLS. These two estimators are effectively
        the same method, and the results table should reflect this.

        See PAPER_CLAIMS.md §"Standard G-estimation provides a meaningful
        comparison" for reporting guidance.
        """
        for proxy in ['strong', 'weak']:
            naive_mae = self._get('Naive OLS', proxy)['mae']
            std_mae = self._get('Standard G-estimation', proxy)['mae']

            # They should be very close (within 5% relative)
            if naive_mae > 0:
                rel_diff = abs(naive_mae - std_mae) / naive_mae
            else:
                rel_diff = abs(naive_mae - std_mae)

            print(f"\n  Diagnostic ({proxy}): Naive MAE = {naive_mae:.3f}, "
                  f"Standard G-est MAE = {std_mae:.3f}, "
                  f"relative diff = {rel_diff:.1%}")

            if rel_diff < 0.05:
                print(f"  ⚠ These estimators are effectively identical in "
                      f"this DGP.")
                print(f"    Consider removing Standard G-estimation from the "
                      f"comparison table")
                print(f"    or relabeling as 'Per-patient OLS "
                      f"(no confounding adjustment)'.")


if __name__ == '__main__':
    unittest.main(verbosity=2)
