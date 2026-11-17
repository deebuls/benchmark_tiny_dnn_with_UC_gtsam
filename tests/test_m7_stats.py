"""
Unit tests for Milestone 7: Experiment Matrix and Statistical Testing.
Verifies bootstrap confidence intervals, TOST equivalence testing, and interaction permutation tests.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.stats import (
    compute_paired_bootstrap_ci,
    tost_equivalence_test,
    permutation_test_interaction,
)


class TestM7Statistics(unittest.TestCase):
    def test_paired_bootstrap_ci_coverage(self):
        # Two identical distributions with 0 true difference
        np.random.seed(42)
        a = np.random.normal(1.0, 0.1, size=50)
        b = a + np.random.normal(0.0, 0.02, size=50)

        ci = compute_paired_bootstrap_ci(a, b, num_bootstrap=500, alpha=0.05)
        # CI should enclose 0
        self.assertLess(ci["ci_lower"], 0.05)
        self.assertGreater(ci["ci_upper"], -0.05)

    def test_tost_equivalence_detected(self):
        # Almost identical samples should pass equivalence test with 15% margin
        np.random.seed(42)
        ref = np.random.normal(0.20, 0.01, size=60)
        test = ref + np.random.normal(0.001, 0.002, size=60)

        res = tost_equivalence_test(test, ref, margin_rel=0.15)
        self.assertTrue(res["is_equivalent"], "TOST should declare near-identical samples equivalent")

    def test_tost_nonequivalence_detected(self):
        # Samples that differ by 50% should fail equivalence test with 15% margin
        ref = np.full(50, 0.20)
        test = np.full(50, 0.35)

        res = tost_equivalence_test(test, ref, margin_rel=0.15)
        self.assertFalse(res["is_equivalent"], "TOST should reject equivalence when difference is large")

    def test_permutation_interaction(self):
        # Synthetic DataFrame with a known interaction effect
        rows = []
        for h in ["gaussian", "student_t"]:
            for bb in ["resnet9", "resnet50"]:
                for rep in range(15):
                    # Interaction: student_t brings resnet9 very close to resnet50, gaussian has large gap
                    base = 0.10 if bb == "resnet50" else (0.12 if h == "student_t" else 0.35)
                    rows.append({
                        "head": h,
                        "backbone": bb,
                        "ate_rmse": base + np.random.normal(0, 0.01),
                    })
        df = pd.DataFrame(rows)
        res = permutation_test_interaction(df, metric="ate_rmse", num_permutations=200)
        self.assertTrue(res["is_significant"], "Permutation test should detect strong interaction")


if __name__ == "__main__":
    unittest.main()
