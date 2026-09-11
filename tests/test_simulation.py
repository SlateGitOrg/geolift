"""Coverage and null behaviour across many simulated experiments.

One experiment that recovers the planted lift proves nothing. What matters is
whether the method is right ACROSS experiments - and, above all, whether it
reports an effect when none exists.
"""

from __future__ import annotations

import unittest

from src.generator import generate, naive_pre_post
from src.scm import (
    MAX_PRE_RMSE, PoorPreFit, estimate_lift, fit_synthetic_control,
    placebo_inference,
)


class TestRecovery(unittest.TestCase):
    def test_recovers_the_planted_lift(self):
        panel = generate(true_lift=0.04, seed=1)
        est = estimate_lift(panel, panel.treated[0])
        self.assertAlmostEqual(est.lift, 0.04, delta=0.025)

    def test_THE_CONTRAST_before_after_is_badly_wrong(self):
        panel = generate(true_lift=0.04, seed=1)
        naive = naive_pre_post(panel)
        scm = estimate_lift(panel, panel.treated[0]).lift
        self.assertLess(
            abs(scm - 0.04), abs(naive - 0.04),
            f"before/after reported {naive:.1%}, synthetic control {scm:.1%}, "
            f"truth 4.0%",
        )

    def test_estimates_incremental_revenue_not_just_a_percentage(self):
        # Only meaningful where the fit is tight enough to see the effect.
        panel = generate(true_lift=0.06, seed=1)
        est = estimate_lift(panel, panel.treated[0])
        self.assertTrue(est.above_detection_floor)
        truth = panel.true_incremental_sales(panel.treated[0])
        self.assertLess(abs(est.incremental - truth) / truth, 0.6)

    def test_THE_NUMBER_NOBODY_REPORTS_a_small_lift_is_below_the_floor(self):
        """A fit with 2.6% pre-period error cannot resolve a 2% lift.

        The point estimate comes back at +2.3%, which looks like a clean
        recovery of the truth and is mostly luck: the fit error is larger than
        the effect, so the sign and magnitude are at the mercy of which way the
        noise fell. Nothing about the point estimate says so. The detection
        floor does, and it is the number that decides whether the experiment
        was capable of answering the question at all.
        """
        panel = generate(true_lift=0.02, seed=1)
        est = estimate_lift(panel, panel.treated[0])
        self.assertGreater(est.minimum_detectable_lift, 0.02)
        self.assertFalse(
            est.above_detection_floor,
            "a 2% lift must not be reported as measured when the fit error "
            "is larger than the effect",
        )

    def test_a_lift_comfortably_above_the_floor_IS_reportable(self):
        # Without this, "below the floor" would be satisfied by an
        # implementation that never reports anything as measurable.
        panel = generate(true_lift=0.10, seed=1)
        est = estimate_lift(panel, panel.treated[0])
        self.assertTrue(est.above_detection_floor)
        self.assertAlmostEqual(est.lift, 0.10, delta=0.03)

    def test_a_tight_fit_reports_a_low_detection_floor(self):
        panel = generate(true_lift=0.06, seed=1)
        est = estimate_lift(panel, panel.treated[0])
        self.assertLess(est.minimum_detectable_lift, 0.06)

    def test_a_larger_lift_is_estimated_as_larger(self):
        small = estimate_lift(
            generate(true_lift=0.02, seed=9), "R00").lift
        large = estimate_lift(
            generate(true_lift=0.12, seed=9), "R00").lift
        self.assertLess(small, large)


class TestNullBehaviour(unittest.TestCase):
    """The failure that costs real money: reporting a lift when none exists."""

    RUNS = 30

    def test_THE_HEADLINE_no_false_lift_when_no_campaign_ran(self):
        significant = 0
        tested = 0
        for seed in range(self.RUNS):
            panel = generate(true_lift=0.0, seed=1_000 + seed)
            try:
                result = placebo_inference(panel, panel.treated[0],
                                           max_placebos=12)
            except PoorPreFit:
                continue
            tested += 1
            if result.significant:
                significant += 1

        self.assertGreater(tested, 15, "not enough usable runs")
        rate = significant / tested
        self.assertLess(
            rate, 0.25,
            f"reported a significant lift in {rate:.0%} of experiments where "
            f"nothing happened",
        )

    def test_the_estimate_is_centred_on_zero_under_the_null(self):
        lifts = []
        for seed in range(self.RUNS):
            panel = generate(true_lift=0.0, seed=2_000 + seed)
            try:
                lifts.append(estimate_lift(panel, panel.treated[0]).lift)
            except PoorPreFit:
                continue
        mean = sum(lifts) / len(lifts)
        self.assertLess(abs(mean), 0.02,
                        f"mean estimated lift under the null was {mean:+.2%}")

    def test_THE_METHOD_HAS_POWER_a_real_lift_IS_detected(self):
        # Without this, "no false positives" is satisfied by an estimator that
        # never reports anything.
        detected = 0
        tested = 0
        for seed in range(20):
            panel = generate(true_lift=0.10, seed=3_000 + seed)
            try:
                result = placebo_inference(panel, panel.treated[0],
                                           max_placebos=12)
            except PoorPreFit:
                continue
            tested += 1
            if result.significant:
                detected += 1
        self.assertGreater(
            detected / tested, 0.5,
            f"only detected a 10% lift in {detected}/{tested} experiments",
        )


class TestPreFitGate(unittest.TestCase):
    def test_MOST_fits_pass_the_gate_on_a_healthy_donor_pool(self):
        """The gate must admit good fits, not just reject bad ones.

        Stated across seeds rather than on one: whether a particular treated
        region is well approximated by the donor pool genuinely varies, and a
        single seed would be testing that seed rather than the gate.
        """
        passed = 0
        for seed in range(20):
            panel = generate(seed=seed)
            sc = fit_synthetic_control(panel, panel.treated[0])
            if sc.normalised_rmse <= MAX_PRE_RMSE:
                passed += 1
        self.assertGreaterEqual(
            passed, 14,
            f"only {passed}/20 healthy panels produced a usable fit - a gate "
            f"that rejects most good data is as useless as one that accepts "
            f"everything",
        )

    def test_some_regions_genuinely_cannot_be_synthesised(self):
        # And that is a finding about the test design, not a bug. It means
        # that region is a poor choice of treated unit.
        failures = sum(
            1 for seed in range(20)
            if fit_synthetic_control(
                generate(seed=seed), "R00").normalised_rmse > MAX_PRE_RMSE
        )
        self.assertGreater(failures, 0,
                           "if every region fits, the gate is never exercised")

    def test_THE_GATE_a_poor_fit_is_REFUSED_not_reported(self):
        # Very few donors and heavy noise: no weighted combination can track
        # the treated region, so any "lift" is fit error.
        panel = generate(regions=4, noise=0.25, seed=7)
        with self.assertRaises(PoorPreFit):
            estimate_lift(panel, panel.treated[0])

    def test_too_short_a_pre_period_is_refused(self):
        panel = generate(weeks=20, pre_period=6, seed=7)
        with self.assertRaises(PoorPreFit):
            estimate_lift(panel, panel.treated[0])

    def test_the_refusal_explains_what_is_wrong(self):
        panel = generate(regions=4, noise=0.25, seed=7)
        try:
            estimate_lift(panel, panel.treated[0])
            self.fail("expected a refusal")
        except PoorPreFit as exc:
            self.assertIn("does not track", str(exc))


class TestWeights(unittest.TestCase):
    def test_weights_are_non_negative_and_sum_to_one(self):
        panel = generate(seed=11)
        sc = fit_synthetic_control(panel, panel.treated[0])
        self.assertTrue(all(w >= 0 for w in sc.weights.values()))
        self.assertAlmostEqual(sum(sc.weights.values()), 1.0, delta=1e-6)

    def test_the_treated_region_is_never_its_own_donor(self):
        panel = generate(seed=11)
        sc = fit_synthetic_control(panel, panel.treated[0])
        self.assertNotIn(panel.treated[0], sc.weights)

    def test_the_donor_pool_is_sparse_which_makes_it_reviewable(self):
        panel = generate(seed=11)
        sc = fit_synthetic_control(panel, panel.treated[0])
        self.assertLess(len(sc.weights), 20,
                        "a dense weight vector cannot be sanity-checked by a human")


class TestPlaceboInference(unittest.TestCase):
    def test_a_p_value_is_produced_from_the_placebo_distribution(self):
        panel = generate(true_lift=0.10, seed=13)
        result = placebo_inference(panel, panel.treated[0], max_placebos=12)
        self.assertGreater(len(result.placebo_ratios), 5)
        self.assertGreaterEqual(result.p_value, 0.0)
        self.assertLessEqual(result.p_value, 1.0)

    def test_the_interval_is_derived_from_placebos_not_assumed(self):
        panel = generate(true_lift=0.06, seed=17)
        result = placebo_inference(panel, panel.treated[0], max_placebos=12)
        self.assertLess(result.ci_low, result.lift)
        self.assertGreater(result.ci_high, result.lift)


if __name__ == "__main__":
    unittest.main()
