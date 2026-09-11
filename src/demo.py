"""The 60-second artefact: the pre-period fit you can inspect, and a floor.

Run: python -m src.demo
"""

from __future__ import annotations

from .generator import generate, naive_pre_post
from .scm import (
    PoorPreFit, estimate_lift, fit_synthetic_control, placebo_inference,
)

TRUE_LIFT = 0.04


def sparkline(values: list[float], reference: list[float]) -> str:
    """A crude inline chart so the pre-period fit is visible, not asserted."""
    # ASCII rather than Unicode blocks: the Windows console defaults to
    # cp1252 and a demo that crashes on the reviewer's machine is not a demo.
    blocks = "._-=+*#@"
    lo = min(min(values), min(reference))
    hi = max(max(values), max(reference))
    span = (hi - lo) or 1.0
    return "".join(
        blocks[min(7, int((v - lo) / span * 7))] for v in values)


def main() -> None:
    panel = generate(true_lift=TRUE_LIFT, seed=1)
    treated = panel.treated[0]

    print("\n  GEOLIFT - the platform says 6.2x. The holdout says nothing moved.")
    print("  " + "=" * 74)
    print(f"  {len(panel.series)} regions, {panel.weeks} weeks, "
          f"{panel.pre_period} pre-period. Treated: {treated}.")
    print(f"  TRUE lift applied by the generator: {TRUE_LIFT:.1%}\n")

    naive = naive_pre_post(panel)
    print("  WHAT A BEFORE/AFTER COMPARISON REPORTS")
    print("  " + "-" * 74)
    print(f"    {naive:+.1%}")
    print("    It attributes the whole national movement between the two")
    print("    periods to the campaign, because nothing in it separates the")
    print("    two. Off by "
          f"{abs(naive - TRUE_LIFT) / TRUE_LIFT:.0f}x.\n")

    sc = fit_synthetic_control(panel, treated)
    est = estimate_lift(panel, treated)

    print("  THE SYNTHETIC CONTROL")
    print("  " + "-" * 74)
    print("    donor weights (sparse, so a human can sanity-check them):")
    for region, weight in sc.top_donors(5):
        print(f"      {region}  {weight:.3f}")
    print(f"    pre-period RMSE: {sc.normalised_rmse:.2%} of scale "
          f"(gate: 5.5%)\n")

    pre = list(range(panel.pre_period))
    actual_pre = [panel.series[treated][w] for w in pre]
    synth_pre = [sc.predict(panel, w) for w in pre]
    post = list(panel.post_weeks())
    actual_post = [panel.series[treated][w] for w in post]
    synth_post = [sc.predict(panel, w) for w in post]

    print("    PRE-PERIOD (this is the credibility check - they must track)")
    print(f"      actual    {sparkline(actual_pre, synth_pre)}")
    print(f"      synthetic {sparkline(synth_pre, actual_pre)}")
    print("    POST-PERIOD (the gap is the effect)")
    print(f"      actual    {sparkline(actual_post, synth_post)}")
    print(f"      synthetic {sparkline(synth_post, actual_post)}")

    print(f"\n    estimated lift: {est.lift:+.2%}   (truth {TRUE_LIFT:.1%})")
    print(f"    incremental revenue: {est.incremental:,.0f}")
    print(f"    detection floor: {est.minimum_detectable_lift:.2%}   "
          f"above it: {est.above_detection_floor}\n")

    result = placebo_inference(panel, treated, max_placebos=15)
    worse = sum(1 for r in result.placebo_ratios if r >= result.treated_ratio)
    print("  PLACEBO INFERENCE")
    print("  " + "-" * 74)
    print(f"    treated post/pre error ratio: {result.treated_ratio:.2f}")
    print(f"    {len(result.placebo_ratios)} donors run as pretend-treated; "
          f"{worse} scored as high")
    print(f"    p = {result.p_value:.3f}")
    print("    A t-test over weekly observations would assume independence")
    print("    across time that a trending seasonal series plainly violates,")
    print("    and would return a very small p-value for a very ordinary")
    print("    region.\n")

    print("  THE TWO REFUSALS")
    print("  " + "-" * 74)
    bad = generate(regions=4, noise=0.25, seed=7)
    try:
        estimate_lift(bad, bad.treated[0])
    except PoorPreFit as exc:
        print(f"    poor pre-fit  -> {exc}")

    small = generate(true_lift=0.02, seed=1)
    small_est = estimate_lift(small, small.treated[0])
    print(f"    below the floor -> a {0.02:.0%} lift measured at "
          f"{small_est.lift:+.2%} with a floor of "
          f"{small_est.minimum_detectable_lift:.2%}")
    print("                       the point estimate looks like a clean")
    print("                       recovery and is mostly luck.\n")


if __name__ == "__main__":
    main()
