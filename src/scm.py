"""Synthetic control, with a pre-period fit gate and placebo inference.

THE DIFFERENTIATOR LIVES HERE.

A geo test that compares treated against untreated regions after the fact
attributes an ordinary regional demand difference to the campaign. Synthetic
control fixes that by building a weighted combination of donor regions that
tracks the treated region BEFORE the campaign, and using it as the
counterfactual afterwards.

Two things make it credible, and both are routinely skipped:

  1. THE PRE-PERIOD FIT GATE. If the synthetic control does not track the
     treated region before treatment, it will not track it after, and the
     "lift" is just fit error. The gate is enforced in code, not advised in a
     footnote.

  2. PLACEBO INFERENCE. With one treated unit there is no sampling
     distribution to appeal to, so a t-test is not available - it assumes
     independence across time that a trending series does not have. Instead
     the same estimator is run pretending each DONOR was treated. If the real
     effect is not unusual against that distribution, it is not evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .generator import Panel


def _project_to_simplex(v: list[float]) -> list[float]:
    """Nearest point on the probability simplex.

    Weights are constrained to be non-negative and sum to one. That is not
    decoration: unconstrained regression will happily extrapolate with large
    positive and negative weights, fit the pre-period almost perfectly, and
    produce a counterfactual outside anything the data supports.
    """
    n = len(v)
    u = sorted(v, reverse=True)
    css = 0.0
    rho = 0
    theta = 0.0
    for i in range(n):
        css += u[i]
        t = (css - 1.0) / (i + 1)
        if u[i] - t > 0:
            rho = i + 1
            theta = t
    return [max(0.0, x - theta) for x in v]


@dataclass(frozen=True)
class SyntheticControl:
    treated: str
    weights: dict[str, float]
    pre_rmse: float
    #: Root mean square of the treated series in the pre-period, so the fit
    #: error can be judged as a proportion rather than in currency units.
    pre_scale: float

    @property
    def normalised_rmse(self) -> float:
        return self.pre_rmse / self.pre_scale if self.pre_scale else math.inf

    def predict(self, panel: Panel, week: int) -> float:
        return sum(
            w * panel.series[donor][week] for donor, w in self.weights.items())

    def top_donors(self, n: int = 5) -> list[tuple[str, float]]:
        return sorted(self.weights.items(), key=lambda kv: -kv[1])[:n]


def fit_synthetic_control(
    panel: Panel, treated: str, donors: list[str] | None = None,
    iterations: int = 400,
) -> SyntheticControl:
    """Projected gradient descent on the pre-period fit.

    The gradient is computed from a precomputed Gram matrix rather than by
    sweeping the panel each iteration. For a 78-week pre-period and 39 donors
    that is the difference between 12 million inner operations per fit and
    about 900,000 - and since placebo inference refits the model once per
    donor, the naive version makes the honest inference step unaffordable and
    quietly encourages skipping it.
    """
    donors = donors or [r for r in panel.series if r != treated]
    n_pre = panel.pre_period
    y = [panel.series[treated][w] for w in range(n_pre)]
    columns = [[panel.series[d][w] for w in range(n_pre)] for d in donors]

    scale = math.sqrt(sum(v * v for v in y) / len(y))
    inv = 1.0 / scale
    columns = [[v * inv for v in col] for col in columns]
    y_scaled = [v * inv for v in y]

    k = len(donors)
    # Gram matrix and the cross-product vector, computed once.
    gram = [[0.0] * k for _ in range(k)]
    for i in range(k):
        ci = columns[i]
        for j in range(i, k):
            cj = columns[j]
            value = sum(a * b for a, b in zip(ci, cj)) / n_pre
            gram[i][j] = value
            gram[j][i] = value
    cross = [sum(a * b for a, b in zip(col, y_scaled)) / n_pre
             for col in columns]

    # Step size from the Gram matrix trace: an upper bound on the curvature,
    # so the descent is stable without hand-tuning a learning rate.
    lipschitz = 2.0 * max(sum(abs(v) for v in row) for row in gram)
    step = 1.0 / lipschitz if lipschitz > 0 else 0.1

    # FISTA: accelerated projected gradient. Plain gradient descent needed
    # ~5,000 iterations to converge here, and since placebo inference refits
    # the model once per donor, that made honest inference too slow to run -
    # which is exactly how the inference step ends up being skipped. The
    # momentum term gets the same fit in a few hundred.
    weights = [1.0 / k] * k
    momentum = list(weights)
    t_k = 1.0
    for _ in range(iterations):
        gradient = [
            2.0 * (sum(gram[i][j] * momentum[j] for j in range(k)) - cross[i])
            for i in range(k)
        ]
        nxt = _project_to_simplex(
            [w - step * g for w, g in zip(momentum, gradient)])
        t_next = (1.0 + math.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        factor = (t_k - 1.0) / t_next
        momentum = [
            n + factor * (n - w) for n, w in zip(nxt, weights)
        ]
        weights = nxt
        t_k = t_next

    errors = [
        sum(w * col[t] for w, col in zip(weights, columns)) - y_scaled[t]
        for t in range(n_pre)
    ]
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors)) * scale
    return SyntheticControl(
        treated=treated,
        weights={d: w for d, w in zip(donors, weights) if w > 1e-6},
        pre_rmse=rmse,
        pre_scale=scale,
    )


class PoorPreFit(Exception):
    """The synthetic control does not track the treated region beforehand.

    Reported rather than worked around. A counterfactual that cannot reproduce
    the past has no claim on the present, and the lift it implies is fit error
    wearing a result's clothing.
    """


#: A synthetic control whose pre-period error exceeds this share of the
#: treated region's scale is not usable.
#:
#: The threshold has to sit ABOVE the irreducible week-to-week noise in the
#: series, or no fit can ever pass and the gate rejects everything - which is
#: as useless as a gate that accepts everything. Regional weekly sales carry
#: roughly 3% idiosyncratic variation, so a good synthetic control lands
#: around 3-4% and this is set just above it. Calibrate against your own data
#: rather than inheriting this number.
MAX_PRE_RMSE = 0.055
#: And it needs enough pre-period weeks for that fit to mean anything.
MIN_PRE_WEEKS = 12


#: How many multiples of the pre-period fit error an effect must exceed
#: before it can be distinguished from that error.
MDE_MULTIPLE = 1.5


@dataclass(frozen=True)
class LiftEstimate:
    treated: str
    #: Proportional lift over the post period.
    lift: float
    incremental: float
    pre_rmse: float
    post_rmse: float
    #: post-period error divided by pre-period error: the standard
    #: synthetic-control test statistic.
    ratio: float
    #: Fit error as a share of the treated region's scale.
    normalised_pre_rmse: float

    @property
    def minimum_detectable_lift(self) -> float:
        """The smallest lift this fit could distinguish from its own error.

        This is the number that is almost never reported, and it is the one
        that decides whether the experiment was capable of answering the
        question. A synthetic control with 5% pre-period error cannot see a 4%
        lift - the estimate it produces is mostly fit error, and it will be
        confidently wrong in whichever direction the noise happened to fall.

        Reporting it turns "we measured a 12% lift" into "we measured a 12%
        lift with a floor of 7%, so treat the magnitude with suspicion".
        """
        return MDE_MULTIPLE * self.normalised_pre_rmse

    @property
    def above_detection_floor(self) -> bool:
        return abs(self.lift) >= self.minimum_detectable_lift


def estimate_lift(
    panel: Panel, treated: str, donors: list[str] | None = None,
    enforce_fit: bool = True,
) -> LiftEstimate:
    if panel.pre_period < MIN_PRE_WEEKS:
        raise PoorPreFit(
            f"only {panel.pre_period} pre-period weeks; at least "
            f"{MIN_PRE_WEEKS} are needed before a fit means anything")

    sc = fit_synthetic_control(panel, treated, donors)
    if enforce_fit and sc.normalised_rmse > MAX_PRE_RMSE:
        raise PoorPreFit(
            f"pre-period RMSE is {sc.normalised_rmse:.1%} of scale "
            f"(limit {MAX_PRE_RMSE:.0%}); the synthetic control does not track "
            f"{treated} before treatment, so it cannot be trusted after")

    observed = 0.0
    counterfactual = 0.0
    post_errors: list[float] = []
    for w in panel.post_weeks():
        actual = panel.series[treated][w]
        predicted = sc.predict(panel, w)
        observed += actual
        counterfactual += predicted
        post_errors.append(actual - predicted)

    post_rmse = math.sqrt(sum(e * e for e in post_errors) / len(post_errors))
    return LiftEstimate(
        treated=treated,
        lift=(observed / counterfactual - 1) if counterfactual else 0.0,
        incremental=observed - counterfactual,
        pre_rmse=sc.pre_rmse,
        post_rmse=post_rmse,
        ratio=post_rmse / sc.pre_rmse if sc.pre_rmse else math.inf,
        normalised_pre_rmse=sc.normalised_rmse,
    )


# ---------------------------------------------------------------------------
# Placebo inference
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlaceboResult:
    treated_ratio: float
    placebo_ratios: list[float]
    #: Share of placebos with a ratio at least as extreme as the treated one.
    p_value: float
    lift: float
    ci_low: float
    ci_high: float

    @property
    def significant(self) -> bool:
        return self.p_value <= 0.10


def placebo_inference(
    panel: Panel, treated: str, max_placebos: int = 30,
) -> PlaceboResult:
    """Run the same estimator pretending each donor was treated.

    This is where the p-value comes from. A t-test over weekly observations
    would assume independence across time, which a trending, seasonal series
    plainly violates - and it produces very small p-values for very ordinary
    regions.
    """
    real = estimate_lift(panel, treated, enforce_fit=False)

    donors = [r for r in panel.series if r not in panel.treated][:max_placebos]
    ratios: list[float] = []
    lifts: list[float] = []
    for donor in donors:
        others = [r for r in panel.series
                  if r != donor and r not in panel.treated]
        if len(others) < 5:
            continue
        try:
            placebo = estimate_lift(panel, donor, donors=others,
                                    enforce_fit=False)
        except PoorPreFit:
            continue
        ratios.append(placebo.ratio)
        lifts.append(placebo.lift)

    extreme = sum(1 for r in ratios if r >= real.ratio)
    p = (extreme + 1) / (len(ratios) + 1)

    # The placebo lift distribution is the reference for how large an
    # apparent effect a region can show for no reason at all.
    spread = _quantiles(lifts, [0.05, 0.95]) if lifts else (0.0, 0.0)
    return PlaceboResult(
        treated_ratio=real.ratio,
        placebo_ratios=ratios,
        p_value=p,
        lift=real.lift,
        ci_low=real.lift - (spread[1] - spread[0]) / 2,
        ci_high=real.lift + (spread[1] - spread[0]) / 2,
    )


def _quantiles(values: list[float], qs: list[float]) -> tuple[float, ...]:
    s = sorted(values)
    out = []
    for q in qs:
        idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
        out.append(s[idx])
    return tuple(out)
