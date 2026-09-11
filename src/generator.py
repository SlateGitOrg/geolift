"""Regional weekly sales with a KNOWN planted lift.

Everything this project claims rests on being able to score the estimator
against a truth it did not see. The generator therefore builds a panel with
shared seasonality, region-specific trends and idiosyncratic noise - the three
things that make a naive before/after comparison wrong - and then applies a
lift of a size chosen here.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Panel:
    #: region -> weekly sales
    series: dict[str, list[float]]
    weeks: int
    pre_period: int
    treated: list[str]
    #: GROUND TRUTH: the multiplicative lift applied to treated regions in the
    #: post period. 0.04 means a 4% increase.
    true_lift: float

    @property
    def donors(self) -> list[str]:
        return [r for r in self.series if r not in self.treated]

    def post_weeks(self) -> range:
        return range(self.pre_period, self.weeks)

    def true_incremental_sales(self, region: str | None = None) -> float:
        """Incremental revenue actually caused, in the post period.

        Per region when one is named. Comparing a single region's estimate
        against the ALL-REGION total (or its average) is an easy mistake and
        produces an error that looks like estimator bias when it is really a
        units mismatch.
        """
        regions = [region] if region else list(self.treated)
        total = 0.0
        for r in regions:
            observed = sum(self.series[r][w] for w in self.post_weeks())
            counterfactual = observed / (1 + self.true_lift)
            total += observed - counterfactual
        return total


def generate(
    regions: int = 40,
    weeks: int = 104,
    pre_period: int = 78,
    treated_count: int = 2,
    true_lift: float = 0.04,
    seed: int = 20260911,
    #: Set to 0.0 for the null case: no campaign ran.
    noise: float = 0.03,
) -> Panel:
    rnd = random.Random(seed)
    names = [f"R{i:02d}" for i in range(regions)]
    treated = names[:treated_count]

    # A shared national factor: promotions, weather, the economy. This is what
    # a before/after comparison mistakes for campaign effect.
    national = []
    level = 0.0
    for w in range(weeks):
        level = 0.82 * level + rnd.gauss(0.0, 0.05)
        seasonal = 0.12 * math.sin(2 * math.pi * w / 52) + \
            0.05 * math.sin(2 * math.pi * w / 13)
        national.append(level + seasonal)

    series: dict[str, list[float]] = {}
    for region in names:
        base = math.exp(rnd.gauss(11.0, 0.45))
        # Region-specific trend. This is what makes a simple treated-vs-control
        # difference wrong even when the control regions look similar.
        trend = rnd.gauss(0.0, 0.0016)
        # How strongly this region tracks the national factor.
        loading = rnd.uniform(0.55, 1.45)

        values: list[float] = []
        for w in range(weeks):
            level = base * math.exp(trend * w + loading * national[w])
            level *= math.exp(rnd.gauss(0.0, noise))
            if region in treated and w >= pre_period:
                level *= 1 + true_lift
            values.append(level)
        series[region] = values

    return Panel(series, weeks, pre_period, treated, true_lift)


def naive_pre_post(panel: Panel) -> float:
    """What a before/after comparison reports, as a proportional lift.

    It attributes the entire national movement between the two periods to the
    campaign, which is why it is usually wrong and occasionally wrong by a
    factor of five.
    """
    pre = sum(
        panel.series[r][w] for r in panel.treated for w in range(panel.pre_period))
    post = sum(
        panel.series[r][w] for r in panel.treated for w in panel.post_weeks())
    pre_mean = pre / (panel.pre_period * len(panel.treated))
    post_mean = post / (
        (panel.weeks - panel.pre_period) * len(panel.treated))
    return post_mean / pre_mean - 1
