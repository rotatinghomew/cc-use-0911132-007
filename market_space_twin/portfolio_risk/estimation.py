"""Estimate collection performance, common factors and dependence.

Contract facts are separated from collection behaviour:

* collectible rent for a month is ``reported_due - explicit_concession``.  A
  concession is relief, not arrears, and must not be read as a missed payment;
* if the reported due differs from the lease's prorated contract rent the month
  is flagged as a contract-basis change and the reported due is used, so the
  performance ratio stays comparable;
* a month without a usable cash-flow fact stays *missing*; it is never filled
  with zero and never silently dropped from the coverage count.

Dependence is estimated with a sector common factor plus idiosyncratic
residuals.  Within-sector loading is shrunk from the observed panel.  Cross-
sector factor correlation is only estimated when there is enough overlap;
otherwise a structural prior and an identification band are reported.  The
central estimate is never forced to zero, so independence is not assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, median, variance
from typing import Any

from market_space_twin.portfolio_risk.model import (
    CashFlow,
    PortfolioInput,
    RiskInputError,
    month_fraction,
    month_start,
    next_month,
)

PRIOR_CROSS_CORRELATION = 0.30
PRIOR_COMMON_SHARE = 0.40
CROSS_OVERLAP_MIN = 6
CROSS_SHRINK_MONTHS = 10
MIN_FACTOR_MONTHS = 4
MIN_PAIR_MONTHS = 6
MEAN_SHRINK_OBSERVATIONS = 3.0
VARIANCE_SHRINK_OBSERVATIONS = 3.0


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _round(value: float, digits: int = 6) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(value, digits)


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    mx, my = mean(x), mean(y)
    dx = [value - mx for value in x]
    dy = [value - my for value in y]
    vx = sum(value * value for value in dx)
    vy = sum(value * value for value in dy)
    if vx <= 0 or vy <= 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / math.sqrt(vx * vy)


def _pooled_variance(samples: list[list[float]]) -> float | None:
    groups = [values for values in samples if len(values) >= 2]
    if not groups:
        return None
    numerator = sum((len(values) - 1) * variance(values) for values in groups)
    denominator = sum(len(values) - 1 for values in groups)
    if denominator <= 0:
        return None
    return numerator / denominator


def stress_parameters(mean_rate: float, variance_rate: float) -> dict[str, float]:
    """Two-state monthly collection model that matches the observed mean and sd.

    A month is either collected in full or is a stressed month with collection
    rate ``1 - depth``; the stressed month occurs with probability ``q``.
    Solving ``mean = 1 - q*depth`` and ``var = q(1-q)*depth^2`` gives
    ``depth = (var + shortfall^2) / shortfall`` and ``q = shortfall/depth``.

    This keeps every simulated rate inside [0, 1] and reproduces the observed
    mean exactly, unlike a normal model that puts mass above 100% collection
    when the observed dip is severe relative to the sample size.
    """
    mean_rate = min(max(mean_rate, 0.0), 1.0)
    variance_rate = max(variance_rate, 0.0)
    shortfall = 1.0 - mean_rate
    if shortfall <= 1e-9 or variance_rate <= 1e-12:
        return {
            "stress_probability": 0.0,
            "stress_depth": 0.0,
            "mean_shortfall": round(shortfall, 6),
            "variance_used": 0.0,
            "variance_capped_to_feasible": False,
        }
    feasible = shortfall * (1.0 - shortfall)
    capped = variance_rate > feasible + 1e-12
    variance_rate = min(variance_rate, feasible)
    depth = (variance_rate + shortfall * shortfall) / shortfall
    depth = min(depth, 1.0)
    q = shortfall / depth
    return {
        "stress_probability": round(_clip(q, 0.0, 1.0), 6),
        "stress_depth": round(depth, 6),
        "mean_shortfall": round(shortfall, 6),
        "variance_used": round(variance_rate, 8),
        "variance_capped_to_feasible": capped,
    }


@dataclass(slots=True)
class TenantObservation:
    tenant_id: str
    sector: str
    month: datetime
    rate: float
    due: float
    received: float
    concession: float
    collectible: float


@dataclass(slots=True)
class TenantEstimate:
    tenant_id: str
    sector: str
    mean_rate: float
    variance_rate: float
    common_share: float
    history_months: int
    covered_months: int
    missing_months: int
    mean_source: str
    variance_source: str
    dependence_source: str
    mean_interval: tuple[float, float]
    mean_interval_source: str
    raw_mean: float | None = None
    raw_variance: float | None = None
    own_common_share: float | None = None
    factor_months: int = 0
    notes: list[str] = field(default_factory=list)

    def stress(self) -> dict[str, float]:
        return stress_parameters(self.mean_rate, self.variance_rate)

    def as_dict(self) -> dict[str, Any]:
        stress = self.stress()
        return {
            "tenant_id": self.tenant_id,
            "sector": self.sector,
            "history_months": self.history_months,
            "covered_months": self.covered_months,
            "missing_months": self.missing_months,
            "mean_collection_rate": _round(self.mean_rate, 4),
            "monthly_rate_sd": _round(math.sqrt(self.variance_rate), 4),
            "stress_probability_monthly": stress["stress_probability"],
            "stress_depth": stress["stress_depth"],
            "common_share": _round(self.common_share, 4),
            "mean_source": self.mean_source,
            "variance_source": self.variance_source,
            "dependence_source": self.dependence_source,
            "mean_interval": [_round(self.mean_interval[0], 4), _round(self.mean_interval[1], 4)],
            "mean_interval_source": self.mean_interval_source,
            "own_mean": None if self.raw_mean is None else _round(self.raw_mean, 4),
            "own_rate_sd": None if self.raw_variance is None else _round(math.sqrt(self.raw_variance), 4),
            "own_common_share": None if self.own_common_share is None else _round(self.own_common_share, 4),
            "factor_months": self.factor_months,
            "notes": self.notes,
        }


@dataclass(slots=True)
class EstimationResult:
    tenants: dict[str, TenantEstimate]
    sectors: dict[str, dict[str, Any]]
    matrix_ids: list[str]
    correlation_matrix: list[list[float]]
    within_sector: list[dict[str, Any]]
    cross_sector: list[dict[str, Any]]
    identification: dict[str, Any]
    data_quality: dict[str, Any]
    observations: list[TenantObservation]


def _month_coverage(portfolio: PortfolioInput) -> dict[tuple[str, str], float]:
    """Prorated contract rent expected for each tenant-month covered by a lease."""
    coverage: dict[tuple[str, str], float] = {}
    for lease in portfolio.leases:
        month = month_start(lease.start)
        while month < lease.end:
            fraction = month_fraction(lease.start, lease.end, month)
            if fraction > 0:
                key = (lease.tenant_id, month.strftime("%Y-%m"))
                coverage[key] = coverage.get(key, 0.0) + fraction * lease.monthly_contract
            month = next_month(month)
    return coverage


def _collect_observations(portfolio: PortfolioInput) -> tuple[list[TenantObservation], dict[str, Any]]:
    coverage = _month_coverage(portfolio)
    lease_sector: dict[str, str] = {}
    for lease in portfolio.leases:
        previous = lease_sector.get(lease.tenant_id)
        if previous is not None and previous != lease.sector:
            raise RiskInputError(
                f"tenant {lease.tenant_id} appears in more than one sector ({previous}, {lease.sector})"
            )
        lease_sector[lease.tenant_id] = lease.sector

    observations: list[TenantObservation] = []
    withheld: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    outside_lease: list[dict[str, Any]] = []
    basis_adjustments: list[dict[str, Any]] = []
    concession_months = 0

    all_cashflows: list[CashFlow] = list(portfolio.cashflows)
    for candidate in portfolio.candidates.values():
        all_cashflows.extend(candidate.history)

    for row in all_cashflows:
        month_label = row.month.strftime("%Y-%m")
        if row.available_at is None or row.available_at > portfolio.as_of:
            withheld.append(
                {
                    "tenant_id": row.tenant_id,
                    "month": month_label,
                    "available_at": None if row.available_at is None else row.available_at.isoformat(),
                    "reason": "available_at missing" if row.available_at is None else "arrives after as_of",
                }
            )
            continue
        if row.tenant_id not in lease_sector and row.tenant_id not in portfolio.candidates:
            raise RiskInputError(f"cashflow tenant {row.tenant_id} has neither a lease nor a candidate record")
        if not row.within_lease and row.tenant_id in lease_sector:
            outside_lease.append({"tenant_id": row.tenant_id, "month": month_label})
            continue
        if row.due is None or row.received is None:
            incomplete.append({"tenant_id": row.tenant_id, "month": month_label, "reason": "due or received missing"})
            continue
        if row.concession is None:
            incomplete.append(
                {
                    "tenant_id": row.tenant_id,
                    "month": month_label,
                    "reason": "concession unknown; collectible rent cannot be computed without assuming zero",
                }
            )
            continue
        collectible = row.due - row.concession
        if collectible <= 0:
            incomplete.append(
                {
                    "tenant_id": row.tenant_id,
                    "month": month_label,
                    "reason": "collectible rent is non-positive after explicit concession",
                }
            )
            continue
        expected_contract = coverage.get((row.tenant_id, month_label))
        if expected_contract is not None and abs(row.due - expected_contract) > max(1.0, abs(expected_contract) * 0.005):
            basis_adjustments.append(
                {
                    "tenant_id": row.tenant_id,
                    "month": month_label,
                    "reported_due": _round(row.due, 4),
                    "lease_prorated_contract": _round(expected_contract, 4),
                    "treatment": "use reported due as the contract basis for this month",
                }
            )
        if row.concession > 0:
            concession_months += 1
        rate = row.received / collectible
        sector = lease_sector.get(row.tenant_id)
        if sector is None:
            sector = portfolio.candidates[row.tenant_id].sector
        observations.append(
            TenantObservation(
                row.tenant_id,
                sector,
                month_start(row.month),
                rate,
                row.due,
                row.received,
                row.concession,
                collectible,
            )
        )

    # Coverage counts: whole/partial months in a lease up to as_of.  The last
    # month is included only when the lease and as_of cover part of it.
    covered: dict[tuple[str, str], float] = {}
    for lease in portfolio.leases:
        end = min(lease.end, portfolio.as_of)
        if end <= lease.start:
            continue
        month = month_start(lease.start)
        while month < end:
            fraction = month_fraction(lease.start, min(lease.end, portfolio.as_of), month)
            if fraction > 0:
                key = (lease.tenant_id, month.strftime("%Y-%m"))
                covered[key] = covered.get(key, 0.0) + fraction
            month = next_month(month)

    quality = {
        "withheld_late_or_unavailable_facts": withheld,
        "incomplete_facts_not_imputed": incomplete,
        "cashflows_outside_lease_ignored": outside_lease,
        "contract_basis_adjustments": basis_adjustments,
        "concession_months": concession_months,
        "concession_treatment": "received / (due - explicit concession); concession is relief, not arrears",
        "coverage_months_by_tenant": {
            tenant: sum(1 for (tid, _) in covered if tid == tenant) for tenant in sorted(lease_sector)
        },
        "observed_months_by_tenant": {
            tenant: sum(1 for item in observations if item.tenant_id == tenant) for tenant in sorted(lease_sector)
        },
        "missing_months_not_zero_filled": {
            tenant: max(
                0,
                sum(1 for (tid, _) in covered if tid == tenant)
                - sum(1 for item in observations if item.tenant_id == tenant),
            )
            for tenant in sorted(lease_sector)
        },
        "candidate_history_months": {key: len(value.history) for key, value in sorted(portfolio.candidates.items())},
        "candidate_history_treatment": "no-history candidates use a sector proxy and an interval; nothing is imputed as zero",
    }
    return observations, quality


def _factor_series(
    observations: list[TenantObservation],
    raw_means: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Median demeaned rate per sector-month, requiring at least two tenants."""
    grouped: dict[tuple[str, str], list[float]] = {}
    for item in observations:
        reference = raw_means[item.tenant_id]
        grouped.setdefault((item.sector, item.month.strftime("%Y-%m")), []).append(item.rate - reference)
    result: dict[str, dict[str, float]] = {}
    for (sector, month), values in grouped.items():
        if len(values) >= 2:
            result.setdefault(sector, {})[month] = median(values)
    return result


def _leave_one_out_factor(
    factor: dict[str, dict[str, float]],
    observations: list[TenantObservation],
    raw_means: dict[str, float],
    tenant_id: str,
) -> list[tuple[str, float]]:
    sector = next(item.sector for item in observations if item.tenant_id == tenant_id)
    values = factor.get(sector, {})
    if not values:
        return []
    rows = [
        item
        for item in observations
        if item.tenant_id == tenant_id and item.month.strftime("%Y-%m") in values
    ]
    result: list[tuple[str, float]] = []
    if not rows:
        return result
    # Recompute the LOO median from the sector's other tenants.
    grouped: dict[str, list[float]] = {}
    for item in observations:
        if item.sector != sector or item.tenant_id == tenant_id:
            continue
        grouped.setdefault(item.month.strftime("%Y-%m"), []).append(item.rate - raw_means[item.tenant_id])
    for item in rows:
        month = item.month.strftime("%Y-%m")
        others = grouped.get(month, [])
        if len(others) >= 1:
            result.append((month, median(others)))
    return result


def _sector_pooled(samples: dict[str, list[float]]) -> tuple[float | None, float | None]:
    means = [mean(values) for values in samples.values() if values]
    if not means:
        return None, None
    pooled_mean = mean(means)
    pooled_var = _pooled_variance(list(samples.values()))
    return pooled_mean, pooled_var


def estimate_dependence(portfolio: PortfolioInput) -> EstimationResult:
    observations, data_quality = _collect_observations(portfolio)
    if not observations:
        raise RiskInputError("no usable cash-flow observations at or before as_of; cannot estimate dependence")

    by_tenant: dict[str, list[TenantObservation]] = {}
    for item in observations:
        by_tenant.setdefault(item.tenant_id, []).append(item)
    sectors: dict[str, list[str]] = {}
    sector_of: dict[str, str] = {}
    for tenant, rows in by_tenant.items():
        sector_of[tenant] = rows[0].sector
        sectors.setdefault(rows[0].sector, []).append(tenant)

    raw_means: dict[str, float] = {}
    own_variances: dict[str, float] = {}
    for tenant, rows in by_tenant.items():
        values = [item.rate for item in rows]
        raw_means[tenant] = mean(values)
        if len(values) >= 2:
            own_variances[tenant] = variance(values)

    factors = _factor_series(observations, raw_means)

    sector_stats: dict[str, dict[str, Any]] = {}
    for sector, tenants in sectors.items():
        sector_means = {tenant: [item.rate for item in by_tenant.get(tenant, [])] for tenant in tenants}
        pooled_mean, pooled_var = _sector_pooled(sector_means)
        factor = factors.get(sector, {})
        factor_values = [factor[month] for month in sorted(factor)]
        factor_var = variance(factor_values) if len(factor_values) >= 2 else None
        sector_stats[sector] = {
            "tenants": sorted(tenants),
            "pooled_mean": pooled_mean,
            "pooled_variance": pooled_var,
            "factor_months": len(factor_values),
            "factor_sd": math.sqrt(factor_var) if factor_var is not None else None,
            "factor_identified": len(factor_values) >= 2,
        }

    global_samples: dict[str, list[float]] = {
        tenant: [item.rate for item in rows] for tenant, rows in by_tenant.items()
    }
    global_mean, global_var = _sector_pooled(global_samples)
    if global_mean is None:
        raise RiskInputError("no usable observations to build a pooled collection assumption")

    # Own common share from the leave-one-out sector factor.
    own_common: dict[str, float | None] = {}
    factor_counts: dict[str, int] = {}
    for tenant in by_tenant:
        pairs = _leave_one_out_factor(factors, observations, raw_means, tenant)
        factor_counts[tenant] = len(pairs)
        own_common[tenant] = None
        if len(pairs) >= MIN_FACTOR_MONTHS:
            rates = {item.month.strftime("%Y-%m"): item.rate for item in by_tenant[tenant]}
            x = [rates[month] - raw_means[tenant] for month, _ in pairs]
            y = [value for _, value in pairs]
            if variance(x) > 0 and variance(y) > 0:
                r = _pearson(x, y)
                if r is not None:
                    own_common[tenant] = _clip(r * r, 0.0, 0.98)

    sector_common: dict[str, float | None] = {}
    for sector, tenants in sectors.items():
        values = [own_common[tenant] for tenant in tenants if own_common[tenant] is not None]
        stats = sector_stats[sector]
        # A sector needs at least two tenants with usable history for a common
        # share inferred from tenant loadings.  A single observed tenant is only
        # one sample and must not set the dependence floor for every pair.
        if len(values) >= 2:
            sector_common[sector] = mean(values)
        elif stats["factor_identified"] and stats["pooled_variance"]:
            factor_share = _clip((stats["factor_sd"] ** 2) / stats["pooled_variance"], 0.05, 0.95)
            # A factor series shorter than the minimum shrinks toward the
            # structural prior instead of letting three months fix the floor.
            weight = stats["factor_months"] / (stats["factor_months"] + MIN_FACTOR_MONTHS)
            sector_common[sector] = weight * factor_share + (1 - weight) * PRIOR_COMMON_SHARE
        else:
            sector_common[sector] = None
    common_fallback = mean([value for value in sector_common.values() if value is not None]) if any(
        value is not None for value in sector_common.values()
    ) else PRIOR_COMMON_SHARE

    tenant_estimates: dict[str, TenantEstimate] = {}
    all_tenant_ids = sorted(by_tenant)
    for tenant in all_tenant_ids:
        rows = by_tenant[tenant]
        sector = sector_of[tenant]
        stats = sector_stats[sector]
        covered = data_quality["coverage_months_by_tenant"].get(tenant, len(rows))
        missing = data_quality["missing_months_not_zero_filled"].get(tenant, 0)

        # Mean: own mean shrunk to the sector pooled mean.
        weight = len(rows) / (len(rows) + MEAN_SHRINK_OBSERVATIONS)
        sector_mean = stats["pooled_mean"] if stats["pooled_mean"] is not None else global_mean
        mean_rate = weight * raw_means[tenant] + (1 - weight) * sector_mean
        if len(rows) >= 2:
            mean_source = "own_history_shrunk_to_sector"
        elif len(rows) == 1:
            mean_source = "single_month_shrunk_to_sector"
        else:
            mean_source = "sector_proxy"

        # Variance: own variance shrunk to sector pooled; never less than a
        # pooled floor when only one or two months exist.
        sector_var = stats["pooled_variance"]
        if sector_var is None:
            sector_var = global_var
        if sector_var is None:
            # The only available spread across tenant means gives a fallback.
            sector_var = max(
                variance([raw_means[t] for t in sectors[sector]]) if len(sectors[sector]) >= 2 else 0.0,
                1e-6,
            )
        if tenant in own_variances:
            vweight = len(rows) / (len(rows) + VARIANCE_SHRINK_OBSERVATIONS)
            variance_rate = vweight * own_variances[tenant] + (1 - vweight) * sector_var
            variance_source = "own_history_shrunk_to_sector"
        elif len(rows) == 1:
            variance_rate = sector_var
            variance_source = "single_month_uses_sector_variance"
        else:
            variance_rate = sector_var
            variance_source = "sector_proxy"
        variance_rate = max(variance_rate, 1e-9)

        # Dependence: own common share shrunk to the sector common share.
        sector_c = sector_common.get(sector)
        if sector_c is None:
            sector_c = common_fallback
        if own_common[tenant] is not None:
            cweight = factor_counts[tenant] / (factor_counts[tenant] + MIN_FACTOR_MONTHS)
            common_share = cweight * own_common[tenant] + (1 - cweight) * sector_c
            dependence_source = "own_factor_loading_shrunk_to_sector"
        else:
            common_share = sector_c
            dependence_source = "sector_proxy"
        common_share = _clip(common_share, 0.02, 0.98)

        # Mean interval.  Own observations use the shrunk variance and an
        # effective sample size; a proxy uses between-tenant dispersion so the
        # lack of tenant-specific history widens the band.
        se_own = math.sqrt(variance_rate / max(len(rows), 1))
        sector_tenants = sectors[sector]
        tenant_means = [raw_means[t] for t in sector_tenants]
        between = math.sqrt(variance(tenant_means)) if len(tenant_means) >= 2 else None
        n_total = sum(len(by_tenant[t]) for t in sector_tenants)
        se_proxy_components = [math.sqrt(variance_rate / max(n_total, 1))]
        if between is not None:
            se_proxy_components.append(between / math.sqrt(len(tenant_means)))
        se_proxy = max(se_proxy_components)
        if len(rows) >= 3:
            se = max(se_own, 0.5 * se_proxy)
            interval_source = "own_history_standard_error"
        elif len(rows) > 0:
            se = se_proxy
            interval_source = "short_history_widened_to_sector"
        else:
            se = se_proxy
            interval_source = "sector_proxy_no_tenant_history"
        interval = (_clip(mean_rate - 1.96 * se, 0.0, 1.0), _clip(mean_rate + 1.96 * se, 0.0, 1.0))

        tenant_estimates[tenant] = TenantEstimate(
            tenant_id=tenant,
            sector=sector,
            mean_rate=mean_rate,
            variance_rate=variance_rate,
            common_share=common_share,
            history_months=len(rows),
            covered_months=covered,
            missing_months=missing,
            mean_source=mean_source,
            variance_source=variance_source,
            dependence_source=dependence_source,
            mean_interval=interval,
            mean_interval_source=interval_source,
            raw_mean=raw_means[tenant],
            raw_variance=own_variances.get(tenant),
            own_common_share=own_common[tenant],
            factor_months=factor_counts[tenant],
        )

    # Candidates without history: sector proxy, never a zero and never hidden.
    for candidate_id, candidate in sorted(portfolio.candidates.items()):
        if candidate_id in tenant_estimates:
            continue
        sector = candidate.sector
        stats = sector_stats.get(sector)
        if stats is None or stats["pooled_mean"] is None:
            sector_mean = global_mean
            sector_var = global_var
            sector_c = common_fallback
            basis_tenants: list[str] = []
            factor_months = 0
            source = "global_proxy_no_sector_history"
        else:
            sector_mean = stats["pooled_mean"]
            sector_var = stats["pooled_variance"] or global_var
            sector_c = sector_common.get(sector) or common_fallback
            basis_tenants = stats["tenants"]
            factor_months = stats["factor_months"]
            source = "sector_proxy"
        sector_var = max(sector_var or 1e-6, 1e-6)
        tenant_means = [tenant_estimates[t].raw_mean for t in basis_tenants if tenant_estimates[t].raw_mean is not None]
        between = math.sqrt(variance(tenant_means)) if len(tenant_means) >= 2 else 0.0
        n_total = sum(tenant_estimates[t].history_months for t in basis_tenants) or 1
        se = max(math.sqrt(sector_var / n_total), between / math.sqrt(max(len(tenant_means), 1)))
        if len(basis_tenants) < 2 or n_total < 4:
            se = max(se, math.sqrt(sector_var))
        interval = (_clip(sector_mean - 1.96 * se, 0.0, 1.0), _clip(sector_mean + 1.96 * se, 0.0, 1.0))
        tenant_estimates[candidate_id] = TenantEstimate(
            tenant_id=candidate_id,
            sector=sector,
            mean_rate=sector_mean,
            variance_rate=sector_var,
            common_share=_clip(sector_c, 0.02, 0.98),
            history_months=0,
            covered_months=0,
            missing_months=0,
            mean_source=source,
            variance_source=source,
            dependence_source="sector_proxy_no_tenant_history",
            mean_interval=interval,
            mean_interval_source="sector_interval_not_a_probability",
            raw_mean=None,
            raw_variance=None,
            own_common_share=None,
            factor_months=factor_months,
            notes=["no historical cashflows; proxy is transparent, not observed tenant data"],
        )

    # Cross-sector correlation: try common factor overlap first, then long
    # tenant-pair overlap.  Anything short of the minimum is not identified.
    cross_sector: list[dict[str, Any]] = []
    sector_pairs: dict[tuple[str, str], float] = {}
    ordered_sectors = sorted(sector_stats)
    for i, left in enumerate(ordered_sectors):
        for right in ordered_sectors[i + 1 :]:
            left_factor = factors.get(left, {})
            right_factor = factors.get(right, {})
            common = sorted(set(left_factor) & set(right_factor))
            estimate: float | None = None
            method = "not_identified"
            overlap_months = len(common)
            sample_rho: float | None = None
            effective_months = len(common)
            if len(common) >= CROSS_OVERLAP_MIN:
                sample_rho = _pearson([left_factor[m] for m in common], [right_factor[m] for m in common])
                if sample_rho is not None:
                    method = "sector_factor_correlation"
            if sample_rho is None:
                pairs: list[tuple[float, int]] = []
                for left_tenant in sectors.get(left, []):
                    for right_tenant in sectors.get(right, []):
                        left_map = {item.month.strftime("%Y-%m"): item.rate for item in by_tenant[left_tenant]}
                        right_map = {item.month.strftime("%Y-%m"): item.rate for item in by_tenant[right_tenant]}
                        months = sorted(set(left_map) & set(right_map))
                        if len(months) >= MIN_PAIR_MONTHS:
                            rho = _pearson([left_map[m] for m in months], [right_map[m] for m in months])
                            if rho is not None:
                                pairs.append((rho, len(months)))
                if pairs:
                    overlap_months = sum(n for _, n in pairs)
                    sample_rho = sum(rho * n for rho, n in pairs) / sum(n for _, n in pairs)
                    effective_months = max(n for _, n in pairs)
                    method = "pooled_tenant_pair_correlation"
            if sample_rho is None:
                central = PRIOR_CROSS_CORRELATION
                lower, upper = 0.0, 0.60
                identified = False
                basis = "structural prior; overlap below minimum so the correlation is not identified"
            else:
                weight = effective_months / (effective_months + CROSS_SHRINK_MONTHS)
                central = _clip(weight * sample_rho + (1 - weight) * PRIOR_CROSS_CORRELATION, -0.6, 0.6)
                fisher = math.atanh(_clip(sample_rho, -0.95, 0.95))
                se = 1.0 / math.sqrt(max(effective_months - 3, 1))
                lower = _clip(math.tanh(fisher - 1.645 * se), -0.8, 0.8)
                upper = _clip(math.tanh(fisher + 1.645 * se), -0.8, 0.8)
                identified = True
                basis = (
                    "shrunk estimate with structural prior; band is a 90% Fisher interval on the "
                    "available overlap, widened by the prior"
                )
            sector_pairs[(left, right)] = central
            cross_sector.append(
                {
                    "sectors": [left, right],
                    "estimate": _round(central, 4),
                    "sample_estimate": None if sample_rho is None else _round(sample_rho, 4),
                    "method": method,
                    "overlap_months": overlap_months,
                    "identified": identified,
                    "prior_used": PRIOR_CROSS_CORRELATION,
                    "identification_band": [_round(lower, 4), _round(upper, 4)],
                    "basis": basis,
                    "not_independence": "central estimate is a prior/estimate mix; zero is only a band edge, not the assumption",
                }
            )

    def sector_corr(left: str, right: str) -> float:
        if left == right:
            return 1.0
        return sector_pairs.get((left, right), sector_pairs.get((right, left), PRIOR_CROSS_CORRELATION))

    matrix_ids = sorted(tenant_estimates)
    matrix: list[list[float]] = []
    for left in matrix_ids:
        a = tenant_estimates[left]
        row = []
        for right in matrix_ids:
            b = tenant_estimates[right]
            value = math.sqrt(a.common_share * b.common_share) * sector_corr(a.sector, b.sector)
            row.append(1.0 if left == right else _round(_clip(value, -0.98, 0.98), 4))
        matrix.append(row)

    within_sector: list[dict[str, Any]] = []
    for sector in ordered_sectors:
        tenants = sorted(sectors.get(sector, []))
        if len(tenants) < 2:
            within_sector.append(
                {
                    "sector": sector,
                    "tenants": tenants,
                    "sample_correlation": None,
                    "estimate": None,
                    "identified": False,
                    "overlap_months": 0,
                    "basis": "fewer than two tenants with history in this sector",
                }
            )
            continue
        pairs: list[tuple[float, int]] = []
        for i, left in enumerate(tenants):
            for right in tenants[i + 1 :]:
                left_map = {item.month.strftime("%Y-%m"): item.rate for item in by_tenant[left]}
                right_map = {item.month.strftime("%Y-%m"): item.rate for item in by_tenant[right]}
                months = sorted(set(left_map) & set(right_map))
                rho = _pearson([left_map[m] for m in months], [right_map[m] for m in months])
                if rho is not None:
                    pairs.append((rho, len(months)))
        sample = sum(rho * n for rho, n in pairs) / sum(n for _, n in pairs) if pairs else None
        overlap = max((n for _, n in pairs), default=0)
        identified = sample is not None and overlap >= MIN_PAIR_MONTHS
        within_sector.append(
            {
                "sector": sector,
                "tenants": tenants,
                "sample_correlation": None if sample is None else _round(sample, 4),
                "estimate": _round(sample, 4) if identified else None,
                "identified": identified,
                "overlap_months": overlap,
                "basis": (
                    "pairwise observed collection-rate correlation"
                    if identified
                    else "overlap below minimum; within-sector dependence comes from the sector common factor"
                ),
            }
        )

    identification = {
        "identified": [
            f"within-sector common factor for {sector} ({len(factors.get(sector, {}))} months)"
            for sector in ordered_sectors
            if len(factors.get(sector, {})) >= 2
        ],
        "not_identified": [
            "cross-sector correlation from factor overlap"
            for _ in [0]
            if all(not item["identified"] for item in cross_sector)
        ]
        + ([
            "candidate tenants have no tenant-specific history; only a sector proxy and an interval are reported"
        ] if portfolio.candidates else []),
        "minimum_overlap_months": {"cross_sector": CROSS_OVERLAP_MIN, "within_sector_pair": MIN_PAIR_MONTHS, "factor": MIN_FACTOR_MONTHS},
        "missing_data_policy": "missing tenant-month observations are excluded, not imputed as zero",
        "residual_assumption": (
            "idiosyncratic residuals are independent conditional on the sector factor; with this panel "
            "length their correlation is not estimable, so this is an assumption and not a measured value"
        ),
        "not_independence": "cross-sector central correlation is never set to zero; the band exposes the uncertainty",
        "as_of": portfolio.as_of.isoformat(),
    }

    return EstimationResult(
        tenants=tenant_estimates,
        sectors=sector_stats,
        matrix_ids=matrix_ids,
        correlation_matrix=matrix,
        within_sector=within_sector,
        cross_sector=cross_sector,
        identification=identification,
        data_quality=data_quality,
        observations=observations,
    )
