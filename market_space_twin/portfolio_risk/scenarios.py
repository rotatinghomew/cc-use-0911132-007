"""Exposure construction, scenario simulation and contribution analysis.

Three layers are kept apart:

* ``exposures`` are contractual facts: who occupies which unit, from when to
  when, at what contract rent.  Vacancies are their own intervals and never a
  tenant's exposure.
* ``historical_term`` loss distributions use the estimated means, variances and
  dependence from history.  They are model estimates on the committed term, not
  a traffic forecast and not a claim that next season equals last season.
* ``management_shocks`` and the structural drills are labelled assumptions.
  They are deterministic what-if calculations, never probabilities.

Replacement plans truncate the old lease at ``starts_at`` and start the new
exposure at ``starts_at``.  A gap left on a unit becomes a vacancy interval,
  not old or new tenant exposure.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime
import statistics
from statistics import mean, stdev
from typing import Any

from market_space_twin.portfolio_risk.estimation import (
    PRIOR_CROSS_CORRELATION,
    EstimationResult,
)
from market_space_twin.portfolio_risk.model import (
    Lease,
    PortfolioInput,
    Replacement,
    RiskInputError,
    ScenarioSpec,
    month_fraction,
    month_start,
    next_month,
    parse_time,
)
from market_space_twin.portfolio_risk.numerics import (
    bivariate_stress_probability,
    cholesky,
    clip,
    normal_quantile,
    psd_correlation,
)

DEFAULT_PATHS = 20000
SIMULATION_SEED = 20261030


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def _clip(value: float, low: float, high: float) -> float:
    return clip(value, low, high)


def _month_label(value: datetime) -> str:
    return value.strftime("%Y-%m")


@dataclass(slots=True)
class ExposureSegment:
    plan: str
    tenant_id: str
    sector: str
    unit_id: str
    lease_id: str | None
    replaces_lease_id: str | None
    proposed: bool
    start: datetime
    end: datetime
    monthly_contract: float
    fractions: list[tuple[datetime, float]]

    @property
    def exposed_rent(self) -> float:
        return sum(self.monthly_contract * fraction for _, fraction in self.fractions)

    @property
    def months(self) -> float:
        return sum(fraction for _, fraction in self.fractions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "sector": self.sector,
            "unit_id": self.unit_id,
            "lease_id": self.lease_id,
            "replaces_lease_id": self.replaces_lease_id,
            "proposed_lease": self.proposed,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "monthly_contract": _round(self.monthly_contract, 2),
            "active_months": _round(self.months, 4),
            "exposed_rent": _round(self.exposed_rent, 2),
            "month_fractions": [
                {"month": _month_label(month), "fraction": _round(fraction, 4)}
                for month, fraction in self.fractions
            ],
        }


@dataclass(slots=True)
class VacancySegment:
    plan: str
    unit_id: str
    start: datetime
    end: datetime
    reference_monthly_contract: float | None

    @property
    def months(self) -> float:
        return sum(fraction for _, fraction in split(self.start, self.end))

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "active_months": _round(self.months, 4),
            "attributed_to_tenant": False,
            "reference_monthly_contract": _round(self.reference_monthly_contract, 2),
            "note": "vacancy interval; no tenant exposure and no contract rent counted",
        }


def split(start: datetime, end: datetime) -> list[tuple[datetime, float]]:
    result: list[tuple[datetime, float]] = []
    if end <= start:
        return result
    month = month_start(start)
    while month < end:
        fraction = month_fraction(start, end, month)
        if fraction > 0:
            result.append((month, round(fraction, 8)))
        month = next_month(month)
    return result


@dataclass(slots=True)
class PlanExposure:
    name: str
    segments: list[ExposureSegment]
    vacancies: list[VacancySegment]
    horizon_start: datetime
    horizon_end: datetime
    replacement: Replacement | None
    shocks: dict[str, float]
    notes: list[str] = field(default_factory=list)

    @property
    def tenants(self) -> list[str]:
        return sorted({segment.tenant_id for segment in self.segments})

    def rent_by_month(self, tenant_id: str) -> dict[datetime, float]:
        result: dict[datetime, float] = {}
        for segment in self.segments:
            if segment.tenant_id != tenant_id:
                continue
            for month, fraction in segment.fractions:
                result[month] = result.get(month, 0.0) + segment.monthly_contract * fraction
        return result

    def total_rent(self) -> float:
        return sum(segment.exposed_rent for segment in self.segments)

    def tenant_exposure(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for segment in self.segments:
            result[segment.tenant_id] = result.get(segment.tenant_id, 0.0) + segment.exposed_rent
        return result

    def months(self) -> list[datetime]:
        labels = {month for segment in self.segments for month, _ in segment.fractions}
        return sorted(labels)

    def as_dict(self, estimates: EstimationResult) -> dict[str, Any]:
        return {
            "segments": [segment.as_dict() for segment in self.segments],
            "vacancies": [vacancy.as_dict() for vacancy in self.vacancies],
            "tenants": [
                {
                    "tenant_id": tenant,
                    "sector": estimates.tenants[tenant].sector,
                    "exposed_rent": _round(sum(segment.exposed_rent for segment in self.segments if segment.tenant_id == tenant), 2),
                    "history_months": estimates.tenants[tenant].history_months,
                    "estimate_source": estimates.tenants[tenant].mean_source,
                }
                for tenant in self.tenants
            ],
            "total_exposed_rent": _round(self.total_rent(), 2),
            "replacement": None
            if self.replacement is None
            else {
                "unit_id": self.replacement.unit_id,
                "new_tenant": self.replacement.new_tenant,
                "starts_at": self.replacement.starts_at.isoformat(),
                "ends_at": None if self.replacement.ends_at is None else self.replacement.ends_at.isoformat(),
            },
            "shocks_as_management_assumption": dict(self.shocks),
            "notes": self.notes,
        }


def normalised_plan(plan: PlanExposure, months: int = 12) -> PlanExposure:
    """Annual run-rate comparison: current roster and contract rents for 12 equal months.

    The committed term in the fixture is only a few days beyond ``as_of``.  A
    term-risk number on that stub would not answer the manager's question, so
    the apples-to-apples comparison uses a stated normalisation: each tenant in
    the plan stays for twelve equal months at its contract rent.  It is a
    normalisation, not a forecast of renewals.
    """
    start = month_start(plan.horizon_start)
    end = start
    for _ in range(months):
        end = next_month(end)
    segments: list[ExposureSegment] = []
    grouped: dict[tuple[str, str], list[ExposureSegment]] = {}
    for segment in plan.segments:
        grouped.setdefault((segment.tenant_id, segment.unit_id), []).append(segment)
    fractions = [(month_start(start), 1.0)]
    month = next_month(start)
    while month < end:
        fractions.append((month, 1.0))
        month = next_month(month)
    for (tenant, unit), items in grouped.items():
        reference = items[0]
        segments.append(
            ExposureSegment(
                plan=plan.name,
                tenant_id=tenant,
                sector=reference.sector,
                unit_id=unit,
                lease_id=reference.lease_id,
                replaces_lease_id=reference.replaces_lease_id,
                proposed=reference.proposed,
                start=start,
                end=end,
                monthly_contract=reference.monthly_contract,
                fractions=list(fractions),
            )
        )
    return PlanExposure(
        name=plan.name,
        segments=segments,
        vacancies=[],
        horizon_start=start,
        horizon_end=end,
        replacement=plan.replacement,
        shocks=plan.shocks,
        notes=[
            "normalised annual run-rate: current roster and contract rents for 12 equal months",
            "no lease expiry or vacancy is assumed in this normalisation; it is not a renewal forecast",
        ],
    )


def scale_plan_rent(plan: PlanExposure, scale: float) -> PlanExposure:
    """Scale every contract rent by one factor, keeping the tenant roster and dependence."""
    segments = [
        ExposureSegment(
            plan=segment.plan,
            tenant_id=segment.tenant_id,
            sector=segment.sector,
            unit_id=segment.unit_id,
            lease_id=segment.lease_id,
            replaces_lease_id=segment.replaces_lease_id,
            proposed=segment.proposed,
            start=segment.start,
            end=segment.end,
            monthly_contract=segment.monthly_contract * scale,
            fractions=list(segment.fractions),
        )
        for segment in plan.segments
    ]
    return PlanExposure(
        plan.name,
        segments,
        [],
        plan.horizon_start,
        plan.horizon_end,
        plan.replacement,
        plan.shocks,
        [
            f"all contract rents scaled by {scale:.4f} to hold the reference revenue assumption",
        ],
    )


def resolve_horizon(portfolio: PortfolioInput) -> tuple[datetime, datetime, list[str]]:
    config = portfolio.raw.get("analysis") or {}
    notes: list[str] = []
    if not isinstance(config, dict):
        raise RiskInputError("analysis must be an object when present")
    horizon_start = portfolio.as_of
    if config.get("horizon_start") is not None:
        horizon_start = parse_time(config["horizon_start"], "analysis.horizon_start")
    if horizon_start < portfolio.as_of:
        raise RiskInputError(
            "analysis.horizon_start precedes as_of; facts for earlier months are not known at as_of"
        )
    horizon_end = min(lease.end for lease in portfolio.leases)
    for lease in portfolio.leases:
        horizon_end = max(horizon_end, lease.end)
    for scenario in portfolio.scenarios:
        replacement = scenario.replacement
        if replacement is None:
            continue
        if replacement.ends_at is not None:
            horizon_end = max(horizon_end, replacement.ends_at)
        else:
            old = next(
                (lease for lease in portfolio.leases if lease.unit_id == replacement.unit_id and lease.start <= replacement.starts_at < lease.end),
                None,
            )
            if old is not None:
                horizon_end = max(horizon_end, old.end)
            else:
                raise RiskInputError(
                    f"replacement on {replacement.unit_id} has neither ends_at nor a covering lease; "
                    "the new exposure term cannot be derived"
                )
    if config.get("horizon_months") is not None and config.get("horizon_end") is not None:
        raise RiskInputError("analysis may set horizon_months or horizon_end, not both")
    if config.get("horizon_months") is not None:
        months = config["horizon_months"]
        if isinstance(months, bool) or not isinstance(months, int) or months <= 0:
            raise RiskInputError("analysis.horizon_months must be a positive integer")
        end = month_start(horizon_start)
        for _ in range(months):
            end = next_month(end)
        horizon_end = end
        notes.append("horizon end derived from analysis.horizon_months")
    if config.get("horizon_end") is not None:
        horizon_end = parse_time(config["horizon_end"], "analysis.horizon_end")
        notes.append("horizon end from analysis.horizon_end")
    if horizon_end <= horizon_start:
        raise RiskInputError("analysis horizon is empty; no committed exposure after as_of")
    notes.append(
        "horizon defaults to the full committed term anchored at as_of; partial first and last months are prorated"
    )
    return horizon_start, horizon_end, notes


def _candidate_terms(portfolio: PortfolioInput, tenant_id: str) -> tuple[str, float | None] | None:
    candidate = portfolio.candidates.get(tenant_id)
    if candidate is None:
        return None
    return candidate.sector, candidate.monthly_contract


def build_plan(portfolio: PortfolioInput, spec: ScenarioSpec, horizon_start: datetime, horizon_end: datetime) -> PlanExposure:
    leases = list(portfolio.leases)
    notes: list[str] = []
    proposed: list[ExposureSegment] = []
    old: Any = None
    if spec.replacement is not None:
        replacement = spec.replacement
        if replacement.starts_at < portfolio.as_of:
            notes.append(
                f"replacement starts_at {replacement.starts_at.isoformat()} precedes as_of; "
                "exposure is counted only from as_of, the information cutoff"
            )
        unit_leases = [lease for lease in leases if lease.unit_id == replacement.unit_id]
        old = next(
            (lease for lease in unit_leases if lease.start <= replacement.starts_at < lease.end),
            None,
        )
        if old is not None:
            replacement_end = replacement.ends_at or old.end
            old_end = min(old.end, replacement.starts_at)
            leases = [lease for lease in leases if lease.lease_id != old.lease_id] + [
                Lease(
                    old.lease_id,
                    old.tenant_id,
                    old.unit_id,
                    old.sector,
                    old.start,
                    old_end,
                    old.monthly_contract,
                )
            ]
            notes.append(
                f"old lease {old.lease_id} exposure truncated at replacement starts_at; "
                "old and new exposure never overlap"
            )
        else:
            if replacement.ends_at is None:
                raise RiskInputError(
                    f"replacement on {replacement.unit_id} has no covering lease; replacement.ends_at is required"
                )
            replacement_end = replacement.ends_at
            notes.append(f"replacement on vacant unit {replacement.unit_id}; no old tenant exposure is created")

        terms = _candidate_terms(portfolio, replacement.new_tenant)
        if terms is None:
            existing = next(
                (lease for lease in portfolio.leases if lease.tenant_id == replacement.new_tenant),
                None,
            )
            if existing is None:
                raise RiskInputError(f"replacement tenant {replacement.new_tenant} has no sector or rent basis")
            sector = existing.sector
            contract = existing.monthly_contract
            notes.append(
                f"proposed tenant {replacement.new_tenant} reuses existing lease terms; no new rent is invented"
            )
        else:
            sector, contract = terms
            if contract is None:
                raise RiskInputError(f"candidate tenant {replacement.new_tenant} has no monthly_contract")
        if replacement_end <= replacement.starts_at:
            raise RiskInputError(f"replacement term for {replacement.unit_id} ends before it starts")
        proposed.append(
            ExposureSegment(
                plan=spec.name,
                tenant_id=replacement.new_tenant,
                sector=sector,
                unit_id=replacement.unit_id,
                lease_id=None,
                replaces_lease_id=old.lease_id if old is not None else None,
                proposed=True,
                start=replacement.starts_at,
                end=replacement_end,
                monthly_contract=contract,
                fractions=[],
            )
        )

    segments: list[ExposureSegment] = []
    for lease in leases:
        start = max(lease.start, horizon_start)
        end = min(lease.end, horizon_end)
        if end <= start:
            continue
        segments.append(
            ExposureSegment(
                plan=spec.name,
                tenant_id=lease.tenant_id,
                sector=lease.sector,
                unit_id=lease.unit_id,
                lease_id=lease.lease_id,
                replaces_lease_id=None,
                proposed=False,
                start=start,
                end=end,
                monthly_contract=lease.monthly_contract,
                fractions=split(start, end),
            )
        )
    for segment in proposed:
        start = max(segment.start, horizon_start)
        end = min(segment.end, horizon_end)
        if end <= start:
            continue
        segment.start = start
        segment.end = end
        segment.fractions = split(start, end)
        segments.append(segment)

    # Every unit must be at most one active segment at any time.
    by_unit: dict[str, list[ExposureSegment]] = {}
    for segment in segments:
        by_unit.setdefault(segment.unit_id, []).append(segment)
    for unit_id, items in by_unit.items():
        ordered = sorted(items, key=lambda item: item.start)
        for left, right in zip(ordered, ordered[1:]):
            if right.start < left.end:
                raise RiskInputError(
                    f"scenario {spec.name}: exposure segments on {unit_id} overlap; "
                    "old and new tenant cannot both own the same interval"
                )

    # Vacancies are the complement of tenant exposure inside the horizon.
    vacancies: list[VacancySegment] = []
    for unit_id, items in by_unit.items():
        ordered = sorted(items, key=lambda item: item.start)
        cursor = horizon_start
        reference: float | None = None
        for item in ordered:
            if item.start > cursor:
                vacancies.append(
                    VacancySegment(spec.name, unit_id, cursor, item.start, reference)
                )
            cursor = max(cursor, item.end)
            reference = item.monthly_contract
        if cursor < horizon_end:
            vacancies.append(VacancySegment(spec.name, unit_id, cursor, horizon_end, reference))

    # Units with no segment at all are wholly vacant.
    seen_units = set(by_unit)
    for lease in portfolio.leases:
        if lease.unit_id in seen_units:
            continue
        seen_units.add(lease.unit_id)
        vacancies.append(VacancySegment(spec.name, lease.unit_id, horizon_start, horizon_end, None))

    # Validate separation explicitly rather than relying on construction order.
    for vacancy in vacancies:
        for segment in segments:
            if vacancy.unit_id == segment.unit_id and min(vacancy.end, segment.end) > max(vacancy.start, segment.start):
                raise RiskInputError(
                    f"scenario {spec.name}: vacancy {vacancy.start.isoformat()}–{vacancy.end.isoformat()} "
                    f"on {vacancy.unit_id} overlaps tenant exposure"
                )

    return PlanExposure(
        name=spec.name,
        segments=segments,
        vacancies=sorted(vacancies, key=lambda item: (item.unit_id, item.start)),
        horizon_start=horizon_start,
        horizon_end=horizon_end,
        replacement=spec.replacement,
        shocks=spec.shocks,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Dependence helpers


def _tenant_correlation(
    estimation: EstimationResult,
    sectors: list[str],
    a: dict[str, Any],
    b: dict[str, Any],
) -> float:
    """Latent stress-propensity correlation between two tenant parameters."""
    base = math.sqrt(max(a["common"] * b["common"], 0.0))
    if a["sector"] == b["sector"]:
        return base
    left, right = sectors[a["sector"]], sectors[b["sector"]]
    for item in estimation.cross_sector:
        if set(item["sectors"]) == {left, right}:
            return base * item["estimate"]
    return base * PRIOR_CROSS_CORRELATION


def _shortfall_covariance(a: dict[str, Any], b: dict[str, Any], latent_correlation: float) -> float:
    """Exact covariance of the two-state shortfall indicators implied by the model."""
    if a["probability"] <= 0.0 or b["probability"] <= 0.0:
        return 0.0
    if a["threshold"] is None or b["threshold"] is None:
        return 0.0
    joint = bivariate_stress_probability(a["threshold"], b["threshold"], latent_correlation)
    independent = a["probability"] * b["probability"]
    return a["depth"] * b["depth"] * (joint - independent)


def _loss_correlation(left: list[float], right: list[float]) -> float:
    """Correlation of two loss vectors; zero when either side never moves."""
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    try:
        return statistics.correlation(left, right)
    except statistics.StatisticsError:
        return 0.0


def sector_correlation(estimation: EstimationResult, sectors: list[str]) -> list[list[float]]:
    values: dict[tuple[str, str], float] = {}
    for item in estimation.cross_sector:
        left, right = item["sectors"]
        values[(left, right)] = item["estimate"]
        values[(right, left)] = item["estimate"]
    matrix: list[list[float]] = []
    for left in sectors:
        row = []
        for right in sectors:
            if left == right:
                row.append(1.0)
            else:
                row.append(values.get((left, right), PRIOR_CROSS_CORRELATION))
        matrix.append(row)
    return psd_correlation(matrix)


# ---------------------------------------------------------------------------
# Historical-term simulation


@dataclass(slots=True)
class SimulationResult:
    plan: str
    paths: int
    seed: int
    total_losses: list[float]
    tenant_losses: dict[str, list[float]]
    mean_loss: float
    sd_loss: float
    sd_tenant_sum: float
    var_95: float
    es_95: float
    component_es: dict[str, float]
    analytic_mean_loss: float
    analytic_portfolio_sd: float
    analytic_tenant_sd_sum: float
    rent_by_month: dict[str, dict[str, float]]
    total_rent: float
    months: list[str]
    tenant_orders: list[str] = field(default_factory=list)

    def metrics(self, plan: PlanExposure) -> dict[str, Any]:
        total = self.total_rent
        realised = {
            left: {
                right: _round(_loss_correlation(self.tenant_losses[left], self.tenant_losses[right]), 4)
                for right in self.tenant_orders
            }
            for left in self.tenant_orders
        }
        return {
            "plan": self.plan,
            "exposure_months": self.months,
            "total_exposed_rent": _round(total, 2),
            "expected_uncollected": _round(self.mean_loss, 2),
            "expected_collection_rate": _round(1 - self.mean_loss / total, 4) if total else None,
            "uncollected_sd": _round(self.sd_loss, 2),
            "uncollected_var_95": _round(self.var_95, 2),
            "uncollected_es_95": _round(self.es_95, 2),
            "es_95_share_of_exposed_rent": _round(self.es_95 / total, 4) if total else None,
            "sum_of_tenant_standard_deviations": _round(self.sd_tenant_sum, 2),
            "diversification_ratio": _round(self.sd_loss / self.sd_tenant_sum, 4) if self.sd_tenant_sum else None,
            "portfolio_sd_not_sum_of_tenant_sd": bool(self.sd_loss < self.sd_tenant_sum - 1e-9),
            "analytic_mean_uncollected": _round(self.analytic_mean_loss, 2),
            "analytic_portfolio_sd": _round(self.analytic_portfolio_sd, 2),
            "analytic_tenant_sd_sum": _round(self.analytic_tenant_sd_sum, 2),
            "analytic_diversification_ratio": _round(self.analytic_portfolio_sd / self.analytic_tenant_sd_sum, 4)
            if self.analytic_tenant_sd_sum
            else None,
            "simulation": {
                "paths": self.paths,
                "seed": self.seed,
                "quantile": 0.95,
                "monthly_dependence": "iid across months; dependence only across tenants within a month",
                "rate_model": (
                    "latent two-state monthly model: each tenant-month is stressed with probability q and "
                    "then collects depth less; latent stress propensity is a sector factor plus an idiosyncratic "
                    "term, so q and depth reproduce the observed mean and variance and every rate stays in [0,1]"
                ),
                "realised_tenant_loss_correlation": realised,
                "portfolio_sd_not_sum_of_tenant_sd": "analytic covariance uses exact bivariate stress probabilities; "
                "the simulated sd is checked against it",
            },
        }


def simulate_plan(
    plan: PlanExposure,
    estimation: EstimationResult,
    paths: int = DEFAULT_PATHS,
    seed: int | None = None,
) -> SimulationResult:
    tenants = plan.tenants
    months = plan.months()
    rent_by_month: dict[str, dict[str, float]] = {
        tenant: {_month_label(month): value for month, value in plan.rent_by_month(tenant).items()}
        for tenant in tenants
    }
    total_rent = plan.total_rent()
    if not tenants or not months or total_rent <= 0:
        return SimulationResult(
            plan.name,
            paths,
            seed or SIMULATION_SEED,
            [0.0],
            {tenant: [0.0] for tenant in tenants},
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            {tenant: 0.0 for tenant in tenants},
            0.0,
            0.0,
            0.0,
            rent_by_month,
            total_rent,
            [_month_label(month) for month in months],
            tenants,
        )

    sectors = sorted({estimation.tenants[tenant].sector for tenant in tenants})
    sector_index = {sector: index for index, sector in enumerate(sectors)}
    correlation = sector_correlation(estimation, sectors)
    factor_cholesky = cholesky(correlation)
    rng = random.Random(SIMULATION_SEED if seed is None else seed)

    tenant_params = {}
    for tenant in tenants:
        estimate = estimation.tenants[tenant]
        stress = estimate.stress()
        probability = stress["stress_probability"]
        threshold = normal_quantile(1.0 - probability) if 0.0 < probability < 1.0 else None
        tenant_params[tenant] = {
            "mean": estimate.mean_rate,
            "variance": stress["variance_used"],
            "probability": probability,
            "depth": stress["stress_depth"],
            "threshold": threshold,
            "common": estimate.common_share,
            "sector": sector_index[estimate.sector],
        }

    total_losses: list[float] = []
    tenant_losses: dict[str, list[float]] = {tenant: [] for tenant in tenants}
    rent_matrix = {
        tenant: [rent_by_month[tenant].get(_month_label(month), 0.0) for month in months]
        for tenant in tenants
    }
    for _ in range(paths):
        total = 0.0
        per_tenant = {tenant: 0.0 for tenant in tenants}
        for month_index in range(len(months)):
            # Correlated sector stress propensities for this month only.  A
            # tenant is stressed when its latent propensity crosses the
            # threshold that reproduces the observed stress probability.
            base = [rng.gauss(0.0, 1.0) for _ in sectors]
            factors = [sum(factor_cholesky[i][j] * base[j] for j in range(i + 1)) for i in range(len(sectors))]
            for tenant in tenants:
                rent = rent_matrix[tenant][month_index]
                if rent <= 0:
                    continue
                params = tenant_params[tenant]
                if params["probability"] <= 0.0 or params["threshold"] is None:
                    continue
                loading = math.sqrt(max(params["common"], 0.0))
                idiosyncratic = math.sqrt(max(1.0 - params["common"], 0.0))
                latent = loading * factors[params["sector"]] + idiosyncratic * rng.gauss(0.0, 1.0)
                if latent >= params["threshold"]:
                    loss = rent * params["depth"]
                    total += loss
                    per_tenant[tenant] += loss
        total_losses.append(total)
        for tenant in tenants:
            tenant_losses[tenant].append(per_tenant[tenant])

    ordered = sorted(total_losses)
    index = min(int(0.95 * paths), paths - 1)
    var_95 = ordered[index]
    tail = [value for value in total_losses if value >= var_95]
    es_95 = mean(tail) if tail else var_95
    component_es = {}
    for tenant in tenants:
        available = [tenant_losses[tenant][i] for i, value in enumerate(total_losses) if value >= var_95]
        component_es[tenant] = mean(available) if available else 0.0

    # Analytic check.  Per month the covariance of tenant shortfalls is
    # depth_a*depth_b*(P(stress both) - q_a*q_b); the bivariate probability uses
    # the same sector correlation the simulation draws from.
    analytic_mean = 0.0
    analytic_tenant_sds: list[float] = []
    for tenant in tenants:
        params = tenant_params[tenant]
        analytic_mean += (1.0 - params["mean"]) * sum(rent_matrix[tenant])
        analytic_tenant_sds.append(
            math.sqrt(params["variance"]) * math.sqrt(sum(value * value for value in rent_matrix[tenant]))
        )

    analytic_variance = 0.0
    for month_index in range(len(months)):
        for left in tenants:
            a = tenant_params[left]
            for right in tenants:
                b = tenant_params[right]
                if left == right:
                    covariance = a["variance"]
                else:
                    correlation_value = _tenant_correlation(estimation, sectors, a, b)
                    covariance = _shortfall_covariance(a, b, correlation_value)
                analytic_variance += (
                    rent_matrix[left][month_index] * rent_matrix[right][month_index] * covariance
                )
    analytic_sd = math.sqrt(max(analytic_variance, 0.0))

    return SimulationResult(
        plan.name,
        paths,
        SIMULATION_SEED if seed is None else seed,
        total_losses,
        tenant_losses,
        mean(total_losses),
        stdev(total_losses),
        sum(stdev(tenant_losses[tenant]) for tenant in tenants),
        var_95,
        es_95,
        component_es,
        analytic_mean,
        analytic_sd,
        sum(analytic_tenant_sds),
        rent_by_month,
        total_rent,
        [_month_label(month) for month in months],
        tenants,
    )


# ---------------------------------------------------------------------------
# Assumption scenarios and drills


def _rate_under_shock(mean_rate: float, shock: float) -> float:
    """Management shock is applied multiplicatively to the expected collected rate."""
    return _clip(mean_rate * (1.0 + shock), 0.0, 1.0)


def _shock_set_id(shocks: dict[str, float]) -> str:
    if not shocks:
        return "no_shock"
    return ";".join(f"{sector}:{value:+.3f}" for sector, value in sorted(shocks.items()))


def management_shock_metrics(
    plans: list[PlanExposure], estimation: EstimationResult, shock_scenarios: list[ScenarioSpec]
) -> dict[str, Any]:
    """Apply each declared management shock set to every plan for a fair comparison."""
    groups: dict[str, dict[str, Any]] = {}
    for spec in shock_scenarios:
        shock_id = _shock_set_id(spec.shocks)
        groups.setdefault(shock_id, {"shocks": dict(spec.shocks), "source_scenarios": [], "plans": {}})
        groups[shock_id]["source_scenarios"].append(spec.name)
    for plan in plans:
        for shock_id, group in groups.items():
            if plan.name in group["plans"]:
                continue
            by_sector: dict[str, float] = {}
            by_tenant: dict[str, dict[str, Any]] = {}
            total = 0.0
            for segment in plan.segments:
                estimate = estimation.tenants[segment.tenant_id]
                shock = group["shocks"].get(segment.sector, 0.0)
                rate = _rate_under_shock(estimate.mean_rate, shock)
                loss = segment.exposed_rent * max(0.0, 1.0 - rate)
                total += loss
                by_sector[segment.sector] = by_sector.get(segment.sector, 0.0) + loss
                row = by_tenant.setdefault(
                    segment.tenant_id,
                    {
                        "tenant_id": segment.tenant_id,
                        "sector": segment.sector,
                        "shock": shock,
                        "stressed_collection_rate": _round(rate, 4),
                        "uncollected": 0.0,
                    },
                )
                row["uncollected"] = _round(row["uncollected"] + loss, 2)
            group["plans"][plan.name] = {
                "shocks": dict(group["shocks"]),
                "expected_uncollected_under_stress": _round(total, 2),
                "uncollected_by_sector": {key: _round(value, 2) for key, value in sorted(by_sector.items())},
                "tenants": [by_tenant[tenant] for tenant in sorted(by_tenant)],
            }
    return {
        "label": "management assumption",
        "not_a_probability": True,
        "not_from_history": True,
        "treatment": "scenario.shocks scale the collected rate by (1+shock); deterministic what-if, no probability",
        "comparison": "each declared shock set is applied to every plan at the same contract-rent assumption",
        "sets": [
            {
                "shock_set": shock_id,
                "shocks": group["shocks"],
                "source_scenarios": group["source_scenarios"],
                "plans": group["plans"],
            }
            for shock_id, group in sorted(groups.items())
        ],
    }


def _seasonal_factors(estimation: EstimationResult) -> dict[str, dict[int, float]]:
    """Calendar-month collection shortfalls per sector, shrunk toward zero."""
    observations = estimation.observations
    raw_by_tenant: dict[str, float] = {}
    by_tenant: dict[str, list[Any]] = {}
    for item in observations:
        by_tenant.setdefault(item.tenant_id, []).append(item)
    for tenant, rows in by_tenant.items():
        raw_by_tenant[tenant] = mean(item.rate for item in rows)
    grouped: dict[str, dict[int, list[float]]] = {}
    for item in observations:
        grouped.setdefault(item.sector, {}).setdefault(item.month.month, []).append(
            item.rate - raw_by_tenant[item.tenant_id]
        )
    result: dict[str, dict[int, float]] = {}
    for sector, months in grouped.items():
        result[sector] = {}
        for month, values in months.items():
            count = len(values)
            shrink = count / (count + 2.0)
            result[sector][month] = shrink * mean(values)
    return result


def _seasonal_source(estimation: EstimationResult) -> dict[str, Any]:
    factors = _seasonal_factors(estimation)
    worst: dict[str, Any] = {}
    for sector, months in factors.items():
        month = min(months, key=lambda value: months[value]) if months else None
        worst[sector] = {
            "month_of_year": month,
            "shortfall": _round(months[month], 4) if month is not None else None,
            "months_observed": sorted(months),
        }
    return {"factors": {sector: {str(month): _round(value, 4) for month, value in months.items()} for sector, months in factors.items()},
            "worst": worst}


def _annualised_drill(
    plan: PlanExposure, estimation: EstimationResult, mode: str, worst: dict[str, Any]
) -> dict[str, Any]:
    """Structural one-year rehearsal; not a forecast of the committed term."""
    tenants = plan.tenants
    monthly_rent = {tenant: sum(segment.monthly_contract for segment in plan.segments if segment.tenant_id == tenant) for tenant in tenants}
    sector_rent = {
        sector: sum(
            monthly_rent[tenant]
            for tenant in tenants
            if estimation.tenants[tenant].sector == sector
        )
        for sector in sorted({estimation.tenants[tenant].sector for tenant in tenants})
    }
    annual_rent = 12.0 * sum(monthly_rent.values())
    expected = sum(
        12.0 * monthly_rent[tenant] * max(0.0, 1.0 - estimation.tenants[tenant].mean_rate)
        for tenant in tenants
    )
    extra_by_tenant: dict[str, float] = {tenant: 0.0 for tenant in tenants}
    placed: dict[str, int] = {}
    for index, sector in enumerate(sorted(sector_rent)):
        info = worst.get(sector, {})
        shortfall = max(0.0, -(info.get("shortfall") or 0.0))
        if mode == "aligned":
            # The sector-rent-weighted month slot keeps aligned concentrated and
            # diversified plans in the same calendar month.
            month_slot = 0
        elif mode == "staggered":
            month_slot = (info.get("month_of_year") or (index + 1)) - 1
            while month_slot in placed.values():
                month_slot = (month_slot + 1) % 12
        else:
            raise RiskInputError(f"unknown seasonal drill mode {mode!r}")
        placed[sector] = month_slot
        for tenant in tenants:
            if estimation.tenants[tenant].sector != sector:
                continue
            extra_by_tenant[tenant] += monthly_rent[tenant] * shortfall
    peak_by_month: list[float] = [0.0] * 12
    for sector, slot in placed.items():
        info = worst.get(sector, {})
        shortfall = max(0.0, -(info.get("shortfall") or 0.0))
        peak_by_month[slot] += sector_rent.get(sector, 0.0) * shortfall
    expected_monthly = expected / 12.0
    return {
        "mode": mode,
        "basis": "annualised structural rehearsal on 12 equal contract-rent months; not a forecast of the committed lease term",
        "annual_exposed_rent": _round(annual_rent, 2),
        "expected_uncollected_before_season": _round(expected, 2),
        "season_extra_uncollected": _round(sum(extra_by_tenant.values()), 2),
        "annual_uncollected": _round(expected + sum(extra_by_tenant.values()), 2),
        "worst_month_uncollected": _round(expected_monthly + max(peak_by_month), 2),
        "sector_shortfall_placements": {sector: placed[sector] + 1 for sector in sorted(placed)},
        "by_tenant_season_extra": {tenant: _round(value, 2) for tenant, value in sorted(extra_by_tenant.items())},
    }


def structural_drills(
    plans: list[PlanExposure], estimation: EstimationResult
) -> dict[str, Any]:
    seasonal = _seasonal_source(estimation)
    worst = seasonal["worst"]
    worst_common = min(
        (info["shortfall"] for info in worst.values() if info.get("shortfall") is not None),
        default=None,
    )
    if worst_common is None or worst_common >= 0:
        pooled_sd = math.sqrt(mean(list(estimation.sectors.values())[0].get("pooled_variance") or 0.0)) if estimation.sectors else 0.05
        worst_common = -max(pooled_sd, 0.05)

    common: dict[str, Any] = {}
    aligned: dict[str, Any] = {}
    staggered: dict[str, Any] = {}
    for plan in plans:
        tenants = plan.tenants
        monthly_rent = {
            tenant: sum(segment.monthly_contract for segment in plan.segments if segment.tenant_id == tenant)
            for tenant in tenants
        }
        annual_rent = 12.0 * sum(monthly_rent.values())
        expected = sum(
            12.0 * monthly_rent[tenant] * max(0.0, 1.0 - estimation.tenants[tenant].mean_rate)
            for tenant in tenants
        )
        # A persistent common downside: every sector takes the same observed
        # worst sector-month shortfall in every month.  Deterministic.
        stressed = sum(
            12.0 * monthly_rent[tenant]
            * max(0.0, 1.0 - _clip(estimation.tenants[tenant].mean_rate + worst_common, 0.0, 1.0))
            for tenant in tenants
        )
        common[plan.name] = {
            "basis": "annualised persistent common downside; observed worst sector-month shortfall repeated in all sectors",
            "common_shortfall": _round(worst_common, 4),
            "annual_exposed_rent": _round(annual_rent, 2),
            "expected_uncollected_before_shock": _round(expected, 2),
            "annual_uncollected": _round(stressed, 2),
            "incremental_uncollected": _round(stressed - expected, 2),
            "not_a_probability": True,
        }
        aligned[plan.name] = _annualised_drill(plan, estimation, "aligned", worst)
        staggered[plan.name] = _annualised_drill(plan, estimation, "staggered", worst)

    return {
        "common_downside": {
            "label": "common downside drill",
            "shared_shortfall_used": _round(worst_common, 4),
            "source": "most negative shrunk calendar-month sector factor observed in history",
            "not_a_probability": True,
            "plans": common,
        },
        "offseason": {
            "label": "staggered off-season drill",
            "not_a_probability": True,
            "seasonal_shortfalls": seasonal,
            "aligned": {
                "definition": "every sector's worst observed shortfall placed in the same calendar month",
                "plans": aligned,
            },
            "staggered": {
                "definition": "each sector's worst observed shortfall placed in a distinct month, using its observed worst month as the anchor",
                "plans": staggered,
            },
        },
    }


def new_tenant_drill(
    plans: list[PlanExposure], estimation: EstimationResult, portfolio: PortfolioInput
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for plan in plans:
        rows = []
        for tenant in plan.tenants:
            estimate = estimation.tenants[tenant]
            if tenant not in portfolio.candidates or estimate.history_months > 0:
                continue
            exposed = sum(segment.exposed_rent for segment in plan.segments if segment.tenant_id == tenant)
            lower, upper = estimate.mean_interval
            def loss(rate: float) -> float:
                return exposed * max(0.0, 1.0 - rate)
            rows.append(
                {
                    "tenant_id": tenant,
                    "sector": estimate.sector,
                    "exposed_rent": _round(exposed, 2),
                    "history_months": 0,
                    "proxy_mean": _round(estimate.mean_rate, 4),
                    "interval": [_round(lower, 4), _round(upper, 4)],
                    "uncollected_if_lower_bound": _round(loss(lower), 2),
                    "uncollected_at_proxy": _round(loss(estimate.mean_rate), 2),
                    "uncollected_if_upper_bound": _round(loss(upper), 2),
                    "interval_is_probability": False,
                }
            )
        if rows:
            result[plan.name] = {
                "basis": "candidate has no tenant-specific cash-flow history; sector proxy plus an interval, no hidden series and no zero fill",
                "not_a_probability": True,
                "candidates": rows,
            }
    return {
        "label": "new tenant without history drill",
        "not_a_probability": True,
        "plans": result,
        "note": "interval is an estimation range, not a confidence claim about future payment behaviour",
    }


# ---------------------------------------------------------------------------
# Contributions and attribution


def contribution_report(
    plans: list[PlanExposure],
    simulations: dict[str, SimulationResult],
    estimation: EstimationResult,
    keep_plan: str | None,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for plan in plans:
        simulation = simulations[plan.name]
        if simulation.total_rent <= 0:
            continue
        tenants = plan.tenants
        expected = {
            tenant: sum(
                segment.exposed_rent * max(0.0, 1.0 - estimation.tenants[tenant].mean_rate)
                for segment in plan.segments
                if segment.tenant_id == tenant
            )
            for tenant in tenants
        }
        total_expected = sum(expected.values())
        rows = []
        for tenant in tenants:
            component = simulation.component_es.get(tenant, 0.0)
            rows.append(
                {
                    "tenant_id": tenant,
                    "sector": estimation.tenants[tenant].sector,
                    "exposed_rent": _round(sum(segment.exposed_rent for segment in plan.segments if segment.tenant_id == tenant), 2),
                    "expected_uncollected": _round(expected[tenant], 2),
                    "expected_share": _round(expected[tenant] / total_expected, 4) if total_expected else None,
                    "es_95_contribution": _round(component, 2),
                    "es_95_share": _round(component / simulation.es_95, 4) if simulation.es_95 else None,
                    "history_months": estimation.tenants[tenant].history_months,
                    "estimate_source": estimation.tenants[tenant].mean_source,
                }
            )
        reports[plan.name] = {
            "expected_uncollected_total": _round(total_expected, 2),
            "es_95_total": _round(simulation.es_95, 2),
            "expected_uncollected_by_tenant": {
                tenant: _round(value, 2) for tenant, value in sorted(expected.items())
            },
            "contributions": rows,
            "sum_check": {
                "expected_contributions_equal_total": abs(sum(value for value in expected.values()) - total_expected) < 1e-9,
                "es_contributions_equal_es": abs(sum(simulation.component_es.values()) - simulation.es_95) < 1e-6,
            },
        }

    delta: dict[str, Any] = {}
    if keep_plan and keep_plan in simulations:
        keep_sim = simulations[keep_plan]
        keep_plan_obj = next(plan for plan in plans if plan.name == keep_plan)
        keep_expected = {
            tenant: sum(
                segment.exposed_rent * max(0.0, 1.0 - estimation.tenants[tenant].mean_rate)
                for segment in keep_plan_obj.segments
                if segment.tenant_id == tenant
            )
            for tenant in keep_plan_obj.tenants
        }
        for plan in plans:
            if plan.name == keep_plan:
                continue
            simulation = simulations[plan.name]
            expected = {
                tenant: sum(
                    segment.exposed_rent * max(0.0, 1.0 - estimation.tenants[tenant].mean_rate)
                    for segment in plan.segments
                    if segment.tenant_id == tenant
                )
                for tenant in plan.tenants
            }
            drivers = []
            for tenant in sorted(set(plan.tenants) | set(keep_plan_obj.tenants)):
                before = keep_expected.get(tenant, 0.0)
                after = expected.get(tenant, 0.0)
                delta_expected = after - before
                es_before = keep_sim.component_es.get(tenant, 0.0)
                es_after = simulation.component_es.get(tenant, 0.0)
                delta_es = es_after - es_before
                if abs(delta_expected) < 0.01 and abs(delta_es) < 0.01:
                    continue
                segments_after = [segment for segment in plan.segments if segment.tenant_id == tenant]
                segments_before = [segment for segment in keep_plan_obj.segments if segment.tenant_id == tenant]
                lease_refs = []
                seen_refs: set[tuple[Any, Any, Any]] = set()
                for segment in segments_after + segments_before:
                    key = (segment.unit_id, segment.lease_id, segment.replaces_lease_id)
                    if key in seen_refs:
                        continue
                    seen_refs.add(key)
                    lease_refs.append(
                        {
                            "unit_id": segment.unit_id,
                            "lease_id": segment.lease_id,
                            "proposed_lease": segment.proposed,
                            "replaces_lease_id": segment.replaces_lease_id,
                        }
                    )
                drivers.append(
                    {
                        "tenant_id": tenant,
                        "sector": estimation.tenants[tenant].sector,
                        "expected_uncollected_before": _round(before, 2),
                        "expected_uncollected_after": _round(after, 2),
                        "expected_delta": _round(delta_expected, 2),
                        "es_95_before": _round(es_before, 2),
                        "es_95_after": _round(es_after, 2),
                        "es_95_delta": _round(delta_es, 2),
                        "lease_references": lease_refs,
                    }
                )
            drivers.sort(key=lambda item: abs(item["es_95_delta"]), reverse=True)
            delta[plan.name] = {
                "reference_plan": keep_plan,
                "expected_uncollected_delta_total": _round(simulation.mean_loss - keep_sim.mean_loss, 2),
                "es_95_delta_total": _round(simulation.es_95 - keep_sim.es_95, 2),
                "drivers": drivers,
                "attribution_note": "drivers compare tenant-level expected and tail contributions to the reference plan; leases list the contractual references",
            }
    return {
        "method": (
            "expected uncollected is contract-rent-weighted shortfall; tail contribution is E[tenant loss | "
            "portfolio loss >= VaR95], so contributions add to the portfolio ES"
        ),
        "plans": reports,
        "scenario_deltas": delta,
    }
