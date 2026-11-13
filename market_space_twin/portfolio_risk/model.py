"""Typed parsing and exposure construction for the portfolio risk analysis."""

from __future__ import annotations

import calendar
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

UTC = timezone.utc
ISO = "%Y-%m-%dT%H:%M:%S%z"


class RiskInputError(ValueError):
    """The input cannot be interpreted without inventing facts."""


def parse_time(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise RiskInputError(f"{field} must be a timezone-aware ISO 8601 timestamp")
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise RiskInputError(f"{field} is not a valid ISO 8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RiskInputError(f"{field} must carry a UTC offset: {value!r}")
    return parsed.astimezone(UTC)


def parse_month(value: Any, field: str = "month") -> datetime:
    if not isinstance(value, str) or len(value) < 7:
        raise RiskInputError(f"{field} must be YYYY-MM, got {value!r}")
    try:
        year, month = int(value[0:4]), int(value[5:7])
        if month < 1 or month > 12:
            raise ValueError
        return datetime(year, month, 1, tzinfo=UTC)
    except ValueError as exc:
        raise RiskInputError(f"{field} must be YYYY-MM, got {value!r}") from exc


def month_start(value: datetime) -> datetime:
    return datetime(value.year, value.month, 1, tzinfo=UTC)


def next_month(value: datetime) -> datetime:
    year = value.year + (1 if value.month == 12 else 0)
    month = 1 if value.month == 12 else value.month + 1
    return datetime(year, month, 1, tzinfo=UTC)


def month_bounds(value: datetime) -> tuple[datetime, datetime]:
    start = month_start(value)
    return start, next_month(start)


def month_days(value: datetime) -> int:
    start = month_start(value)
    return calendar.monthrange(start.year, start.month)[1]


def iso(value: datetime) -> str:
    return value.isoformat()


def optional_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RiskInputError(f"{field} must be a non-empty string")
    return value.strip()


@dataclass(slots=True)
class Lease:
    lease_id: str
    tenant_id: str
    unit_id: str
    sector: str
    start: datetime
    end: datetime
    monthly_contract: float


@dataclass(slots=True)
class CashFlow:
    tenant_id: str
    month: datetime
    due: float | None
    received: float | None
    concession: float | None
    available_at: datetime | None
    within_lease: bool


@dataclass(slots=True)
class Candidate:
    tenant_id: str
    sector: str
    monthly_contract: float | None
    history: list[CashFlow]


@dataclass(slots=True)
class Replacement:
    unit_id: str
    new_tenant: str
    starts_at: datetime
    ends_at: datetime | None


@dataclass(slots=True)
class ScenarioSpec:
    name: str
    replacement: Replacement | None
    shocks: dict[str, float]


@dataclass(slots=True)
class PortfolioInput:
    schema_version: int
    as_of: datetime
    source_versions: dict[str, Any]
    leases: list[Lease]
    cashflows: list[CashFlow]
    candidates: dict[str, Candidate]
    scenarios: list[ScenarioSpec]
    baseline_case_id: str | None
    raw: dict[str, Any]


def _parse_leases(value: Any) -> list[Lease]:
    if not isinstance(value, list) or not value:
        raise RiskInputError("leases must be a non-empty list")
    leases: list[Lease] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise RiskInputError(f"leases[{index}] must be an object")
        lease_id = require_string(item.get("lease_id"), f"leases[{index}].lease_id")
        tenant_id = require_string(item.get("tenant_id"), f"leases[{index}].tenant_id")
        unit_id = require_string(item.get("unit_id"), f"leases[{index}].unit_id")
        sector = require_string(item.get("sector"), f"leases[{index}].sector")
        if lease_id in seen_ids:
            raise RiskInputError(f"duplicate lease_id: {lease_id}")
        seen_ids.add(lease_id)
        start = parse_time(item.get("start"), f"leases[{index}].start")
        end = parse_time(item.get("end"), f"leases[{index}].end")
        if end <= start:
            raise RiskInputError(f"leases[{index}] end must be after start")
        contract = optional_number(item.get("monthly_contract"))
        if contract is None or contract < 0:
            raise RiskInputError(f"leases[{index}].monthly_contract must be a non-negative number")
        leases.append(Lease(lease_id, tenant_id, unit_id, sector, start, end, contract))
    by_unit: dict[str, list[Lease]] = {}
    for lease in leases:
        by_unit.setdefault(lease.unit_id, []).append(lease)
    for unit_id, items in by_unit.items():
        ordered = sorted(items, key=lambda item: item.start)
        for left, right in zip(ordered, ordered[1:]):
            if right.start < left.end:
                raise RiskInputError(
                    f"leases for unit {unit_id} overlap ({left.lease_id} and {right.lease_id}); "
                    "resolve occupancy before estimating exposure"
                )
    return leases


def _parse_candidates(value: Any) -> dict[str, Candidate]:
    result: dict[str, Candidate] = {}
    if value is None:
        return result
    if not isinstance(value, list):
        raise RiskInputError("candidate_tenants must be a list")
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise RiskInputError(f"candidate_tenants[{index}] must be an object")
        tenant_id = require_string(item.get("tenant_id"), f"candidate_tenants[{index}].tenant_id")
        sector = require_string(item.get("sector"), f"candidate_tenants[{index}].sector")
        contract = optional_number(item.get("monthly_contract"))
        history = item.get("historical_cashflows") or []
        if not isinstance(history, list):
            raise RiskInputError(f"candidate_tenants[{index}].historical_cashflows must be a list")
        rows = []
        seen_months: set[str] = set()
        for row_index, row in enumerate(history):
            if not isinstance(row, dict):
                raise RiskInputError(f"candidate_tenants[{index}].historical_cashflows[{row_index}] must be an object")
            month = parse_month(row.get("month"), "historical_cashflows.month")
            label = month.strftime("%Y-%m")
            if label in seen_months:
                raise RiskInputError(
                    f"candidate_tenants[{index}] has duplicate historical cashflow for {label}"
                )
            seen_months.add(label)
            rows.append(
                CashFlow(
                    tenant_id,
                    month,
                    optional_number(row.get("due")),
                    optional_number(row.get("received")),
                    optional_number(row.get("concession")),
                    parse_time(row["available_at"], "available_at") if row.get("available_at") else None,
                    True,
                )
            )
        if tenant_id in result:
            raise RiskInputError(f"duplicate candidate tenant_id: {tenant_id}")
        result[tenant_id] = Candidate(tenant_id, sector, contract, rows)
    return result


def _parse_cashflows(value: Any, leases: list[Lease], as_of: datetime) -> list[CashFlow]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RiskInputError("cashflows must be a list")
    lease_by_tenant: dict[str, list[Lease]] = {}
    for lease in leases:
        lease_by_tenant.setdefault(lease.tenant_id, []).append(lease)
    rows: list[CashFlow] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise RiskInputError(f"cashflows[{index}] must be an object")
        tenant_id = require_string(item.get("tenant_id"), f"cashflows[{index}].tenant_id")
        month = parse_month(item.get("month"), f"cashflows[{index}].month")
        key = (tenant_id, month.strftime("%Y-%m"))
        if key in seen:
            raise RiskInputError(f"duplicate cashflow for {tenant_id} in {key[1]}")
        seen.add(key)
        available_at = None
        if item.get("available_at") is not None:
            available_at = parse_time(item.get("available_at"), f"cashflows[{index}].available_at")
        tenant_leases = lease_by_tenant.get(tenant_id, [])
        within = False
        if tenant_leases:
            range_start = min(lease.start for lease in tenant_leases)
            range_end = max(lease.end for lease in tenant_leases)
            month_end = next_month(month)
            within = month < range_end and month_end > range_start
        rows.append(
            CashFlow(
                tenant_id,
                month,
                optional_number(item.get("due")),
                optional_number(item.get("received")),
                optional_number(item.get("concession")),
                available_at,
                within,
            )
        )
    return rows


def _parse_scenarios(value: Any, leases: list[Lease], candidates: dict[str, Candidate]) -> list[ScenarioSpec]:
    if value is None:
        return [ScenarioSpec("keep", None, {})]
    if not isinstance(value, list) or not value:
        raise RiskInputError("scenarios must be a non-empty list")
    unit_ids = {lease.unit_id for lease in leases}
    result: list[ScenarioSpec] = []
    names: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise RiskInputError(f"scenarios[{index}] must be an object")
        name = require_string(item.get("name"), f"scenarios[{index}].name")
        if name in names:
            raise RiskInputError(f"duplicate scenario name: {name}")
        names.add(name)
        raw_shocks = item.get("shocks") or {}
        if not isinstance(raw_shocks, dict):
            raise RiskInputError(f"scenarios[{index}].shocks must be an object")
        shocks: dict[str, float] = {}
        for sector, amount in raw_shocks.items():
            parsed = optional_number(amount)
            if parsed is None:
                raise RiskInputError(f"scenarios[{index}].shocks[{sector!r}] must be a finite number")
            shocks[str(sector)] = parsed
        raw_replacement = item.get("replacement")
        replacement = None
        if raw_replacement is not None:
            if not isinstance(raw_replacement, dict):
                raise RiskInputError(f"scenarios[{index}].replacement must be an object or null")
            unit_id = require_string(raw_replacement.get("unit_id"), f"scenarios[{index}].replacement.unit_id")
            if unit_id not in unit_ids:
                raise RiskInputError(f"replacement unit {unit_id} is not present in leases")
            new_tenant = require_string(raw_replacement.get("new_tenant"), f"scenarios[{index}].replacement.new_tenant")
            if new_tenant not in candidates:
                known = any(lease.tenant_id == new_tenant for lease in leases)
                if not known:
                    raise RiskInputError(
                        f"replacement tenant {new_tenant} is neither a candidate tenant nor an existing lease"
                    )
            starts_at = parse_time(raw_replacement.get("starts_at"), f"scenarios[{index}].replacement.starts_at")
            ends_at = None
            if raw_replacement.get("ends_at") is not None:
                ends_at = parse_time(raw_replacement.get("ends_at"), f"scenarios[{index}].replacement.ends_at")
                if ends_at <= starts_at:
                    raise RiskInputError(f"scenarios[{index}].replacement.ends_at must be after starts_at")
            replacement = Replacement(unit_id, new_tenant, starts_at, ends_at)
        result.append(ScenarioSpec(name, replacement, shocks))
    return result


def load_portfolio_input(raw: dict[str, Any]) -> PortfolioInput:
    if not isinstance(raw, dict):
        raise RiskInputError("input must be a JSON object")
    schema_version = raw.get("schema_version", 1)
    if not isinstance(schema_version, int) or schema_version != 1:
        raise RiskInputError(f"unsupported schema_version: {schema_version!r}")
    as_of = parse_time(raw.get("as_of"), "as_of")
    source_versions = raw.get("source_versions")
    if not isinstance(source_versions, dict) or not source_versions:
        raise RiskInputError("source_versions must be a non-empty object")
    leases = _parse_leases(raw.get("leases"))
    cashflows = _parse_cashflows(raw.get("cashflows"), leases, as_of)
    candidates = _parse_candidates(raw.get("candidate_tenants"))
    scenarios = _parse_scenarios(raw.get("scenarios"), leases, candidates)
    baseline_case_id = raw.get("baseline_case_id")
    if baseline_case_id is not None and not isinstance(baseline_case_id, str):
        raise RiskInputError("baseline_case_id must be a string or null")
    return PortfolioInput(
        schema_version=schema_version,
        as_of=as_of,
        source_versions=dict(source_versions),
        leases=leases,
        cashflows=cashflows,
        candidates=candidates,
        scenarios=scenarios,
        baseline_case_id=baseline_case_id,
        raw=raw,
    )


def month_fraction(start: datetime, end: datetime, month: datetime) -> float:
    """Fraction of a calendar month covered by the half-open interval [start, end)."""
    bucket_start, bucket_end = month_bounds(month)
    overlap_start = max(start, bucket_start)
    overlap_end = min(end, bucket_end)
    if overlap_end <= overlap_start:
        return 0.0
    seconds = (overlap_end - overlap_start).total_seconds()
    return seconds / (month_days(month) * 86400.0)


def split_interval(start: datetime, end: datetime, horizon_start: datetime, horizon_end: datetime) -> list[tuple[datetime, float]]:
    """Return (month_start, active_fraction) pairs inside the horizon."""
    start = max(start, horizon_start)
    end = min(end, horizon_end)
    if end <= start:
        return []
    result = []
    month = month_start(start)
    while month < end:
        fraction = month_fraction(start, end, month)
        if fraction > 0:
            result.append((month, round(fraction, 8)))
        month = next_month(month)
    return result
