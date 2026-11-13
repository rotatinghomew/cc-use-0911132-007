"""Assemble the portfolio cash-flow risk comparison.

Public entry point: :func:`analyze`.  The result deliberately separates

``exposures``      contract facts (plan, unit, lease, prorated committed rent, vacancies)
``dependence``     estimated co-movement with an identification band
``scenarios``      history-based term risk, labelled management shocks, structural drills
``contributions``  tenant-level expected/tail contributions and plan drivers
``limitations``    what the data cannot identify

Predicting footfall, ranking tenants for a leasing decision and treating
management shocks as probabilities are out of scope by construction.
"""

from __future__ import annotations

import json
from typing import Any

from market_space_twin.portfolio_risk.estimation import EstimationResult, estimate_dependence
from market_space_twin.portfolio_risk.model import (
    PortfolioInput,
    RiskInputError,
    load_portfolio_input,
)
from market_space_twin.portfolio_risk.scenarios import (
    DEFAULT_PATHS,
    SIMULATION_SEED,
    PlanExposure,
    SimulationResult,
    build_plan,
    contribution_report,
    management_shock_metrics,
    new_tenant_drill,
    normalised_plan,
    resolve_horizon,
    scale_plan_rent,
    simulate_plan,
    structural_drills,
)
from market_space_twin.portfolio_risk.storage import (
    METHOD_VERSION,
    RiskStore,
    risk_run_seed,
)
from market_space_twin.simulation.twin_store import fingerprint


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _dependence_section(estimation: EstimationResult, portfolio: PortfolioInput) -> dict[str, Any]:
    interpretation = (
        "within-sector loading is measured from the leave-one-out sector factor; cross-sector "
        "correlation is shrunk toward a structural prior and reported with an identification band, "
        "so independence is never assumed. Candidate tenants use a sector proxy and an interval."
    )
    return {
        "tenants": [estimation.tenants[tenant].as_dict() for tenant in estimation.matrix_ids],
        "correlation_order": estimation.matrix_ids,
        "correlation_matrix": estimation.correlation_matrix,
        "matrix_semantics": (
            "latent stress-propensity correlation used by the model; the realised shortfall-indicator "
            "correlation is lower because stress is a threshold event, and both are driven by the same numbers"
        ),
        "within_sector": estimation.within_sector,
        "cross_sector": estimation.cross_sector,
        "identification": estimation.identification,
        "data_quality": estimation.data_quality,
        "interpretation": interpretation,
    }


def _run_plans(
    plans: list[PlanExposure], estimation: EstimationResult, paths: int
) -> tuple[
    dict[str, SimulationResult],
    list[PlanExposure],
    dict[str, SimulationResult],
    list[PlanExposure],
    dict[str, SimulationResult],
]:
    """Three views of the same roster.

    * committed: contract term exactly as signed (a stub in this fixture);
    * annual: each roster held 12 equal months at its own contract rent;
    * matched annual: every plan scaled to the reference plan's annual rent, so
      only tenant mix and dependence change while revenue is unchanged.
    """
    committed = {plan.name: simulate_plan(plan, estimation, paths=paths) for plan in plans}
    annual = [normalised_plan(plan, months=12) for plan in plans]
    annual_sims = {
        plan.name: simulate_plan(plan, estimation, paths=paths, seed=SIMULATION_SEED + 1)
        for plan in annual
    }
    reference_total = annual_sims[plans[0].name].total_rent
    matched = []
    for plan in annual:
        # The reference plan defines the revenue assumption; every plan is
        # scaled to that same annual rent so risk differences are mix-only.
        plan_total = annual_sims[plan.name].total_rent
        scale = reference_total / plan_total if plan_total > 0 else 1.0
        scaled = scale_plan_rent(plan, scale)
        if abs(scale - 1.0) >= 0.0001:
            scaled.notes.append(
                f"rents scaled by {scale:.4f} to the reference plan's annual rent "
                f"({reference_total:.2f}); revenue assumption unchanged, mix and dependence change"
            )
        matched.append(scaled)
    matched_sims = {
        plan.name: simulate_plan(plan, estimation, paths=paths, seed=SIMULATION_SEED + 2)
        for plan in matched
    }
    return committed, annual, annual_sims, matched, matched_sims


def _headline(
    matched_plans: list[PlanExposure],
    matched_sims: dict[str, SimulationResult],
    matched_contributions: dict[str, Any],
    keep_name: str | None,
) -> dict[str, Any]:
    """One-screen comparison; the detailed sections remain the evidence."""
    rows = []
    for plan in matched_plans:
        sim = matched_sims[plan.name]
        row = {
            "plan": plan.name,
            "annual_exposed_rent": _round(sim.total_rent, 2),
            "expected_uncollected": _round(sim.mean_loss, 2),
            "uncollected_sd": _round(sim.sd_loss, 2),
            "uncollected_es_95": _round(sim.es_95, 2),
            "sum_of_tenant_standard_deviations": _round(sim.sd_tenant_sum, 2),
            "diversification_ratio": _round(sim.sd_loss / sim.sd_tenant_sum, 4) if sim.sd_tenant_sum else None,
            "tenants": plan.tenants,
            "sectors": sorted({segment.sector for segment in plan.segments}),
        }
        if keep_name and plan.name != keep_name:
            delta = matched_contributions["scenario_deltas"].get(plan.name, {})
            row["es_95_delta_vs_reference"] = _round(sim.es_95 - matched_sims[keep_name].es_95, 2)
            row["largest_drivers"] = [
                {
                    "tenant_id": driver["tenant_id"],
                    "sector": driver["sector"],
                    "es_95_delta": driver["es_95_delta"],
                    "lease_references": driver["lease_references"],
                }
                for driver in delta.get("drivers", [])[:3]
            ]
        rows.append(row)
    return {
        "reference_plan": keep_name,
        "basis": "annual matched comparison at equal contract rent; see scenarios.annual_normalisation for the normalisation",
        "plans": rows,
    }


def _assemble(
    portfolio: PortfolioInput,
    estimation: EstimationResult,
    plans: list[PlanExposure],
    horizon_notes: list[str],
    paths: int,
) -> dict[str, Any]:
    keep_name = plans[0].name if plans else None
    committed_sims, annual_plans, annual_sims, matched_plans, matched_sims = _run_plans(plans, estimation, paths)

    exposure_section = {
        "horizon": {
            "start": plans[0].horizon_start.isoformat(),
            "end": plans[0].horizon_end.isoformat(),
            "basis": "committed term from as_of; first and last months prorated by active fraction",
            "notes": horizon_notes,
        },
        "lease_references": [
            {
                "lease_id": lease.lease_id,
                "tenant_id": lease.tenant_id,
                "unit_id": lease.unit_id,
                "sector": lease.sector,
                "start": lease.start.isoformat(),
                "end": lease.end.isoformat(),
                "monthly_contract": _round(lease.monthly_contract, 2),
            }
            for lease in portfolio.leases
        ],
        "plans": {plan.name: plan.as_dict(estimation) for plan in plans},
        "vacancy_policy": (
            "vacancies are reported as their own intervals; a vacant interval never carries tenant exposure "
            "and never appears in a tenant's contribution"
        ),
    }

    plan_metrics = {}
    for plan in plans:
        metrics = committed_sims[plan.name].metrics(plan)
        metrics["vacant_months"] = _round(sum(vacancy.months for vacancy in plan.vacancies), 2)
        metrics["vacant_units"] = sorted({vacancy.unit_id for vacancy in plan.vacancies})
        plan_metrics[plan.name] = metrics

    shocks = management_shock_metrics(plans, estimation, portfolio.scenarios)
    drills = structural_drills(plans, estimation)
    drills["new_tenant_no_history"] = new_tenant_drill(plans, estimation, portfolio)
    annual_normalisation = {
        "label": "annual run-rate comparison",
        "basis": (
            "the committed term in this fixture ends days after as_of; to compare the plans the manager asked about, "
            "each roster is held for 12 equal months at contract rent. The matched view then scales every plan to "
            "the reference plan's annual rent, so revenue is unchanged and only mix and dependence move. "
            "This is a normalisation, not a renewal forecast."
        ),
        "revenue_assumption_unchanged": True,
        "annual_plans": {
            plan.name: annual_sims[plan.name].metrics(plan) for plan in annual_plans
        },
        "matched": {
            "scale_notes": {plan.name: plan.notes for plan in matched_plans},
            "plans": {
                plan.name: matched_sims[plan.name].metrics(plan) for plan in matched_plans
            },
            "comparison": [
                {
                    "plan": plan.name,
                    "expected_uncollected_reference": _round(matched_sims[keep_name].mean_loss, 2),
                    "expected_uncollected_plan": _round(matched_sims[plan.name].mean_loss, 2),
                    "expected_uncollected_delta_vs_reference": _round(
                        matched_sims[plan.name].mean_loss - matched_sims[keep_name].mean_loss, 2
                    ),
                    "uncollected_sd_delta_vs_reference": _round(
                        matched_sims[plan.name].sd_loss - matched_sims[keep_name].sd_loss, 2
                    ),
                    "es_95_reference": _round(matched_sims[keep_name].es_95, 2),
                    "es_95_plan": _round(matched_sims[plan.name].es_95, 2),
                    "es_95_delta_vs_reference": _round(
                        matched_sims[plan.name].es_95 - matched_sims[keep_name].es_95, 2
                    ),
                }
                for plan in matched_plans
            ]
            if keep_name
            else [],
        },
    }

    scenario_section = {
        "historical_term": {
            "label": "estimate from history",
            "basis": (
                "estimated collection means and monthly variance on the committed term; monthly dependence "
                "is assumed iid and dependence is across tenants within a month. This is a term-risk estimate, "
                "not a seasonal forecast and not a footfall model."
            ),
            "plans": plan_metrics,
            "comparison": [
                {
                    "plan": plan.name,
                    "expected_uncollected_delta_vs_reference": _round(
                        committed_sims[plan.name].mean_loss - committed_sims[keep_name].mean_loss, 2
                    ),
                    "es_95_delta_vs_reference": _round(
                        committed_sims[plan.name].es_95 - committed_sims[keep_name].es_95, 2
                    ),
                    "sd_delta_vs_reference": _round(
                        committed_sims[plan.name].sd_loss - committed_sims[keep_name].sd_loss, 2
                    ),
                }
                for plan in plans
            ]
            if keep_name
            else [],
        },
        "management_shocks": shocks,
        "annual_normalisation": annual_normalisation,
        "drills": drills,
        "separation": (
            "historical_term is estimated from observed cash flows; management_shocks and drills are "
            "deterministic management assumptions with not_a_probability=true. They are never mixed into "
            "the estimate or reported as a verified probability."
        ),
    }

    committed_contributions = contribution_report(plans, committed_sims, estimation, keep_name)
    matched_contributions = contribution_report(
        matched_plans, matched_sims, estimation, keep_name
    )
    annual_normalisation["matched"]["contributions"] = matched_contributions["plans"]
    annual_normalisation["matched"]["drivers_vs_reference"] = matched_contributions["scenario_deltas"]

    histories = {tenant: estimate.history_months for tenant, estimate in estimation.tenants.items()}
    observed = [value for value in histories.values() if value > 0]
    missing_total = sum(estimation.data_quality["missing_months_not_zero_filled"].values())
    unidentified_cross = [item["sectors"] for item in estimation.cross_sector if not item["identified"]]
    identified_within = [item["sector"] for item in estimation.within_sector if item["identified"]]
    no_history = sorted(
        tenant for tenant, estimate in estimation.tenants.items() if estimate.history_months == 0
    )
    limitations = {
        "short_history": [
            f"observed tenant histories range from {min(observed)} to {max(observed)} months",
            "calendar-month factors are observed at most once per month-of-year; a single observed March is not a reliable March effect",
            "monthly observations are few relative to the parameter count, so means, variances and factor loadings are shrunk toward sector pools",
        ],
        "missing_data": [
            f"{missing_total} covered tenant-months have no usable fact; they are missing, never zero",
            "partial lease months are prorated; months outside the observed window stay unknown",
            "candidate tenants with no history: "
            + (", ".join(no_history) if no_history else "none")
            + "; their means and intervals are sector proxies, not tenant observations",
        ],
        "dependence": [
            f"within-sector dependence is identified for: {', '.join(identified_within) if identified_within else 'none'}",
            "cross-sector dependence is shrunk toward the structural prior and carries an identification band; identified sectors: "
            + ("none" if unidentified_cross else "all pairs")
            + (f"; not identified for {unidentified_cross}" if unidentified_cross else ""),
            "idiosyncratic residuals are assumed independent conditional on the sector factor; their pairwise correlation is not estimable from this panel",
            "zero correlation is never assumed; the central cross-sector estimate may sit near zero when the observed overlap is negative",
        ],
        "tail": [
            "VaR95 and ES95 come from the latent two-state monthly model; outcomes beyond the observed stress depth are outside the data",
            "monthly independence is an assumption; clustered seasonal stress is shown separately as a drill, not folded into the estimate",
            "the tail is conditional on the estimated parameters and does not add parameter uncertainty",
        ],
        "assumptions_not_probabilities": [
            "scenarios.shocks are management assumptions applied multiplicatively, not probabilities",
            "common-downside and off-season drills are deterministic rehearsals, not term forecasts",
            "the matched annual view is a revenue-held normalisation, not a renewal forecast",
            "the analysis does not predict footfall and does not decide the leasing list",
        ],
        "unidentifiable": [
            "future tenant cash flow, traffic and conversion",
            "cross-sector co-movement below the minimum overlap",
            "idiosyncratic pairwise dependence from this short panel",
        ],
        "model_choices": [
            "collection performance is received / (reported due - explicit concession); a concession is relief, not arrears",
            "dependence is a sector common factor plus idiosyncratic residual; the simulation and the analytic covariance use the same numbers",
            "risk is reported on contract rent at the prorated committed term and, for plan comparison, on a 12-month matched normalisation",
        ],
    }

    return {
        "method_version": METHOD_VERSION,
        "source_versions": portfolio.source_versions,
        "as_of": portfolio.as_of.isoformat(),
        "headline": _headline(
            matched_plans, matched_sims, matched_contributions, keep_name
        ),
        "exposures": exposure_section,
        "dependence": _dependence_section(estimation, portfolio),
        "scenarios": scenario_section,
        "contributions": committed_contributions,
        "limitations": limitations,
    }


def analyze(
    input_path: str,
    database: str,
    output_path: str | None = None,
    paths: int = DEFAULT_PATHS,
) -> dict[str, Any]:
    with open(input_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    portfolio = load_portfolio_input(raw)
    input_fingerprint = fingerprint(raw)
    store = RiskStore(database)
    twin = store.find_twin_baseline(portfolio.baseline_case_id)
    if portfolio.baseline_case_id and twin is None:
        raise RiskInputError(
            f"baseline_case_id {portfolio.baseline_case_id!r} does not reference an existing twin run in the database; "
            "run the baseline fixture first"
        )
    twin_run_id = twin["run_id"] if twin else None
    risk_run_id = risk_run_seed(raw, twin)
    remembered = store.stored(risk_run_id, input_fingerprint)
    if remembered is not None:
        result = dict(remembered)
        result.pop("_stored_twin_run_id", None)
        result["replayed"] = True
        result["storage"] = store.stats()
        if output_path:
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
        return result

    estimation = estimate_dependence(portfolio)
    horizon_start, horizon_end, horizon_notes = resolve_horizon(portfolio)
    plans = [
        build_plan(portfolio, spec, horizon_start, horizon_end)
        for spec in portfolio.scenarios
    ]
    if not plans:
        raise RiskInputError("no scenarios to compare")
    body = _assemble(portfolio, estimation, plans, horizon_notes, paths)
    twin_reference = {
        "baseline_case_id": portfolio.baseline_case_id,
        "run_id": twin_run_id,
        "state": twin["state"] if twin else None,
        "revision": twin["revision"] if twin else None,
        "referenced": twin is not None,
    }
    envelope = {
        "risk_run_id": risk_run_id,
        "project": "market_space_twin",
        "method_version": METHOD_VERSION,
        "input_fingerprint": input_fingerprint,
        "twin_reference": twin_reference,
        "plans": [plan.name for plan in plans],
        "replayed": False,
        **body,
    }
    store.remember(
        risk_run_id,
        input_fingerprint,
        twin_run_id,
        portfolio.as_of.isoformat(),
        envelope,
    )
    # Storage counters describe the database after this run is recorded, so
    # they are attached after the insert and before the response is returned.
    response = dict(envelope, storage=store.stats())
    if output_path:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(response, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
    return response
