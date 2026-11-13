"""Business counterexamples the analysis must satisfy.

These tests are statistical/contractual checks, not schema checks.  They cover
the cases the business description calls out:

* a vacancy interval must never be old or new tenant exposure;
* a missing month is not zero, and a short-history tenant enters the same
  analysis as a long-history one without pretending to have data;
* independence is not assumed: the portfolio sd is not the sum of tenant sds;
* management shocks stay labelled assumptions and never become probabilities;
* an unbalanced panel does not turn absence into a 100% or 0% collection rate;
* replays of a published comparison return the same run id.
"""

from __future__ import annotations

import copy
import json
import math
import statistics
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from market_space_twin.portfolio_risk.analysis import analyze
from market_space_twin.portfolio_risk.estimation import (
    estimate_dependence,
    stress_parameters,
)
from market_space_twin.portfolio_risk.model import (
    RiskInputError,
    load_portfolio_input,
    month_fraction,
)
from market_space_twin.portfolio_risk.scenarios import (
    build_plan,
    normalised_plan,
    resolve_horizon,
    scale_plan_rent,
    simulate_plan,
)

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures/analysis/input.json"
BASELINE = Path(__file__).resolve().parents[2] / "fixtures/demo.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class BaseCase(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = load_fixture()
        self.portfolio = load_portfolio_input(self.raw)
        self.estimation = estimate_dependence(self.portfolio)
        self.horizon = resolve_horizon(self.portfolio)
        self.plans = [
            build_plan(self.portfolio, spec, self.horizon[0], self.horizon[1])
            for spec in self.portfolio.scenarios
        ]

    def mutate(self) -> dict:
        return copy.deepcopy(self.raw)


class VacancyOwnershipTests(BaseCase):
    def test_replacement_does_not_double_expose_the_unit(self):
        keep = next(plan for plan in self.plans if plan.name == "keep")
        concentrated = next(plan for plan in self.plans if plan.name == "concentrated")
        starts_at = self.portfolio.scenarios[1].replacement.starts_at
        # The old tenant must not be exposed at or after starts_at.
        for segment in concentrated.segments:
            if segment.tenant_id == "tenant-5":
                self.assertLessEqual(segment.end, starts_at)
        # Up to starts_at both plans agree, and the old lease is only truncated.
        for segment in keep.segments:
            if segment.tenant_id == "tenant-5":
                self.assertGreater(segment.end, starts_at)
        self.assertNotIn("tenant-5", concentrated.tenants)
        self.assertIn("tenant-new", concentrated.tenants)

    def test_no_unit_has_two_segments_at_once(self):
        for plan in self.plans:
            by_unit = {}
            for segment in plan.segments:
                by_unit.setdefault(segment.unit_id, []).append(segment)
            for unit, items in by_unit.items():
                ordered = sorted(items, key=lambda item: item.start)
                for left, right in zip(ordered, ordered[1:]):
                    self.assertLessEqual(
                        right.start,
                        left.end,
                        msg=f"{plan.name}: {unit} has overlapping old/new exposure",
                    )

    def test_vacancy_is_not_tenant_exposure(self):
        gap = self.mutate()
        # The old lease ends 2026-11-09 and the replacement only starts
        # 2026-12-01, so unit-5 is genuinely vacant between them.
        gap["scenarios"][1]["replacement"]["starts_at"] = "2026-12-01T00:00:00+00:00"
        gap["scenarios"][1]["replacement"]["ends_at"] = "2027-01-31T00:00:00+00:00"
        gap["analysis"] = {"horizon_end": "2027-01-31T00:00:00+00:00"}
        portfolio = load_portfolio_input(gap)
        horizon = resolve_horizon(portfolio)
        plan = build_plan(portfolio, portfolio.scenarios[1], horizon[0], horizon[1])
        unit5_vacancies = [vacancy for vacancy in plan.vacancies if vacancy.unit_id == "unit-5"]
        self.assertTrue(unit5_vacancies, "the gap between old and new lease must be a vacancy")
        vacancy = unit5_vacancies[0]
        self.assertEqual(vacancy.start, datetime(2026, 11, 9, tzinfo=timezone.utc))
        self.assertEqual(vacancy.end, datetime(2026, 12, 1, tzinfo=timezone.utc))
        old = next(segment for segment in plan.segments if segment.tenant_id == "tenant-5")
        self.assertEqual(old.end, datetime(2026, 11, 9, tzinfo=timezone.utc))
        new = next(segment for segment in plan.segments if segment.tenant_id == "tenant-new")
        self.assertEqual(new.start, datetime(2026, 12, 1, tzinfo=timezone.utc))
        # No tenant exposure may span the vacant interval.
        for segment in plan.segments:
            if segment.unit_id != "unit-5":
                continue
            self.assertFalse(segment.start < vacancy.end and segment.end > vacancy.start)
        # Vacant rent is absent from every tenant's exposure.
        self.assertAlmostEqual(sum(plan.tenant_exposure().values()), plan.total_rent())

    def test_contribution_has_no_vacancy_row(self):
        impulse = self.mutate()
        impulse["scenarios"][1]["replacement"]["starts_at"] = "2026-11-01T00:00:00+00:00"
        impulse["scenarios"][1]["replacement"]["ends_at"] = "2027-01-31T00:00:00+00:00"
        impulse["analysis"] = {"horizon_end": "2026-12-31T00:00:00+00:00"}
        portfolio = load_portfolio_input(impulse)
        horizon = resolve_horizon(portfolio)
        plan = build_plan(portfolio, portfolio.scenarios[1], horizon[0], horizon[1])
        self.assertTrue(plan.vacancies)
        self.assertEqual(
            sorted(plan.tenant_exposure()),
            sorted({segment.tenant_id for segment in plan.segments}),
        )


class ShortHistoryTests(BaseCase):
    def test_missing_months_are_not_zero(self):
        quality = self.estimation.data_quality
        for tenant, missing in quality["missing_months_not_zero_filled"].items():
            self.assertGreaterEqual(missing, 0)
        # tenant-5 has 3 observed months and 6 covered months; the difference
        # must be carried as missing, not used as zero and not as a 100% month.
        self.assertEqual(self.estimation.tenants["tenant-5"].history_months, 3)
        self.assertEqual(quality["coverage_months_by_tenant"]["tenant-5"], 6)
        self.assertGreater(quality["missing_months_not_zero_filled"]["tenant-5"], 0)
        self.assertNotEqual(self.estimation.tenants["tenant-5"].mean_rate, 1.0)

    def test_short_and_long_history_share_one_analysis(self):
        short = self.estimation.tenants["tenant-5"]
        long = self.estimation.tenants["tenant-0"]
        self.assertEqual(short.sector, "food")
        self.assertEqual(long.sector, "textile")
        self.assertLess(short.history_months, long.history_months)
        self.assertTrue(all(math.isfinite(value) for value in short.mean_interval))
        self.assertGreaterEqual(short.mean_interval[1], short.mean_interval[0])
        # Both appear in the same plan and the same correlation matrix.
        plan = normalised_plan(next(p for p in self.plans if p.name == "keep"), 12)
        self.assertIn("tenant-5", plan.tenants)
        self.assertIn("tenant-5", self.estimation.matrix_ids)
        # The short-history tenant is shrunk toward its sector, not dropped and
        # not treated as a 100% payer.
        self.assertNotEqual(short.mean_rate, 1.0)
        self.assertNotEqual(short.mean_rate, short.raw_mean)

    def test_new_tenant_proxy_is_not_zero_and_has_interval(self):
        for tenant_id in ("tenant-new", "tenant-other"):
            estimate = self.estimation.tenants[tenant_id]
            self.assertEqual(estimate.history_months, 0)
            self.assertGreater(estimate.mean_rate, 0.0)
            self.assertLess(estimate.mean_rate, 1.0)
            self.assertIn("proxy", estimate.mean_source)
            lower, upper = estimate.mean_interval
            self.assertGreater(upper, lower)
            self.assertFalse(upper - lower < 1e-9)

    def test_candidate_not_imputed_from_other_sector(self):
        native = self.estimation.tenants["tenant-new"]
        food = self.estimation.tenants["tenant-4"]
        self.assertNotEqual(round(native.mean_rate, 6), round(food.mean_rate, 6))

    def test_concession_is_not_arrears(self):
        # June 2026 has due=100, received=90, concession=5 for tenant-0.
        # Collection rate must be 90/95, not 90/100 or a 10-point miss.
        observations = [
            item
            for item in self.estimation.observations
            if item.tenant_id == "tenant-0" and item.month.month == 6
        ]
        self.assertEqual(len(observations), 1)
        self.assertAlmostEqual(observations[0].rate, 90.0 / 95.0, places=9)
        self.assertEqual(self.estimation.data_quality["concession_months"], 6)


class DependenceTests(BaseCase):
    def test_portfolio_sd_is_not_sum_of_tenant_sd(self):
        plan = normalised_plan(next(p for p in self.plans if p.name == "concentrated"), 12)
        simulation = simulate_plan(plan, self.estimation, paths=8000, seed=7)
        tenant_sd_sum = sum(
            statistics.pstdev(losses) for losses in simulation.tenant_losses.values()
        )
        self.assertGreater(len(simulation.tenant_losses), 1)
        self.assertLess(simulation.sd_loss, tenant_sd_sum * 0.99)
        self.assertTrue(simulation.metrics(plan)["portfolio_sd_not_sum_of_tenant_sd"])
        # The analytic covariance figure must track the simulation as a check
        # that the simulation draws the stated dependence.
        self.assertLess(abs(simulation.sd_loss - simulation.analytic_portfolio_sd) / simulation.sd_loss, 0.08)
        self.assertLess(abs(tenant_sd_sum - simulation.analytic_tenant_sd_sum) / tenant_sd_sum, 0.08)

    def test_within_sector_dependence_is_not_independent(self):
        row = next(item for item in self.estimation.within_sector if item["sector"] == "textile")
        self.assertTrue(row["identified"])
        self.assertGreater(row["estimate"], 0.5)
        for left in ("tenant-0", "tenant-1"):
            for right in ("tenant-2", "tenant-3"):
                index_left = self.estimation.matrix_ids.index(left)
                index_right = self.estimation.matrix_ids.index(right)
                self.assertGreater(self.estimation.correlation_matrix[index_left][index_right], 0.5)

    def test_cross_sector_is_banded_not_assumed_zero(self):
        self.assertTrue(self.estimation.cross_sector)
        for item in self.estimation.cross_sector:
            self.assertIn("identification_band", item)
            lower, upper = item["identification_band"]
            self.assertLess(lower, upper)
            self.assertIn("not_independence", item)
            self.assertNotEqual(item["basis"], "")

    def test_structured_concentration_raises_tail(self):
        annual = normalised_plan(next(p for p in self.plans if p.name == "concentrated"), 12)
        matched = scale_plan_rent(annual, 1.0)
        keep = normalised_plan(next(p for p in self.plans if p.name == "keep"), 12)
        concentrated_sim = simulate_plan(matched, self.estimation, paths=12000, seed=11)
        keep_sim = simulate_plan(keep, self.estimation, paths=12000, seed=11)
        self.assertGreater(concentrated_sim.es_95, keep_sim.es_95)

    def test_stress_probability_model_stays_inside_unit_interval(self):
        for tenant_id, estimate in self.estimation.tenants.items():
            params = stress_parameters(estimate.mean_rate, estimate.variance_rate)
            self.assertGreaterEqual(params["stress_probability"], 0.0)
            self.assertLessEqual(params["stress_probability"], 1.0)
            self.assertGreaterEqual(params["stress_depth"], 0.0)
            self.assertLessEqual(params["stress_depth"], 1.0)
            reconstructed = 1.0 - params["stress_probability"] * params["stress_depth"]
            self.assertAlmostEqual(reconstructed, estimate.mean_rate, places=6)

    def test_observed_panel_is_deterministic(self):
        repeat = estimate_dependence(load_portfolio_input(load_fixture()))
        for tenant in self.estimation.matrix_ids:
            self.assertAlmostEqual(
                repeat.tenants[tenant].mean_rate, self.estimation.tenants[tenant].mean_rate, places=12
            )
        self.assertEqual(repeat.correlation_matrix, self.estimation.correlation_matrix)


class ScenarioSeparationTests(BaseCase):
    def _run(self, paths: int = 4000):
        from market_space_twin.service import run_fixture

        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        database = Path(workspace.name) / "analysis.sqlite3"
        run_fixture(str(BASELINE), str(database))
        return analyze(str(FIXTURE), str(database), None, paths=paths)

    def test_management_shocks_are_labelled_assumptions(self):
        result = self._run()
        shocks = result["scenarios"]["management_shocks"]
        self.assertTrue(shocks["not_a_probability"])
        self.assertTrue(shocks["not_from_history"])
        for group in shocks["sets"]:
            for plan_name, payload in group["plans"].items():
                self.assertIn("expected_uncollected_under_stress", payload)
        # Historical estimates must not carry the assumption flag.
        self.assertNotIn("not_a_probability", result["scenarios"]["historical_term"])
        # Each declared shock set is applied to each plan for comparability.
        for group in shocks["sets"]:
            self.assertEqual(sorted(group["plans"]), sorted(result["plans"]))

    def test_revenue_assumption_unchanged_in_matched_view(self):
        result = self._run()
        matched = result["scenarios"]["annual_normalisation"]["matched"]
        rows = matched["comparison"]
        self.assertGreaterEqual(len(rows), 2)
        for row in rows:
            self.assertLess(
                abs(row["es_95_plan"] - row["es_95_reference"] - row["es_95_delta_vs_reference"]),
                0.05,
            )
        # Scaled to the same annual rent, so the revenue assumption is held.
        for plan_metrics in matched["plans"].values():
            self.assertAlmostEqual(plan_metrics["total_exposed_rent"], 9000.0, places=2)


class ReplayTests(unittest.TestCase):
    def test_replay_returns_the_same_run_id_and_payload(self):
        database = Path(self.id().replace(":", "_") + ".sqlite3")
        output_one = Path(self.id().replace(":", "_") + "-1.json")
        output_two = Path(self.id().replace(":", "_") + "-2.json")
        try:
            # Seed the baseline twin run so baseline_case_id can be resolved.
            from market_space_twin.service import run_fixture

            run_fixture(str(BASELINE), str(database))
            first = analyze(str(FIXTURE), str(database), str(output_one), paths=3000)
            second = analyze(str(FIXTURE), str(database), str(output_two), paths=3000)
            self.assertFalse(first["replayed"])
            self.assertTrue(second["replayed"])
            self.assertEqual(first["risk_run_id"], second["risk_run_id"])
            for section in ("exposures", "dependence", "scenarios", "contributions", "limitations"):
                self.assertEqual(first[section], second[section])
            self.assertTrue(first["twin_reference"]["referenced"])
            for required in ("risk_run_id", "exposures", "dependence", "scenarios", "contributions", "limitations"):
                self.assertIn(required, first)
        finally:
            for path in (database, output_one, output_two):
                path.unlink(missing_ok=True)
            for suffix in ("-wal", "-shm"):
                Path(str(database) + suffix).unlink(missing_ok=True)

    def test_missing_baseline_is_an_error_not_a_silent_pass(self):
        database = Path(self.id().replace(":", "_") + ".sqlite3")
        try:
            with self.assertRaises(RiskInputError):
                analyze(str(FIXTURE), str(database), None, paths=1000)
        finally:
            database.unlink(missing_ok=True)


class InputContractTests(BaseCase):
    def test_late_fact_is_withheld_until_available(self):
        late = self.mutate()
        late["cashflows"][0]["available_at"] = "2026-12-01T00:00:00+00:00"
        portfolio = load_portfolio_input(late)
        estimation = estimate_dependence(portfolio)
        quality = estimation.data_quality
        self.assertTrue(
            any(item["tenant_id"] == "tenant-0" and item["month"] == "2026-01" for item in quality["withheld_late_or_unavailable_facts"])
        )
        self.assertEqual(estimation.tenants["tenant-0"].history_months, 6)

    def test_incomplete_fact_is_excluded_not_zeroed(self):
        incomplete = self.mutate()
        incomplete["cashflows"][0]["received"] = None
        portfolio = load_portfolio_input(incomplete)
        estimation = estimate_dependence(portfolio)
        quality = estimation.data_quality
        self.assertTrue(
            any(item["tenant_id"] == "tenant-0" and item["month"] == "2026-01" for item in quality["incomplete_facts_not_imputed"])
        )
        self.assertEqual(estimation.tenants["tenant-0"].history_months, 6)

    def test_contract_basis_change_is_flagged_and_used(self):
        rebased = self.mutate()
        rebased["cashflows"][0]["due"] = 130.0  # reported due differs from prorated contract 100
        portfolio = load_portfolio_input(rebased)
        estimation = estimate_dependence(portfolio)
        self.assertTrue(estimation.data_quality["contract_basis_adjustments"])
        adjusted = estimation.data_quality["contract_basis_adjustments"][0]
        self.assertEqual(adjusted["reported_due"], 130.0)
        self.assertIn("reported due", adjusted["treatment"])

    def test_overlapping_leases_are_rejected(self):
        overlap = self.mutate()
        overlap["leases"].append(
            {
                "lease_id": "lease-overlap",
                "tenant_id": "tenant-9",
                "unit_id": "unit-0",
                "sector": "textile",
                "start": "2026-06-01T00:00:00+00:00",
                "end": "2026-12-01T00:00:00+00:00",
                "monthly_contract": 100,
            }
        )
        with self.assertRaises(RiskInputError):
            load_portfolio_input(overlap)

    def test_naive_timestamps_are_rejected(self):
        naive = self.mutate()
        naive["as_of"] = "2026-10-30T00:00:00"
        with self.assertRaises(RiskInputError):
            load_portfolio_input(naive)

    def test_concession_unknown_is_not_treated_as_zero(self):
        unknown = self.mutate()
        unknown["cashflows"][4]["concession"] = None
        portfolio = load_portfolio_input(unknown)
        estimation = estimate_dependence(portfolio)
        quality = estimation.data_quality
        self.assertTrue(
            any(item.get("month") == "2026-05" and item["tenant_id"] == "tenant-0" for item in quality["incomplete_facts_not_imputed"])
        )

    def test_partial_month_is_prorated(self):
        # as_of is 30 October, so October is 2/31 active, not a whole month.
        self.assertAlmostEqual(
            month_fraction(
                datetime(2026, 10, 30, tzinfo=timezone.utc),
                datetime(2026, 11, 9, tzinfo=timezone.utc),
                datetime(2026, 10, 1, tzinfo=timezone.utc),
            ),
            2.0 / 31.0,
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
