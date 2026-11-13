import json
import unittest
from pathlib import Path


class AnalysisInputsTests(unittest.TestCase):
    def load_data(self):
        return json.loads((Path(__file__).resolve().parents[1] / "fixtures/analysis/input.json").read_text())

    def test_new_tenant(self):
        data = self.load_data()
        self.assertTrue(not data['candidate_tenants'][0]['historical_cashflows'])

    def test_unequal_history(self):
        data = self.load_data()
        self.assertTrue(sum(row['tenant_id'] == 'tenant-5' for row in data['cashflows']) < sum(row['tenant_id'] == 'tenant-0' for row in data['cashflows']))

    def test_matched_scenario_rents(self):
        data = self.load_data()
        self.assertTrue(data['candidate_tenants'][0]['monthly_contract'] == data['candidate_tenants'][1]['monthly_contract'])

    def test_cashflow_inside_lease(self):
        data = self.load_data()
        self.assertTrue(all(row['month'] >= next(lease['start'][:7] for lease in data['leases'] if lease['tenant_id'] == row['tenant_id']) for row in data['cashflows']))

    def test_source_versions_and_cutoff(self):
        data = self.load_data()
        self.assertEqual(data["schema_version"], 1)
        self.assertTrue(data["source_versions"])
        self.assertIn("+00:00", data["as_of"])
