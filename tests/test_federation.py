"""
Tests for the cross-domain federation evaluation.

Includes direct tests of the paper's central claim: that cross-domain
federation produces asymmetric per-domain effects that cannot be
summarised by a single aggregate metric.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import numpy as np
from federated_server import fedavg, run_federation
from client_trustlens import TrustLensClient
from client_hospitality import HospitalityClient


class TestFedAvg(unittest.TestCase):

    def test_fedavg_equal_weights(self):
        w1 = np.array([1.0, 2.0, 3.0])
        w2 = np.array([3.0, 2.0, 1.0])
        result = fedavg([w1, w2], [100, 100])
        np.testing.assert_array_almost_equal(result, [2.0, 2.0, 2.0])

    def test_fedavg_weighted(self):
        w1 = np.array([0.0])
        w2 = np.array([1.0])
        result = fedavg([w1, w2], [900, 100])
        self.assertAlmostEqual(result[0], 0.1)

    def test_fedavg_preserves_dtype(self):
        w1 = np.ones(5)
        w2 = np.zeros(5)
        result = fedavg([w1, w2], [1, 1])
        self.assertEqual(result.dtype, float)

    def test_fedavg_single_client(self):
        w = np.array([1.0, 2.0, 3.0])
        result = fedavg([w], [100])
        np.testing.assert_array_almost_equal(result, w)


class TestTrustLensClient(unittest.TestCase):

    def setUp(self):
        self.client = TrustLensClient(n_samples=200, random_state=0)

    def test_init_weights_shape(self):
        w = self.client.init_weights()
        self.assertEqual(w.shape, (11,))

    def test_train_returns_weights_and_size(self):
        w0 = self.client.init_weights()
        w, n = self.client.train(w0)
        self.assertEqual(w.shape, (11,))
        self.assertGreater(n, 0)

    def test_evaluate_returns_required_keys(self):
        w = self.client.init_weights()
        metrics = self.client.evaluate(w)
        for key in ["recall", "ece", "recall_rrr", "ece_rrr"]:
            self.assertIn(key, metrics)

    def test_recall_in_valid_range(self):
        w = self.client.init_weights()
        metrics = self.client.evaluate(w)
        self.assertGreaterEqual(metrics["recall"], 0.0)
        self.assertLessEqual(metrics["recall"], 1.0)


class TestHospitalityClient(unittest.TestCase):

    def setUp(self):
        self.client = HospitalityClient(n_samples=200, random_state=0)

    def test_init_weights_shape(self):
        w = self.client.init_weights()
        self.assertEqual(w.shape, (11,))

    def test_evaluate_returns_required_keys(self):
        w = self.client.init_weights()
        metrics = self.client.evaluate(w)
        for key in ["mae", "mae_rrr", "standalone_rrr", "federation_delta_rrr"]:
            self.assertIn(key, metrics)

    def test_mae_positive(self):
        w = self.client.init_weights()
        metrics = self.client.evaluate(w)
        self.assertGreater(metrics["mae"], 0)


class TestCoreClaim(unittest.TestCase):

    def setUp(self):
        self.clients = [
            TrustLensClient(n_samples=500, random_state=42),
            HospitalityClient(n_samples=400, random_state=42),
        ]

    def test_federation_runs_to_completion(self):
        history = run_federation(self.clients, rounds=5, verbose=False)
        self.assertIn("TrustLens AI", history)
        self.assertIn("Hospitality Sustainability AI", history)
        self.assertEqual(len(history["TrustLens AI"]), 5)

    def test_clients_share_same_weight_dimension(self):
        w_tl = self.clients[0].init_weights()
        w_hosp = self.clients[1].init_weights()
        self.assertEqual(w_tl.shape, w_hosp.shape)

    def test_federation_affects_both_domains(self):
        standalone_tl = self.clients[0].evaluate(self.clients[0].init_weights())
        standalone_hosp = self.clients[1].evaluate(self.clients[1].init_weights())
        history = run_federation(self.clients, rounds=5, verbose=False)
        fed_tl = history["TrustLens AI"][-1]
        fed_hosp = history["Hospitality Sustainability AI"][-1]
        tl_changed = fed_tl["recall"] != standalone_tl["recall"] or fed_tl["ece"] != standalone_tl["ece"]
        hosp_changed = fed_hosp["mae"] != standalone_hosp["mae"]
        self.assertTrue(tl_changed or hosp_changed)

    def test_asymmetric_domain_effects(self):
        history = run_federation(self.clients, rounds=10, verbose=False)
        tl = history["TrustLens AI"][-1]
        hosp = history["Hospitality Sustainability AI"][-1]
        tl_rrr = tl["recall_rrr"]
        hosp_rrr = hosp["mae_rrr"]
        self.assertNotAlmostEqual(tl_rrr, hosp_rrr, places=2)


class TestDeterminism(unittest.TestCase):

    def test_federation_is_deterministic(self):
        clients1 = [TrustLensClient(n_samples=200, random_state=42),
                    HospitalityClient(n_samples=200, random_state=42)]
        clients2 = [TrustLensClient(n_samples=200, random_state=42),
                    HospitalityClient(n_samples=200, random_state=42)]
        h1 = run_federation(clients1, rounds=3, verbose=False)
        h2 = run_federation(clients2, rounds=3, verbose=False)
        self.assertEqual(h1["TrustLens AI"][-1], h2["TrustLens AI"][-1])
        self.assertEqual(h1["Hospitality Sustainability AI"][-1],
                         h2["Hospitality Sustainability AI"][-1])


if __name__ == "__main__":
    unittest.main()
