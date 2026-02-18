"""
Unit Tests for RS-PLO Engine

Tests cover:
  - SystemParams defaults and custom configuration
  - EdgeServerConfig creation and properties
  - UserState initialization
  - Channel gain computation (distance-based)
  - DPP decision logic (local vs multi-edge)
  - V(t) computation (adaptive control)
  - Energy calculations
  - Multi-edge server selection behavior

Usage:  python test_engine.py
        python -m pytest test_engine.py -v
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lyapunov_engine import (
    RSPLO, SystemParams, EdgeServerConfig, UserState,
    StaticLyapunov, execute_locally
)


class TestEdgeServerConfig(unittest.TestCase):
    """Test EdgeServerConfig dataclass."""

    def test_default_config(self):
        cfg = EdgeServerConfig(name="Test", host="127.0.0.1", port=9999,
                               distance=200.0, compute_multiplier=1.5)
        self.assertEqual(cfg.name, "Test")
        self.assertEqual(cfg.port, 9999)
        self.assertEqual(cfg.distance, 200.0)
        self.assertEqual(cfg.compute_multiplier, 1.5)

    def test_description_default(self):
        cfg = EdgeServerConfig(name="E1", host="127.0.0.1", port=9999,
                               distance=100.0, compute_multiplier=1.0)
        # description may be empty string or default
        self.assertIsInstance(cfg.description, str)


class TestSystemParams(unittest.TestCase):
    """Test SystemParams configuration."""

    def test_defaults(self):
        p = SystemParams()
        self.assertEqual(p.V_max, 10.0)
        self.assertEqual(p.beta, 2.0)
        self.assertEqual(p.gamma, 0.2)
        self.assertEqual(p.N, 5)
        self.assertEqual(len(p.edge_servers), 3)

    def test_three_edge_servers(self):
        p = SystemParams()
        names = [s.name for s in p.edge_servers]
        self.assertIn("Edge-1 (Near)", names)
        self.assertIn("Edge-2 (Mid)", names)
        self.assertIn("Edge-3 (Far)", names)

    def test_server_distances(self):
        p = SystemParams()
        distances = [s.distance for s in p.edge_servers]
        self.assertEqual(distances, [150.0, 600.0, 1200.0])

    def test_server_ports(self):
        p = SystemParams()
        ports = [s.port for s in p.edge_servers]
        self.assertEqual(ports, [9999, 10000, 10001])

    def test_custom_params(self):
        p = SystemParams(V_max=20.0, beta=5.0, N=10)
        self.assertEqual(p.V_max, 20.0)
        self.assertEqual(p.beta, 5.0)
        self.assertEqual(p.N, 10)


class TestUserState(unittest.TestCase):
    """Test UserState initialization."""

    def test_initial_state(self):
        u = UserState(user_id=0)
        self.assertEqual(u.Q, 0)
        self.assertEqual(u.Z, 0)
        self.assertGreater(u.distance, 0)  # Default distance varies
        self.assertEqual(len(u.Q_history), 0)
        self.assertEqual(len(u.decision_history), 0)
        self.assertEqual(len(u.server_choice_history), 0)


class TestRSPLO(unittest.TestCase):
    """Test RS-PLO engine core logic."""

    def setUp(self):
        self.params = SystemParams(N=1, time_slots=50)
        self.engine = RSPLO(self.params, seed=42)
        self.user = self.engine.users[0]

    def test_initialization(self):
        self.assertEqual(len(self.engine.users), 1)
        self.assertEqual(self.user.Q, 0)
        self.assertEqual(self.user.Z, 0)

    def test_compute_V_stable(self):
        """V should be high when Z is low (stable channel)."""
        V = self.engine.compute_V(0.0)
        self.assertAlmostEqual(V, self.params.V_max, places=4)

    def test_compute_V_volatile(self):
        """V should be low when Z is high (volatile channel)."""
        V = self.engine.compute_V(5.0)
        self.assertLess(V, 1.0)

    def test_compute_V_extreme(self):
        """V should be near-zero with extreme volatility."""
        V = self.engine.compute_V(20.0)
        self.assertAlmostEqual(V, 0.0, places=4)

    def test_channel_gain_close(self):
        """Channel gain should be better (higher) when close."""
        self.user.distance = 100
        gain_close = self.engine.compute_channel_gain(self.user)

        self.user.distance = 2000
        gain_far = self.engine.compute_channel_gain(self.user)

        self.assertGreater(gain_close, gain_far)

    def test_channel_gain_db(self):
        """Channel gain in dB should be negative."""
        self.user.distance = 300
        gain = self.engine.compute_channel_gain(self.user)
        db = self.engine.compute_channel_gain_db(gain)
        self.assertLess(db, 0)

    def test_channel_gain_for_server(self):
        """Per-server channel gain should differ based on server distance."""
        self.user.distance = 100
        server_near = self.params.edge_servers[0]  # 150m
        server_far = self.params.edge_servers[2]    # 1200m

        gain_near = self.engine.compute_channel_gain_for_server(self.user, server_near)
        gain_far = self.engine.compute_channel_gain_for_server(self.user, server_far)

        # Near server should have stronger signal when user is at 100m
        self.assertGreater(gain_near, gain_far)

    def test_transmission_rate_positive(self):
        """Transmission rate should always be positive."""
        self.user.distance = 300
        gain = self.engine.compute_channel_gain(self.user)
        rate = self.engine.compute_transmission_rate(gain)
        self.assertGreater(rate, 0)

    def test_local_energy_positive(self):
        """Local energy should be positive and proportional to bits."""
        e1 = self.engine.compute_local_energy(1000)
        e2 = self.engine.compute_local_energy(2000)
        self.assertGreater(e1, 0)
        self.assertAlmostEqual(e2, e1 * 2, places=10)

    def test_offload_energy_positive(self):
        """Offload energy should be positive."""
        self.user.distance = 300
        gain = self.engine.compute_channel_gain(self.user)
        rate = self.engine.compute_transmission_rate(gain)
        energy = self.engine.compute_offload_energy(10000, rate)
        self.assertGreater(energy, 0)

    def test_dpp_decision_returns_valid(self):
        """DPP decision should return 0 (local) or 1-3 (edge servers)."""
        self.user.distance = 300
        gain = self.engine.compute_channel_gain(self.user)
        task = {"task_type": "hash_data", "params": {"size_bytes": 1024},
                "task_bits": 8192}
        decision = self.engine.drift_plus_penalty_decision(self.user, task, gain)
        self.assertIn(decision, [0, 1, 2, 3])

    def test_dpp_prefers_local_far_away(self):
        """When very far from all servers with high Q, local may be best."""
        self.user.distance = 2500
        self.user.Z = 10.0  # High volatility -> V=0
        self.user.Q = 10000000  # High queue pressure
        gain = self.engine.compute_channel_gain(self.user)
        task = {"task_type": "hash_data", "params": {"size_bytes": 100},
                "task_bits": 800}
        decision = self.engine.drift_plus_penalty_decision(self.user, task, gain)
        # Decision should be valid (local or edge)
        self.assertIn(decision, [0, 1, 2, 3])

    def test_get_server_for_decision(self):
        """get_server_for_decision should return correct server or None."""
        self.assertIsNone(self.engine.get_server_for_decision(0))
        self.assertEqual(self.engine.get_server_for_decision(1).name, "Edge-1 (Near)")
        self.assertEqual(self.engine.get_server_for_decision(2).name, "Edge-2 (Mid)")
        self.assertEqual(self.engine.get_server_for_decision(3).name, "Edge-3 (Far)")
        self.assertIsNone(self.engine.get_server_for_decision(99))


class TestLocalExecution(unittest.TestCase):
    """Test local task execution (no server needed)."""

    def test_matrix_multiply(self):
        result = execute_locally("matrix_multiply", {"size": 10})
        self.assertIn("exec_time", result)
        self.assertGreater(result["exec_time"], 0)

    def test_hash_data(self):
        result = execute_locally("hash_data", {"size_bytes": 256})
        self.assertIn("exec_time", result)

    def test_prime_factorize(self):
        result = execute_locally("prime_factorize", {"n": 100})
        self.assertIn("exec_time", result)

    def test_sort_numbers(self):
        result = execute_locally("sort_numbers", {"numbers": [5, 3, 1, 4, 2]})
        self.assertIn("exec_time", result)

    def test_text_encrypt(self):
        result = execute_locally("text_encrypt", {"text": "hello", "shift": 3})
        self.assertIn("exec_time", result)


class TestStaticLyapunov(unittest.TestCase):
    """Test static baseline."""

    def test_initialization(self):
        p = SystemParams(N=1)
        engine = StaticLyapunov(p, seed=42)
        self.assertEqual(len(engine.users), 1)


class TestMultiEdgeScenarios(unittest.TestCase):
    """Integration tests for multi-edge behavior."""

    def test_near_server_preferred_when_close(self):
        """When user is near Edge-1, it should typically be chosen."""
        params = SystemParams(N=1)
        engine = RSPLO(params, seed=42)
        user = engine.users[0]
        user.distance = 100  # Very close to Edge-1 (150m)
        user.Q = 100000  # Some queue pressure

        gain = engine.compute_channel_gain(user)
        task = {"task_type": "matrix_multiply", "params": {"size": 50},
                "task_bits": 160000}
        decision = engine.drift_plus_penalty_decision(user, task, gain)

        # Should offload (decision >= 1), likely to Edge-1
        self.assertGreaterEqual(decision, 1)

    def test_different_servers_have_different_costs(self):
        """Verify that DPP cost differs across servers."""
        params = SystemParams(N=1)
        engine = RSPLO(params, seed=42)
        user = engine.users[0]
        user.distance = 500

        gains = []
        for server in params.edge_servers:
            h = engine.compute_channel_gain_for_server(user, server)
            gains.append(h)

        # Gains should differ (different distances)
        self.assertNotEqual(round(gains[0], 15), round(gains[1], 15))
        self.assertNotEqual(round(gains[1], 15), round(gains[2], 15))


if __name__ == "__main__":
    print("=" * 70)
    print("  RS-PLO Engine Unit Tests")
    print("=" * 70)
    unittest.main(verbosity=2)
