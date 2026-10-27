"""Unit tests for the V4.1 evidence-aware pipeline.  python3 -m unittest discover -s tests -v"""
import os, sys, unittest
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
torch.set_num_threads(1)

from aurasync.v41.pipeline import EvidencePipeline, QUIET, UNOBSERVED, FUSED, CONTRADICTORY, CORRUPT, REINIT
from aurasync.v4.geometry import PIVOT_Y
from aurasync.event_camera import SceneRenderer

Q = np.diag([1e-9, 1e-5, 1e-8, 1e-4]) * 1e4


def pole_surface(theta, x=0.0, l=0.25, noise=0, seed=0):
    r = SceneRenderer()
    t = np.clip((r.wy - PIVOT_Y) / (2 * l), 0, 1)
    d = np.abs(r.wx - (x + (r.wy - PIVOT_Y) * np.tan(theta))) * r.S
    surf = np.zeros((4, r.H, r.W), np.float32)
    surf[0][(d < 0.7) & (t > 0) & (t < 1)] = 1.0
    if noise:
        rng = np.random.default_rng(seed); surf[1].flat[rng.integers(0, r.H * r.W, noise)] = 1.0
    return surf


def background(n=3, seed=0):
    """A live sensor in a still scene: a few noise events far from the pole."""
    surf = np.zeros((4, 32, 96), np.float32)
    surf[1, 30, :n] = 1.0
    return surf


class TestEvidence(unittest.TestCase):
    def make(self, **kw):
        p = EvidencePipeline(Q, **kw); p.reset(np.zeros(4)); return p

    def test_quiet_bounds_uncertainty(self):
        p = self.make()
        for k in range(300):
            p.update(k, (k + 1) * 1e-3, None, background(), 0.0)
        self.assertEqual(p.state, QUIET)
        self.assertLessEqual(np.degrees(np.sqrt(p.Sigma[2, 2])), 3.0 + 1e-6)

    def test_absence_is_not_stillness_when_motion_predicted(self):
        p = self.make()
        p.s[3] = 2.0                                   # estimate says the pole is swinging
        before = p.Sigma[2, 2]
        for k in range(10):
            p.update(k, (k + 1) * 1e-3, None, background(), 0.0)
        self.assertEqual(p.state, UNOBSERVED)
        self.assertGreater(p.Sigma[2, 2], before)
        self.assertGreater(np.sqrt(p.Sigma[3, 3]), 0.5)           # velocity doubt beyond the quiet cap

    def test_partial_activity_without_fit_is_unobserved(self):
        p = self.make()
        surf = background(); surf[0, 10:14, 47:49] = 1.0   # activity in the pole region, too little for a fit
        p.update(0, 1e-3, None, surf, 0.0)
        self.assertEqual(p.state, UNOBSERVED)

    def test_consistent_measurement_is_fused(self):
        p = self.make()
        p.update(0, 1e-3, None, pole_surface(np.radians(2)), 0.0)
        self.assertEqual(p.state, FUSED)
        self.assertGreater(p.s[2], 0)

    def test_contradiction_needs_corroboration_then_reinit(self):
        p = self.make()
        for k in range(100):                               # settle on a confident upright estimate
            p.update(k, (k + 1) * 1e-3, None, pole_surface(0.0), 0.0)
        p.update(100, 0.101, None, pole_surface(np.radians(20)), 0.0)
        self.assertIn(p.state, (CONTRADICTORY, CORRUPT))   # a 20 deg jump in 1 ms is first treated as suspect
        self.assertLess(abs(np.degrees(p.s[2])), 5)        # a single contradicting fit does not move the estimate
        states = []
        for k in range(101, 125):
            p.update(k, (k + 1) * 1e-3, None, pole_surface(np.radians(20)), 0.0); states.append(p.state)
        self.assertIn(REINIT, states)
        self.assertAlmostEqual(np.degrees(p.s[2]), 20, delta=2.5)

    def test_noise_burst_is_rejected(self):
        p = self.make()
        for k in range(200):
            p.update(k, (k + 1) * 1e-3, None, background(), 0.0)
        th = p.s[2]
        p.update(200, 0.201, None, pole_surface(np.radians(25), noise=900, seed=1), 0.0)
        self.assertEqual(p.state, CORRUPT)
        self.assertEqual(p.s[2], th)

    def test_neural_fusion_is_position_only(self):
        p = self.make()
        p.Sigma = np.diag([1e-4, 1e-2, np.radians(2) ** 2, 0.5])  # no cross-covariance
        p.models = []; p.u_hist.append(0.0)
        s_c = np.array([0.0, 0.0, np.radians(1), 9.0])           # network claims a wild angular velocity
        p.job = (0, 0, s_c, np.diag([1e-4, 1e-2, np.radians(1) ** 2, 0.01]), None, p.s.copy())
        p._finish_job(0)
        self.assertEqual(p.s[3], 0.0)                              # velocity untouched
        self.assertGreater(p.s[2], 0.0)                            # angle pulled toward the network

    def test_neural_answer_outside_gate_is_rejected(self):
        p = self.make()
        p.Sigma = np.diag([1e-4, 1e-2, np.radians(1) ** 2, 0.5]); p.models = []; p.u_hist.append(0.0)
        s_c = np.array([0.0, 0.0, np.radians(30), 0.0])
        p.job = (0, 0, s_c, np.diag([1e-4, 1e-2, np.radians(1) ** 2, 0.01]), None, p.s.copy())
        p._finish_job(0)
        self.assertEqual(p.s[2], 0.0); self.assertEqual(p.log["rejected"], 1)


@unittest.skipUnless(os.path.exists("checkpoints/v3/event_s0.pt"), "needs V3 checkpoints")
class TestBudget(unittest.TestCase):
    def test_budget_and_max_interval(self):
        import benchmark_v3 as B3
        cal = B3.calibration()
        p = EvidencePipeline(Q, neural=B3.event_model("event_s0"), temp=np.array(cal["temp"]["varhead"]),
                             latency_samples=cal["latency"], min_gap=10, max_interval=50)
        p.reset(np.zeros(4))
        calls_k = []
        for k in range(300):
            c = p.log["calls"]
            surf = background(); surf[0, 10:14, 47:49] = 1.0     # persistent unusable activity -> wants help
            p.update(k, (k + 1) * 1e-3, None, surf, 0.0)
            if p.log["calls"] > c:
                calls_k.append(k)
        gaps = np.diff(calls_k)
        self.assertTrue(len(calls_k) > 0)
        self.assertGreaterEqual(gaps.min(), 10)                    # budget: <= 100 calls/s
        q = EvidencePipeline(Q, neural=B3.event_model("event_s0"), temp=np.array(cal["temp"]["varhead"]),
                             latency_samples=cal["latency"], min_gap=10, max_interval=50)
        q.reset(np.zeros(4)); ks = []
        for k in range(300):
            c = q.log["calls"]; q.update(k, (k + 1) * 1e-3, None, background(), 0.0)
            if q.log["calls"] > c:
                ks.append(k)
        self.assertLessEqual(np.diff(ks).max(), 50)                # periodic check-in even when quiet


if __name__ == "__main__":
    unittest.main(verbosity=2)
