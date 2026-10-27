"""Unit and integration tests for AuraSync V4.  Run:  python3 -m unittest discover -s tests -v"""
import os, sys, glob, unittest
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
torch.set_num_threads(1)

from aurasync.event_camera import SceneRenderer, TimeSurfaceEncoder
from aurasync.conditions import NOMINAL, SensorConditions
from aurasync.v4.geometry import EventGeometry, PIVOT_Y
from aurasync.v4.pipeline import ConsistencyDetector


def synthetic_surface(theta, x=0.0, l=0.25, noise=0, seed=0):
    """Time surface with fresh ON events exactly along a pole at angle theta (plus optional noise)."""
    r = SceneRenderer()
    t = np.clip((r.wy - PIVOT_Y) / (2 * l), 0, 1)
    d = np.abs(r.wx - (x + (r.wy - PIVOT_Y) * np.tan(theta))) * r.S
    surf = np.zeros((4, r.H, r.W), np.float32)
    surf[0][(d < 0.7) & (t > 0) & (t < 1)] = 1.0
    if noise:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, r.H * r.W, noise)
        surf[1].flat[idx] = 1.0
    return surf


class TestGeometry(unittest.TestCase):
    def test_measures_known_angle(self):
        g = EventGeometry()
        for deg in (-30, -10, 0, 7, 25):
            m = g.measure(synthetic_surface(np.radians(deg)), 0.0)
            self.assertIsNotNone(m); self.assertTrue(m["valid"])
            self.assertLess(abs(np.degrees(m["theta"]) - deg), 1.5, deg)

    def test_robust_to_moderate_noise(self):
        m = EventGeometry().measure(synthetic_surface(np.radians(12), noise=40, seed=3), 0.0)
        self.assertTrue(m is not None and m["valid"])
        self.assertLess(abs(np.degrees(m["theta"]) - 12), 2.0)

    def test_refuses_pure_noise(self):
        surf = np.zeros((4, 32, 96), np.float32)
        rng = np.random.default_rng(1); surf[0].flat[rng.integers(0, 32 * 96, 60)] = 1.0
        m = EventGeometry().measure(surf, 0.0)
        self.assertTrue(m is None or not m["valid"])

    def test_deterministic(self):
        s = synthetic_surface(np.radians(5), noise=30)
        a = EventGeometry(seed=4).measure(s, 0.0); b = EventGeometry(seed=4).measure(s, 0.0)
        self.assertEqual(a["theta"], b["theta"])


class TestDetector(unittest.TestCase):
    def test_alarm_on_persistent_disagreement_only(self):
        det = ConsistencyDetector()
        Sigma = np.diag([1e-4, 1e-2, np.radians(1) ** 2, 0.1])
        m = dict(theta=np.radians(20), sigma_theta=np.radians(1), quality=1.0, valid=True)
        s_hat = np.array([0, 0, np.radians(-5), 0])
        out = det.update(0, s_hat, Sigma, m, 50)
        self.assertFalse(out["alarm"])                         # one outlier is not enough
        for k in range(1, 5):
            out = det.update(k, s_hat, Sigma, m, 50)
        self.assertTrue(out["alarm"]); self.assertIn("sign_mismatch", out["reasons"])
        det.reset()
        agree = dict(m, theta=np.radians(-5))
        for k in range(50):
            out = det.update(k, s_hat, Sigma, agree, 50)
        self.assertFalse(out["alarm"])


class TestConditions(unittest.TestCase):
    def test_nominal_renderer_identical(self):
        s = np.array([0.1, 0, 0.2, 0])
        self.assertTrue(np.array_equal(SceneRenderer().render(s, 0.25), SceneRenderer(cond=NOMINAL).render(s, 0.25)))

    def test_miscalibration_shifts_image(self):
        s = np.zeros(4)
        a = SceneRenderer().render(s, 0.25); b = SceneRenderer(cond=SensorConditions("m", x_offset_m=0.03)).render(s, 0.25)
        self.assertFalse(np.array_equal(a, b))


@unittest.skipUnless(os.path.exists("checkpoints/v3/event_s0.pt") and glob.glob("data/v4dev/ev-trig-s0_*.npz"),
                     "needs V3 checkpoints and the V4 development set")
class TestPipelineIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import benchmark_v3 as B
        cls.B = B
        d = dict(np.load(sorted(glob.glob("data/v4dev/ev-trig-s0_*.npz"))[0]))
        cls.d = d; cls.surf = d["surf"][:400].astype(np.float32)

    def _kw(self):
        cal, tu = self.B.calibration(), self.B.tuning()
        return dict(temp=np.array(cal["temp"]["varhead"]), latency_samples=cal["latency"], Q=np.diag(cal["Q_diag"]), **tu["trigger"])

    def _replay(self, p, n=400):
        p.reset(self.d["s0"].astype(float)); out = []
        for k in range(n):
            p.update(k, (k + 1) * 1e-3, None, self.surf[k], float(self.d["u"][k]))   # s_true=None: no ground truth
            out.append(p.out[-1][0].copy())
        return np.array(out)

    def test_detect_only_equals_v3(self):
        from aurasync.scheduler import TriggeredEventPipeline
        from aurasync.v4.pipeline import V4Pipeline
        m = self.B.event_model("event_s0")
        a = self._replay(TriggeredEventPipeline(m, mode="triggered", **self._kw()))
        b = self._replay(V4Pipeline(m, mode="triggered", correct=False, inflate=False, **self._kw()))
        self.assertLess(np.abs(a - b).max(), 1e-12)

    def test_correction_reduces_error(self):
        from aurasync.v4.pipeline import V4Pipeline
        m = self.B.event_model("event_s0")
        v3 = self._replay(V4Pipeline(m, mode="triggered", correct=False, inflate=False, **self._kw()))
        v4 = self._replay(V4Pipeline(m, mode="triggered", **self._kw()))
        truth = self.d["s"][:400, 2]
        self.assertLess(np.abs(v4[50:, 2] - truth[50:]).mean(), np.abs(v3[50:, 2] - truth[50:]).mean())

    def test_kalman_fusion_moves_toward_measurement(self):
        from aurasync.v4.pipeline import V4Pipeline
        p = V4Pipeline(self.B.event_model("event_s0"), mode="triggered", **self._kw())
        p.reset(np.zeros(4)); p.Sigma = np.diag([1e-4, 1e-2, np.radians(3) ** 2, 0.5])
        before = p.Sigma[2, 2]
        p._fuse(dict(theta=np.radians(4), sigma_theta=np.radians(1), x_pivot=0.0, n_inliers=40))
        self.assertGreater(p.s[2], 0); self.assertLess(p.s[2], np.radians(4)); self.assertLess(p.Sigma[2, 2], before)

    def test_reinit_needs_corroboration(self):
        from aurasync.v4.pipeline import V4Pipeline
        p = V4Pipeline(self.B.event_model("event_s0"), mode="triggered", **self._kw())
        p.reset(np.zeros(4))
        p.buf.extend([(10, np.radians(20), 0.02, 0.0), (11, np.radians(-5), 0.02, 0.0)])
        self.assertFalse(p._confirmed(12))                      # too few, inconsistent
        p.buf.extend([(12, np.radians(20.5), 0.02, 0.0), (13, np.radians(19.5), 0.02, 0.0), (14, np.radians(20.2), 0.02, 0.0),
                      (15, np.radians(20.1), 0.02, 0.0)])
        p.buf = type(p.buf)([b for b in p.buf if abs(np.degrees(b[1]) - 20) < 2], maxlen=32)
        self.assertTrue(p._confirmed(15))


class TestScenarioInjection(unittest.TestCase):
    def test_sign_flip_flips_estimate(self):
        import benchmark_v4 as B4
        from aurasync.v4.geo_pipeline import GeoPipeline
        g = GeoPipeline(); g.reset(np.array([0, 0, 0.1, 0.5]))
        sc = B4.Scenario(g, "sign_flip", np.random.default_rng(0))
        sc.reset(np.array([0, 0, 0.1, 0.5]))
        empty = np.zeros((4, 32, 96), np.float32)
        for k in range(B4.KICK_TICK + 61):
            sc.update(k, (k + 1) * 1e-3, np.zeros(4), empty, 0.0)
        self.assertLess(sc.estimate()[2] * 0.1, 0)              # angle sign flipped relative to the start


if __name__ == "__main__":
    unittest.main(verbosity=2)
