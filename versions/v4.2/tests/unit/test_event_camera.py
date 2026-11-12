"""Unit tests for event-camera rendering and simulation."""

import unittest
import numpy as np

from aurasync.event_camera import SceneRenderer


class TestSceneRenderer(unittest.TestCase):

    def test_nominal_render_shape_and_brightness(self):
        renderer = SceneRenderer(H=32, W=96)
        state = np.zeros(4)

        image = renderer.render(state, l=0.25)

        self.assertEqual(image.shape, (32, 96))
        self.assertTrue(np.all(np.isfinite(image)))
        self.assertGreaterEqual(float(image.min()), renderer.bg)
        self.assertLessEqual(float(image.max()), renderer.fg)
        self.assertGreater(float(image.max()), renderer.bg)

    def test_nominal_render_is_deterministic(self):
        renderer = SceneRenderer(H=32, W=96)
        state = np.array([0.1, 0.0, 0.2, 0.0])

        image1 = renderer.render(state, l=0.25)
        image2 = renderer.render(state, l=0.25)

        np.testing.assert_array_equal(image1, image2)

    def test_cart_position_changes_rendering(self):
        renderer = SceneRenderer(H=32, W=96)

        center = renderer.render(np.zeros(4), l=0.25)
        shifted_state = np.array([0.3, 0.0, 0.0, 0.0])
        shifted = renderer.render(shifted_state, l=0.25)

        self.assertFalse(np.array_equal(center, shifted))


if __name__ == "__main__":
    unittest.main()

class TestTimeSurfaceEncoder(unittest.TestCase):

    def test_empty_surface_is_zero(self):
        from aurasync.event_camera import TimeSurfaceEncoder

        encoder = TimeSurfaceEncoder(H=4, W=6)
        surface = encoder.tensor(t=1.0)

        self.assertEqual(surface.shape, (3, 4, 6))
        self.assertEqual(surface.dtype, np.float32)
        np.testing.assert_array_equal(
            surface, np.zeros((3, 4, 6), dtype=np.float32)
        )

    def test_recent_event_and_temporal_decay(self):
        from aurasync.event_camera import TimeSurfaceEncoder

        encoder = TimeSurfaceEncoder(H=4, W=6)

        pos = np.full((4, 6), np.nan)
        neg = np.full((4, 6), np.nan)
        pos[1, 2] = 1.0

        encoder.update(pos, neg)

        recent = encoder.tensor(t=1.0)
        later = encoder.tensor(t=1.01)

        self.assertAlmostEqual(float(recent[0, 1, 2]), 1.0)
        self.assertAlmostEqual(float(recent[2, 1, 2]), 1.0)
        self.assertEqual(float(recent[1, 1, 2]), 0.0)

        self.assertLess(later[0, 1, 2], recent[0, 1, 2])
        self.assertLess(later[2, 1, 2], recent[2, 1, 2])

    def test_memory_size_remains_fixed(self):
        from aurasync.event_camera import TimeSurfaceEncoder

        encoder = TimeSurfaceEncoder(H=4, W=6)
        initial_bytes = encoder.nbytes

        pos = np.full((4, 6), 1.0)
        neg = np.full((4, 6), np.nan)

        for _ in range(10):
            encoder.update(pos, neg)

        self.assertEqual(encoder.nbytes, initial_bytes)
        self.assertEqual(initial_bytes, 2 * 4 * 6 * 8)

class TestEventCamera(unittest.TestCase):

    def test_static_image_generates_no_events(self):
        from aurasync.event_camera import EventCamera

        camera = EventCamera(
            H=4, W=6, C=0.18, C_sigma=0,
            noise_rate_hz=0, seed=42
        )

        image = np.full((4, 6), 0.5)
        camera.reset(image)

        pos, neg, count = camera.step(image, t0=0, dt=0.001)

        self.assertEqual(count, 0)
        self.assertTrue(np.isnan(pos).all())
        self.assertTrue(np.isnan(neg).all())
        self.assertEqual(camera.n_events, 0)

    def test_brightness_increase_generates_positive_events(self):
        from aurasync.event_camera import EventCamera

        camera = EventCamera(
            H=4, W=6, C=0.18, C_sigma=0,
            noise_rate_hz=0, seed=42
        )

        camera.reset(np.full((4, 6), 0.2))

        pos, neg, count = camera.step(
            np.full((4, 6), 0.8),
            t0=1.0,
            dt=0.001
        )

        self.assertGreater(count, 0)
        self.assertTrue(np.isfinite(pos).all())
        self.assertTrue(np.isnan(neg).all())
        self.assertTrue(np.all(pos >= 1.0))
        self.assertTrue(np.all(pos <= 1.001))
        self.assertEqual(camera.n_events, count)

    def test_brightness_decrease_generates_negative_events(self):
        from aurasync.event_camera import EventCamera

        camera = EventCamera(
            H=4, W=6, C=0.18, C_sigma=0,
            noise_rate_hz=0, seed=42
        )

        camera.reset(np.full((4, 6), 0.8))

        pos, neg, count = camera.step(
            np.full((4, 6), 0.2),
            t0=2.0,
            dt=0.001
        )

        self.assertGreater(count, 0)
        self.assertTrue(np.isnan(pos).all())
        self.assertTrue(np.isfinite(neg).all())
        self.assertTrue(np.all(neg >= 2.0))
        self.assertTrue(np.all(neg <= 2.001))
        self.assertEqual(camera.n_events, count)

    def test_full_event_drop_suppresses_output(self):
        from aurasync.event_camera import EventCamera

        camera = EventCamera(
            H=4, W=6,
            C=0.18,
            C_sigma=0,
            noise_rate_hz=0,
            drop_prob=1.0,
            seed=42,
        )

        camera.reset(np.full((4, 6), 0.2))

        pos, neg, count = camera.step(
            np.full((4, 6), 0.8),
            t0=1.0,
            dt=0.001,
        )

        self.assertEqual(count, 0)
        self.assertTrue(np.isnan(pos).all())
        self.assertTrue(np.isnan(neg).all())
        self.assertEqual(camera.n_events, 0)

    def test_partial_event_drop_count_is_consistent(self):
        from aurasync.event_camera import EventCamera

        camera = EventCamera(
            H=8, W=8,
            C=0.18,
            C_sigma=0,
            noise_rate_hz=0,
            drop_prob=0.5,
            seed=42,
        )

        camera.reset(np.full((8, 8), 0.2))

        pos, neg, count = camera.step(
            np.full((8, 8), 0.8),
            t0=1.0,
            dt=0.001,
        )

        # Each surviving pixel represents multiple crossings.
        crossings_per_pixel = int(
            np.floor(
                np.log((0.8 + camera.eps) /
                       (0.2 + camera.eps)) / 0.18
            )
        )

        delivered_pixels = np.isfinite(pos).sum()

        self.assertEqual(
            count, delivered_pixels * crossings_per_pixel
        )
        self.assertTrue(np.isnan(neg).all())
        self.assertGreater(count, 0)
        self.assertLess(count, 8 * 8 * crossings_per_pixel)
        self.assertEqual(camera.n_events, count)

    def test_background_noise_generates_events(self):
        from aurasync.event_camera import EventCamera

        camera = EventCamera(
            H=4, W=6,
            C=0.18,
            C_sigma=0,
            noise_rate_hz=1000,
            seed=42,
        )

        image = np.full((4, 6), 0.5)
        camera.reset(image)

        pos, neg, count = camera.step(
            image,
            t0=1.0,
            dt=0.001,
        )

        # Noise probability is 1 for every pixel.
        self.assertEqual(count, 24)
        self.assertEqual(
            np.isfinite(pos).sum() + np.isfinite(neg).sum(),
            24,
        )
        self.assertTrue(
            np.all(pos[np.isfinite(pos)] >= 1.0)
        )
        self.assertTrue(
            np.all(neg[np.isfinite(neg)] <= 1.001)
        )
        self.assertEqual(camera.n_events, count)

    def test_record_mode_stores_event_coordinates(self):
        from aurasync.event_camera import EventCamera

        camera = EventCamera(
            H=4, W=6,
            C=0.18,
            C_sigma=0,
            noise_rate_hz=0,
            record=True,
            seed=42,
        )

        camera.reset(np.full((4, 6), 0.2))
        camera.step(
            np.full((4, 6), 0.8),
            t0=1.0,
            dt=0.001,
        )

        self.assertEqual(len(camera.events), 24)

        for x, y, timestamp, polarity in camera.events:
            self.assertTrue(0 <= x < 6)
            self.assertTrue(0 <= y < 4)
            self.assertTrue(1.0 <= timestamp <= 1.001)
            self.assertEqual(polarity, 1)

    def test_slow_photoreceptor_filters_brightness_change(self):
        from aurasync.event_camera import EventCamera

        camera = EventCamera(
            H=4, W=6,
            C=0.18,
            C_sigma=0,
            noise_rate_hz=0,
            tau=0.01,
            seed=42,
        )

        initial = np.full((4, 6), 0.2)
        changed = np.full((4, 6), 0.8)

        camera.reset(initial)
        initial_log = camera.L_prev.copy()
        target_log = np.log(changed + camera.eps)

        camera.step(changed, t0=1.0, dt=0.001)

        expected = initial_log + (
            target_log - initial_log
        ) * 0.1

        np.testing.assert_allclose(
            camera.L_filt, expected
        )
        np.testing.assert_allclose(
            camera.L_prev, expected
        )

class TestConditionalRendering(unittest.TestCase):

    def test_nominal_conditions_match_default(self):
        from aurasync.conditions import NOMINAL

        state = np.zeros(4)

        default = SceneRenderer()
        nominal = SceneRenderer(cond=NOMINAL)

        np.testing.assert_array_equal(
            default.render(state, 0.25),
            nominal.render(state, 0.25),
        )

    def test_brightness_and_hdr_conditions(self):
        from aurasync.conditions import SensorConditions

        state = np.zeros(4)
        nominal = SceneRenderer()

        conditions = SensorConditions(
            name="custom_hdr",
            brightness=0.5,
            hdr_gain=4.0,
        )

        modified = SceneRenderer(cond=conditions)

        original = nominal.render(state, 0.25)
        changed = modified.render(state, 0.25)

        self.assertEqual(changed.shape, original.shape)
        self.assertTrue(np.all(np.isfinite(changed)))
        self.assertFalse(np.array_equal(original, changed))

    def test_occlusion_changes_image_only_during_window(self):
        from aurasync.conditions import SensorConditions

        conditions = SensorConditions(
            name="test_occlusion",
            occlusion=(0.2, 0.3),
        )

        renderer = SceneRenderer(cond=conditions)
        state = np.zeros(4)

        before = renderer.render(state, 0.25, t=0.1)
        during = renderer.render(state, 0.25, t=0.25)
        after = renderer.render(state, 0.25, t=0.4)

        np.testing.assert_array_equal(before, after)
        self.assertFalse(np.array_equal(before, during))

    def test_camera_miscalibration_changes_world_mapping(self):
        from aurasync.conditions import SensorConditions

        nominal = SceneRenderer()

        conditions = SensorConditions(
            name="miscalibrated",
            x_offset_m=0.05,
            scale=1.1,
        )

        modified = SceneRenderer(cond=conditions)

        self.assertFalse(
            np.array_equal(nominal.wx, modified.wx)
        )
        self.assertFalse(
            np.array_equal(nominal.wy, modified.wy)
        )
