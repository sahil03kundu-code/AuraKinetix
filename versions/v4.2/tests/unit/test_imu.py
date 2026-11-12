"""Deterministic checks for the proprioceptive IMU + EKF baseline."""

import numpy as np

from aurasync.imu import DT, G, IMUPipeline


def test_reset_initializes_six_state_filter_and_delay_buffers():
    pipeline = IMUPipeline(delay_ticks=2)
    state = np.array([0.02, 0.0, 0.04, 0.0])
    pipeline.reset(state)
    assert pipeline.x.shape == (6,)
    assert pipeline.P.shape == (6, 6)
    assert len(pipeline.buf) == 3
    assert len(pipeline.cov_buf) == 3
    np.testing.assert_allclose(pipeline.estimate(), state)
    assert pipeline.covariance().shape == (4, 4)
    state[0] = 999.0
    assert pipeline.estimate()[0] != 999.0


def test_sensor_model_at_upright_rest_uses_gravity_and_bias():
    pipeline = IMUPipeline()
    x = np.array([0.1, 0.0, 0.0, 0.2, 0.03, 0.0])
    np.testing.assert_allclose(
        pipeline._h(x, 0.0), [0.1, 0.23, G, 0.0], atol=1e-12
    )


def test_sensor_model_uses_explicit_acceleration_state():
    pipeline = IMUPipeline()
    x = np.array([0.0, 0.0, np.pi / 2, 0.0, 0.0, 2.0])
    np.testing.assert_allclose(
        pipeline._h(x, 0.0), [0.0, 0.0, 2.0, -G], atol=1e-12
    )


def test_jacobian_matches_linear_function():
    matrix = np.array([[1.0, 2.0], [-3.0, 4.0], [0.5, 0.0]])
    value = np.array([0.2, -0.7])
    np.testing.assert_allclose(
        IMUPipeline._jac(lambda x: matrix @ x, value),
        matrix, atol=1e-9
    )


def test_dynamics_predictor_preserves_bias_and_auxiliary_acceleration():
    pipeline = IMUPipeline()
    x = np.array([0.0, 0.0, 0.02, 0.0, 0.015, 2.0])
    result = pipeline._f(x, 0.0)
    assert result.shape == (6,)
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result[4:], x[4:])
    assert DT == 0.001


def test_zero_noise_measurement_quantizes_encoder():
    pipeline = IMUPipeline(
        gyro_sigma=0.0, acc_sigma=0.0, bias_sigma=0.0, enc_res=1e-4
    )
    s = np.array([0.00016, 0.0, 0.0, 0.1])
    pipeline.reset(s)
    reading = pipeline.measure(s, 0.0, pipeline.p)
    assert reading.shape == (4,)
    assert np.isfinite(reading).all()
    np.testing.assert_allclose(reading[0], 0.0002, atol=1e-12)
    np.testing.assert_allclose(reading[1], 0.1, atol=1e-12)


def test_ekf_update_produces_finite_symmetric_covariance():
    pipeline = IMUPipeline(delay_ticks=0)
    s = np.array([0.01, 0.0, 0.02, 0.0])
    pipeline.reset(s)
    pipeline.update(k=0, t=0.0, s_true=s, surface=None, u=0.0)
    estimate = pipeline.estimate()
    covariance = pipeline.covariance()
    assert estimate.shape == (4,)
    assert np.isfinite(estimate).all()
    assert covariance.shape == (4, 4)
    assert np.isfinite(covariance).all()
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-10)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-10
