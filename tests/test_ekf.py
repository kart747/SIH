"""
Unit tests for 9-DoF Extended Kalman Filter (EKF)
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.sensor_fusion.ekf import ExtendedKalmanFilter9DoF
from src.utils import wrap_to_pi, GRAVITY_STANDARD


def test_ekf_initialization():
    ekf = ExtendedKalmanFilter9DoF(dt=0.01)
    assert ekf.x.shape == (9,)
    assert ekf.P.shape == (9, 9)
    assert np.all(np.diag(ekf.P) > 0)
    assert ekf.Q.shape == (9, 9)


def test_ekf_prediction_stationary():
    ekf = ExtendedKalmanFilter9DoF(dt=0.1)
    # Accelerometer measures -g in body frame when stationary upright
    acc_stationary = np.array([0.0, 0.0, -GRAVITY_STANDARD])
    gyro_stationary = np.array([0.0, 0.0, 0.0])
    
    # Predict 10 steps
    for _ in range(10):
        ekf.predict(acc_stationary, gyro_stationary)
        
    state = ekf.get_state()
    # Velocity and position should remain virtually zero
    assert np.allclose(state[0:3], np.zeros(3), atol=1e-3)
    assert np.allclose(state[3:6], np.zeros(3), atol=1e-3)


def test_ekf_prediction_linear_acceleration():
    ekf = ExtendedKalmanFilter9DoF(dt=0.1)
    # Accelerometer: 2 m/s^2 forward (X-axis) + gravity reaction
    acc_forward = np.array([2.0, 0.0, -GRAVITY_STANDARD])
    gyro_zero = np.array([0.0, 0.0, 0.0])
    
    for _ in range(10): # 1.0 second total
        ekf.predict(acc_forward, gyro_zero)
        
    state = ekf.get_state()
    # v = a * t = 2 * 1 = 2 m/s
    assert np.isclose(state[3], 2.0, atol=0.1)
    # x = 0.5 * a * t^2 = 1.0 m
    assert np.isclose(state[0], 1.0, atol=0.1)


def test_ekf_gnss_update_covariance_contraction():
    ekf = ExtendedKalmanFilter9DoF(dt=0.01)
    prior_cov_trace = np.trace(ekf.P[0:3, 0:3])
    
    # Provide GNSS observation
    gnss_pos = np.array([10.0, 5.0, -2.0])
    gnss_vel = np.array([5.0, 0.0, 0.0])
    
    updated_state, mahalanobis = ekf.update_gnss(gnss_pos, gnss_vel)
    post_cov_trace = np.trace(ekf.P[0:3, 0:3])
    
    # Uncertainty must strictly decrease after informative update
    assert post_cov_trace < prior_cov_trace
    # State should move towards observation
    assert np.all(updated_state[0:3] != 0.0)


def test_ekf_baro_and_mag_updates():
    ekf = ExtendedKalmanFilter9DoF(dt=0.01)
    # Update barometer multiple steps for convergence
    for _ in range(5):
        state_b, _ = ekf.update_barometer(baro_z_ned=-15.0)
    assert np.isclose(state_b[2], -15.0, atol=2.0)
    
    # Update magnetometer heading multiple steps for convergence
    for _ in range(5):
        state_m, _ = ekf.update_magnetometer_yaw(mag_yaw=np.pi / 4)
    assert np.isclose(state_m[8], np.pi / 4, atol=0.2)


def test_ekf_adaptive_noise():
    ekf = ExtendedKalmanFilter9DoF(dt=0.01, adaptive_alpha=0.8)
    # Large outlier innovation
    outlier_pos = np.array([100.0, 100.0, 100.0])
    outlier_vel = np.array([20.0, 20.0, 20.0])
    
    initial_R_diag = np.diag(ekf.R_gnss_adaptive).copy()
    ekf.update_gnss(outlier_pos, outlier_vel, use_adaptive_noise=True)
    adapted_R_diag = np.diag(ekf.R_gnss_adaptive)
    
    # Adaptive noise should inflate when encountering huge unexpected residuals
    assert np.all(adapted_R_diag >= initial_R_diag)
