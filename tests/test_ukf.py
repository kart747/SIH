"""
Unit tests for 9-DoF Unscented Kalman Filter (UKF) and Adaptive Switcher
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.sensor_fusion.ukf import UnscentedKalmanFilter9DoF, MerweScaledSigmaPoints
from src.sensor_fusion.fusion import AdaptiveFusionEngine
from src.utils import wrap_to_pi, GRAVITY_STANDARD


def test_sigma_points_weights_and_shape():
    sigma_gen = MerweScaledSigmaPoints(n=9, alpha=0.1, beta=2.0, kappa=0.0)
    assert sigma_gen.num_sigmas == 19
    assert np.isclose(np.sum(sigma_gen.Wm), 1.0, atol=1e-5)
    
    x = np.zeros(9)
    P = np.eye(9) * 2.0
    sigmas = sigma_gen.sigma_points(x, P)
    assert sigmas.shape == (19, 9)
    assert np.allclose(sigmas[0], x)


def test_ukf_prediction_and_turn():
    ukf = UnscentedKalmanFilter9DoF(dt=0.1)
    acc = np.array([5.0, 0.0, -GRAVITY_STANDARD])
    gyro = np.array([0.0, 0.0, 0.5]) # turning yaw rate
    
    for _ in range(10): # 1.0s
        ukf.predict(acc, gyro)
        
    state = ukf.get_state()
    # Heading should have integrated
    assert state[8] > 0.3
    # Linear velocities should reflect curved motion
    assert state[3] > 0.0


def test_ukf_gnss_update():
    ukf = UnscentedKalmanFilter9DoF(dt=0.01)
    prior_cov = np.trace(ukf.P)
    
    pos_obs = np.array([25.0, -10.0, -5.0])
    vel_obs = np.array([12.0, 2.0, 0.0])
    
    updated_state, mahalanobis = ukf.update_gnss(pos_obs, vel_obs)
    post_cov = np.trace(ukf.P)
    
    assert post_cov < prior_cov
    assert np.all(updated_state[0:2] != 0.0)


def test_adaptive_fusion_engine_switching():
    engine = AdaptiveFusionEngine(dt=0.1, nonlinearity_threshold=0.4)
    assert engine.active_mode == "EKF"
    
    # 1. Straight cruising -> EKF
    acc_straight = np.array([1.0, 0.0, -GRAVITY_STANDARD])
    gyro_straight = np.array([0.0, 0.0, 0.01])
    
    _, mode1, score1 = engine.predict(acc_straight, gyro_straight)
    assert mode1 == "EKF"
    
    # 2. Aggressive cornering -> UKF
    acc_turn = np.array([2.0, 5.0, -GRAVITY_STANDARD])
    gyro_turn = np.array([0.0, 0.0, 0.8]) # High yaw rate
    
    _, mode2, score2 = engine.predict(acc_turn, gyro_turn)
    assert mode2 == "UKF"
    assert score2 > score1
