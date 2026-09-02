"""
Adaptive Sensor Fusion Engine (Dynamic EKF / UKF Switcher)
Problem Statement: SIH26168 (Smart India Hackathon 2026)

This module provides:
- Seamless adaptive switching between EKF (linear cruising, low compute) and UKF (non-linear turning, sharp maneuvers)
- Real-time motion non-linearity metric computation:
    1. Innovation residual statistics and Mahalanobis divergence
    2. Kinematic angular rate magnitudes (|gyro_z|, |gyro_rates|)
    3. Lateral centripetal acceleration (|v * yaw_rate|)
- Synchronized state & covariance transfer between filters
- Fused 9-DoF navigation state output (Position, Velocity, Orientation)
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict, Any, Literal
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.sensor_fusion.ekf import ExtendedKalmanFilter9DoF
from src.sensor_fusion.ukf import UnscentedKalmanFilter9DoF


class AdaptiveFusionEngine:
    """
    Adaptive Non-Linearity Aware Sensor Fusion Engine switching between EKF and UKF.
    """

    def __init__(
        self,
        dt: float = 0.01,
        nonlinearity_threshold: float = 0.45,
        turn_rate_threshold_rads: float = 0.15, # ~8.5 deg/s
        hysteresis_steps: int = 15,
        **kwargs: Any
    ) -> None:
        """
        Initialize the Adaptive Fusion Engine.
        
        Args:
            dt: Sampling time step in seconds
            nonlinearity_threshold: Normalized index [0, 1] above which UKF is selected
            turn_rate_threshold_rads: Angular velocity threshold for switching to UKF
            hysteresis_steps: Minimum timesteps to retain UKF mode before switching back to EKF
        """
        self.dt = dt
        self.nonlinearity_threshold = nonlinearity_threshold
        self.turn_rate_threshold = turn_rate_threshold_rads
        self.hysteresis_steps = hysteresis_steps
        
        # Instantiate both underlying filters
        self.ekf = ExtendedKalmanFilter9DoF(dt=dt, **kwargs)
        self.ukf = UnscentedKalmanFilter9DoF(dt=dt, **kwargs)
        
        # Active filter tracking
        self.active_mode: Literal["EKF", "UKF"] = "EKF"
        self.nonlinearity_score: float = 0.0
        self.ukf_hold_counter: int = 0
        self.switch_history: list = []

    def set_initial_state(self, state: np.ndarray, cov: Optional[np.ndarray] = None) -> None:
        """Set initial state and covariance for both filters."""
        self.ekf.set_state(state, cov)
        self.ukf.set_state(state, cov)

    def compute_nonlinearity_index(
        self,
        acc_body: np.ndarray,
        gyro_body: np.ndarray
    ) -> float:
        """
        Compute a normalized non-linearity metric [0.0, 1.0] from IMU dynamics.
        
        High angular rate or high lateral centripetal acceleration indicates
        significant non-linear kinematic coupling where UKF excels.
        """
        gyro_norm = float(np.linalg.norm(gyro_body))
        yaw_rate = float(np.abs(gyro_body[2]))
        
        # Current estimated speed
        current_state = self.ekf.get_state() if self.active_mode == "EKF" else self.ukf.get_state()
        speed = float(np.linalg.norm(current_state[3:6]))
        
        # Centripetal acceleration proxy: a_c = v * omega_z
        centripetal_acc = speed * yaw_rate
        
        # Innovation Mahalanobis penalty from previous step
        mahalanobis = self.ekf.last_mahalanobis_dist if self.active_mode == "EKF" else self.ukf.last_mahalanobis_dist
        
        # Weighted combination normalized via sigmoid-like response
        raw_score = (
            (yaw_rate / max(self.turn_rate_threshold, 1e-4)) * 0.4 +
            (centripetal_acc / 5.0) * 0.3 +
            (min(mahalanobis, 5.0) / 5.0) * 0.3
        )
        
        # Bound score in [0.0, 1.0]
        score = float(np.clip(raw_score, 0.0, 1.0))
        return score

    def predict(
        self,
        acc_body: np.ndarray,
        gyro_body: np.ndarray,
        dt: Optional[float] = None
    ) -> Tuple[np.ndarray, str, float]:
        """
        Run adaptive prediction step.
        
        Evaluates non-linearity, switches active filter if necessary,
        synchronizes states, and performs inertial prediction.
        
        Returns:
            Tuple of (predicted_state, active_mode_name, nonlinearity_score)
        """
        dt = float(dt if dt is not None else self.dt)
        score = self.compute_nonlinearity_index(acc_body, gyro_body)
        self.nonlinearity_score = score
        
        # Decision logic with hysteresis
        should_use_ukf = score >= self.nonlinearity_threshold or float(np.abs(gyro_body[2])) >= self.turn_rate_threshold
        
        if should_use_ukf:
            self.ukf_hold_counter = self.hysteresis_steps
            desired_mode = "UKF"
        else:
            if self.ukf_hold_counter > 0:
                self.ukf_hold_counter -= 1
                desired_mode = "UKF"
            else:
                desired_mode = "EKF"
                
        # Handle state synchronization on switchover
        if desired_mode != self.active_mode:
            if desired_mode == "UKF":
                # Transfer EKF state and covariance to UKF
                self.ukf.set_state(self.ekf.get_state(), self.ekf.P)
            else:
                # Transfer UKF state and covariance to EKF
                self.ekf.set_state(self.ukf.get_state(), self.ukf.P)
            self.active_mode = desired_mode
            
        # Execute prediction on active filter and background-sync shadow filter
        if self.active_mode == "UKF":
            pred_state = self.ukf.predict(acc_body, gyro_body, dt)
            # Sync to EKF
            self.ekf.set_state(self.ukf.get_state(), self.ukf.P)
        else:
            pred_state = self.ekf.predict(acc_body, gyro_body, dt)
            # Sync to UKF
            self.ukf.set_state(self.ekf.get_state(), self.ekf.P)
            
        return pred_state, self.active_mode, self.nonlinearity_score

    def update_gnss(
        self,
        gnss_pos_ned: np.ndarray,
        gnss_vel_ned: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Execute GNSS update on the active filter and synchronize.
        """
        if self.active_mode == "UKF":
            updated_state, mahalanobis = self.ukf.update_gnss(gnss_pos_ned, gnss_vel_ned)
            self.ekf.set_state(self.ukf.get_state(), self.ukf.P)
        else:
            updated_state, mahalanobis = self.ekf.update_gnss(gnss_pos_ned, gnss_vel_ned)
            self.ukf.set_state(self.ekf.get_state(), self.ekf.P)
            
        return updated_state, mahalanobis

    def update_barometer(self, baro_z_ned: float) -> Tuple[np.ndarray, float]:
        """Execute Barometer altitude update."""
        if self.active_mode == "UKF":
            updated_state, mahalanobis = self.ukf.update_barometer(baro_z_ned)
            self.ekf.set_state(self.ukf.get_state(), self.ukf.P)
        else:
            updated_state, mahalanobis = self.ekf.update_barometer(baro_z_ned)
            self.ukf.set_state(self.ekf.get_state(), self.ekf.P)
        return updated_state, mahalanobis

    def update_magnetometer_yaw(self, mag_yaw: float) -> Tuple[np.ndarray, float]:
        """Execute Magnetometer heading update."""
        if self.active_mode == "UKF":
            updated_state, mahalanobis = self.ukf.update_magnetometer_yaw(mag_yaw)
            self.ekf.set_state(self.ukf.get_state(), self.ukf.P)
        else:
            updated_state, mahalanobis = self.ekf.update_magnetometer_yaw(mag_yaw)
            self.ukf.set_state(self.ekf.get_state(), self.ekf.P)
        return updated_state, mahalanobis

    def get_state(self) -> np.ndarray:
        """Get current 9D navigation state."""
        return self.ukf.get_state() if self.active_mode == "UKF" else self.ekf.get_state()

    def get_covariance(self) -> np.ndarray:
        """Get current 9x9 state estimation error covariance."""
        return np.copy(self.ukf.P if self.active_mode == "UKF" else self.ekf.P)
