"""
Extended Kalman Filter (EKF) for 9-DoF Inertial Navigation & Dead Reckoning
Problem Statement: SIH26168 (Smart India Hackathon 2026)

State Vector (9D):
    x = [x, y, z, vx, vy, vz, roll, pitch, yaw]^T
    where:
        - [x, y, z]: Position in local NED frame (meters)
        - [vx, vy, vz]: Linear velocity in local NED frame (m/s)
        - [roll, pitch, yaw]: Euler angles (radians)

Inputs:
    - IMU: 3-axis Accelerometer (specific force), 3-axis Gyroscope (angular velocity)
    - Optional Updates: GNSS (position + velocity), Barometer (altitude), Magnetometer (heading)

Key Features:
    - Rigorous 9-DoF non-linear strapdown mechanization with Earth gravity compensation
    - First-order analytical Jacobian computation for state transition matrix F
    - Joseph-form covariance update for numerical stability and positive semi-definiteness
    - Innovation-based adaptive measurement noise estimation (adaptive_noise_estimation)
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict, Any
import sys
import os

# Ensure local imports work cleanly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import wrap_to_pi, euler_to_rotation_matrix, GRAVITY_STANDARD


class ExtendedKalmanFilter9DoF:
    """
    9-State Extended Kalman Filter for Dead Reckoning and Sensor Fusion.
    """

    def __init__(
        self,
        dt: float = 0.01,
        q_pos: float = 0.01,
        q_vel: float = 0.1,
        q_att: float = 0.005,
        r_gnss_pos: float = 1.5,
        r_gnss_vel: float = 0.1,
        r_baro: float = 0.5,
        r_mag: float = 0.05,
        adaptive_alpha: float = 0.95
    ) -> None:
        """
        Initialize the 9-DoF EKF.

        Args:
            dt: Time step in seconds
            q_pos: Process noise variance for position (m^2)
            q_vel: Process noise variance for velocity ((m/s)^2)
            q_att: Process noise variance for attitude (rad^2)
            r_gnss_pos: Measurement noise variance for GNSS position (m^2)
            r_gnss_vel: Measurement noise variance for GNSS velocity ((m/s)^2)
            r_baro: Measurement noise variance for barometer altitude (m^2)
            r_mag: Measurement noise variance for magnetometer yaw (rad^2)
            adaptive_alpha: Exponential smoothing factor for adaptive noise estimation [0, 1]
        """
        self.dt = float(dt)
        self.state_dim = 9
        
        # State vector: [x, y, z, vx, vy, vz, roll, pitch, yaw]^T
        self.x = np.zeros(self.state_dim, dtype=np.float64)
        
        # State estimation error covariance P (9x9)
        self.P = np.diag([
            1.0, 1.0, 1.0,       # Position variance
            0.5, 0.5, 0.5,       # Velocity variance
            0.05, 0.05, 0.1      # Attitude variance (rad^2)
        ]).astype(np.float64)
        
        # Process noise covariance Q (9x9)
        self.Q = np.diag([
            q_pos, q_pos, q_pos,
            q_vel, q_vel, q_vel,
            q_att, q_att, q_att
        ]).astype(np.float64)
        
        # Nominal Measurement covariances
        self.R_gnss = np.diag([
            r_gnss_pos, r_gnss_pos, r_gnss_pos * 2.0,
            r_gnss_vel, r_gnss_vel, r_gnss_vel * 2.0
        ]).astype(np.float64)
        
        self.R_baro = np.array([[r_baro]], dtype=np.float64)
        self.R_mag = np.array([[r_mag]], dtype=np.float64)
        
        # Adaptive noise tracking
        self.adaptive_alpha = adaptive_alpha
        self.R_gnss_adaptive = np.copy(self.R_gnss)
        self.last_innovation: Optional[np.ndarray] = None
        self.last_innovation_cov: Optional[np.ndarray] = None
        self.last_mahalanobis_dist: float = 0.0

    def get_state(self) -> np.ndarray:
        """Get copy of current 9D state vector."""
        return np.copy(self.x)

    def set_state(self, state: np.ndarray, cov: Optional[np.ndarray] = None) -> None:
        """
        Manually set or reset filter state and covariance.
        """
        self.x = np.array(state, dtype=np.float64).flatten()
        if cov is not None:
            self.P = np.array(cov, dtype=np.float64)

    def _state_transition_jacobian(
        self,
        x: np.ndarray,
        acc_body: np.ndarray,
        gyro_body: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Compute analytical Jacobian matrix F = d(f(x, u))/dx for the 9-DoF kinematics.
        """
        roll, pitch, yaw = x[6], x[7], x[8]
        ax, ay, az = acc_body[0], acc_body[1], acc_body[2]
        p, q, r = gyro_body[0], gyro_body[1], gyro_body[2]
        
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        tp = np.tan(pitch)
        sec_p = 1.0 / np.cos(pitch) if np.abs(np.cos(pitch)) > 1e-4 else 1e4
        
        # Continuous-time Jacobian F_c (9x9)
        F_c = np.zeros((9, 9), dtype=np.float64)
        
        # d(pos)/d(vel) = I_3x3
        F_c[0, 3] = 1.0
        F_c[1, 4] = 1.0
        F_c[2, 5] = 1.0
        
        # d(vel)/d(attitude) = d(R_bn * acc_body)/d(roll, pitch, yaw)
        # dv_n/d(roll)
        F_c[3, 6] = (cr * sp * cy + sr * sy) * ay + (-sr * sp * cy + cr * sy) * az
        F_c[4, 6] = (cr * sp * sy - sr * cy) * ay + (-sr * sp * sy - cr * cy) * az
        F_c[5, 6] = (cr * cp) * ay - (sr * cp) * az
        
        # dv_n/d(pitch)
        F_c[3, 7] = (-sp * cy) * ax + (sr * cp * cy) * ay + (cr * cp * cy) * az
        F_c[4, 7] = (-sp * sy) * ax + (sr * cp * sy) * ay + (cr * cp * sy) * az
        F_c[5, 7] = (-cp) * ax - (sr * sp) * ay - (cr * sp) * az
        
        # dv_n/d(yaw)
        F_c[3, 8] = (-cp * sy) * ax + (-sr * sp * sy - cr * cy) * ay + (-cr * sp * sy + sr * cy) * az
        F_c[4, 8] = (cp * cy) * ax + (sr * sp * cy - cr * sy) * ay + (cr * sp * cy + sr * sy) * az
        F_c[5, 8] = 0.0
        
        # d(attitude)/d(attitude) from Euler kinematic rates matrix
        # roll_rate = p + (q*sr + r*cr)*tp
        # pitch_rate = q*cr - r*sr
        # yaw_rate = (q*sr + r*cr)*sec_p
        F_c[6, 6] = (q * cr - r * sr) * tp
        F_c[6, 7] = (q * sr + r * cr) * (sec_p ** 2)
        
        F_c[7, 6] = -q * sr - r * cr
        F_c[7, 7] = 0.0
        
        F_c[8, 6] = (q * cr - r * sr) * sec_p
        F_c[8, 7] = (q * sr + r * cr) * (sec_p * tp)
        
        # Discretize via 1st order Taylor expansion: F_k = I + F_c * dt
        F = np.eye(9, dtype=np.float64) + F_c * dt
        return F

    def predict(
        self,
        acc_body: np.ndarray,
        gyro_body: np.ndarray,
        dt: Optional[float] = None
    ) -> np.ndarray:
        """
        Execute EKF Prediction Step using 6-axis IMU inputs.
        
        Args:
            acc_body: 3-axis accelerometer readings [ax, ay, az] (m/s^2)
            gyro_body: 3-axis gyroscope readings [p, q, r] (rad/s)
            dt: Optional time step override
            
        Returns:
            Predicted 9D state vector
        """
        dt = float(dt if dt is not None else self.dt)
        acc_body = np.asarray(acc_body, dtype=np.float64).flatten()
        gyro_body = np.asarray(gyro_body, dtype=np.float64).flatten()
        
        pos = self.x[0:3]
        vel = self.x[3:6]
        roll, pitch, yaw = self.x[6], self.x[7], self.x[8]
        
        # 1. Orientation Integration
        cr, sr = np.cos(roll), np.sin(roll)
        cp = np.cos(pitch)
        tp = np.tan(pitch)
        sec_p = 1.0 / cp if np.abs(cp) > 1e-4 else 1e4
        
        p, q, r = gyro_body[0], gyro_body[1], gyro_body[2]
        d_roll = p + (q * sr + r * cr) * tp
        d_pitch = q * cr - r * sr
        d_yaw = (q * sr + r * cr) * sec_p
        
        new_roll = wrap_to_pi(roll + d_roll * dt)
        new_pitch = wrap_to_pi(pitch + d_pitch * dt)
        new_yaw = wrap_to_pi(yaw + d_yaw * dt)
        
        # 2. Velocity & Position Integration
        R_bn = euler_to_rotation_matrix(new_roll, new_pitch, new_yaw)
        # Acceleration in NED navigation frame with gravity compensation
        g_ned = np.array([0.0, 0.0, GRAVITY_STANDARD], dtype=np.float64)
        acc_ned = R_bn @ acc_body + g_ned
        
        new_vel = vel + acc_ned * dt
        new_pos = pos + vel * dt + 0.5 * acc_ned * (dt ** 2)
        
        # Update state vector
        self.x[0:3] = new_pos
        self.x[3:6] = new_vel
        self.x[6] = new_roll
        self.x[7] = new_pitch
        self.x[8] = new_yaw
        
        # 3. Covariance Propagation: P = F * P * F^T + Q
        F = self._state_transition_jacobian(self.x, acc_body, gyro_body, dt)
        self.P = F @ self.P @ F.T + self.Q * dt
        # Ensure symmetry
        self.P = 0.5 * (self.P + self.P.T)
        
        return np.copy(self.x)

    def adaptive_noise_estimation(
        self,
        innovation: np.ndarray,
        H: np.ndarray,
        nominal_R: np.ndarray
    ) -> np.ndarray:
        """
        Estimate adaptive measurement covariance R using innovation residual sequence:
        R_k = alpha * R_{k-1} + (1 - alpha) * (y * y^T - H * P * H^T)
        """
        y = innovation.reshape(-1, 1)
        # Empirical residual covariance
        inst_R = y @ y.T - H @ self.P @ H.T
        
        # Guarantee diagonal positivity and smooth
        inst_R_diag = np.diag(inst_R)
        nominal_diag = np.diag(nominal_R)
        
        # Lower-bound by 50% nominal R to prevent covariance collapse
        clamped_diag = np.maximum(inst_R_diag, 0.5 * nominal_diag)
        updated_R = np.diag(
            self.adaptive_alpha * np.diag(self.R_gnss_adaptive) + (1.0 - self.adaptive_alpha) * clamped_diag
        )
        return updated_R

    def update_gnss(
        self,
        gnss_pos_ned: np.ndarray,
        gnss_vel_ned: np.ndarray,
        use_adaptive_noise: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Update state with GNSS 3D position and 3D velocity in NED frame.
        
        Args:
            gnss_pos_ned: [x, y, z] in meters
            gnss_vel_ned: [vx, vy, vz] in m/s
            use_adaptive_noise: Enable adaptive R estimation
            
        Returns:
            Tuple of (updated_state, mahalanobis_distance)
        """
        z = np.hstack([gnss_pos_ned, gnss_vel_ned]).astype(np.float64) # 6D
        
        # Measurement matrix H (6x9) mapping state to [pos, vel]
        H = np.zeros((6, 9), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 3:6] = np.eye(3)
        
        # Innovation y = z - Hx
        pred_z = H @ self.x
        innovation = z - pred_z
        
        # Measurement noise covariance
        if use_adaptive_noise:
            self.R_gnss_adaptive = self.adaptive_noise_estimation(innovation, H, self.R_gnss)
            R = self.R_gnss_adaptive
        else:
            R = self.R_gnss
            
        # Innovation covariance S = H P H^T + R
        S = H @ self.P @ H.T + R
        S = 0.5 * (S + S.T)
        
        # Compute Mahalanobis distance for anomaly checking
        try:
            S_inv = np.linalg.inv(S)
            mahalanobis = float(np.sqrt(innovation.T @ S_inv @ innovation))
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            mahalanobis = 0.0
            
        # Kalman Gain K = P H^T S^-1
        K = self.P @ H.T @ S_inv
        
        # State update
        self.x = self.x + K @ innovation
        self.x[6] = wrap_to_pi(self.x[6])
        self.x[7] = wrap_to_pi(self.x[7])
        self.x[8] = wrap_to_pi(self.x[8])
        
        # Covariance update via Joseph Form: P = (I - KH) P (I - KH)^T + K R K^T
        I_KH = np.eye(9) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        
        self.last_innovation = innovation
        self.last_innovation_cov = S
        self.last_mahalanobis_dist = mahalanobis
        
        return np.copy(self.x), mahalanobis

    def update_barometer(self, baro_z_ned: float) -> Tuple[np.ndarray, float]:
        """
        Update state with Barometer altitude (NED z-coordinate in meters).
        """
        z = np.array([baro_z_ned], dtype=np.float64)
        H = np.zeros((1, 9), dtype=np.float64)
        H[0, 2] = 1.0 # z position
        
        innovation = z - H @ self.x
        S = H @ self.P @ H.T + self.R_baro
        S_inv = 1.0 / S[0, 0]
        mahalanobis = float(np.abs(innovation[0]) * np.sqrt(S_inv))
        
        K = self.P @ H.T * S_inv
        self.x = self.x + (K @ innovation).flatten()
        
        I_KH = np.eye(9) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_baro @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        
        return np.copy(self.x), mahalanobis

    def update_magnetometer_yaw(self, mag_yaw: float) -> Tuple[np.ndarray, float]:
        """
        Update state with Magnetometer heading (yaw in radians).
        """
        z = np.array([mag_yaw], dtype=np.float64)
        H = np.zeros((1, 9), dtype=np.float64)
        H[0, 8] = 1.0 # yaw
        
        innovation = wrap_to_pi(z - H @ self.x)
        S = H @ self.P @ H.T + self.R_mag
        S_inv = 1.0 / S[0, 0]
        mahalanobis = float(np.abs(innovation[0]) * np.sqrt(S_inv))
        
        K = self.P @ H.T * S_inv
        self.x = self.x + (K @ innovation).flatten()
        self.x[8] = wrap_to_pi(self.x[8])
        
        I_KH = np.eye(9) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_mag @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        
        return np.copy(self.x), mahalanobis
