"""
Unscented Kalman Filter (UKF) for 9-DoF Inertial Navigation & Dead Reckoning
Problem Statement: SIH26168 (Smart India Hackathon 2026)

State Vector (9D):
    x = [x, y, z, vx, vy, vz, roll, pitch, yaw]^T

Key Features:
    - Van der Merwe Scaled Unscented Transform for non-linear dynamics
    - Captures high-order moments during aggressive turns, acceleration, and non-linearities
    - Handles angular discontinuities with proper circular mean and covariance computation
    - Provides standalone high-performance implementation with optional filterpy compatibility
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import sys
import os
from scipy.linalg import cholesky

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import wrap_to_pi, euler_to_rotation_matrix, GRAVITY_STANDARD


class MerweScaledSigmaPoints:
    """
    Generates sigma points using the Van der Merwe scaled unscented transform.
    """

    def __init__(
        self,
        n: int = 9,
        alpha: float = 0.1,
        beta: float = 2.0,
        kappa: float = 0.0
    ) -> None:
        self.n = n
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)
        
        self.lambda_ = self.alpha ** 2 * (self.n + self.kappa) - self.n
        self.num_sigmas = 2 * self.n + 1
        
        # Calculate weights for mean (Wm) and covariance (Wc)
        self.Wm = np.zeros(self.num_sigmas, dtype=np.float64)
        self.Wc = np.zeros(self.num_sigmas, dtype=np.float64)
        
        c = self.n + self.lambda_
        self.Wm[0] = self.lambda_ / c
        self.Wc[0] = self.lambda_ / c + (1.0 - self.alpha ** 2 + self.beta)
        
        for i in range(1, self.num_sigmas):
            self.Wm[i] = 1.0 / (2.0 * c)
            self.Wc[i] = 1.0 / (2.0 * c)

    def sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Generate 2n+1 sigma points given mean state x and covariance P.
        """
        x = np.asarray(x, dtype=np.float64).flatten()
        P = 0.5 * (P + P.T)
        
        sigmas = np.zeros((self.num_sigmas, self.n), dtype=np.float64)
        sigmas[0] = x
        
        scaling = np.sqrt(self.n + self.lambda_)
        
        # Robust Cholesky decomposition with jitter if ill-conditioned
        try:
            U = cholesky((self.n + self.lambda_) * P, lower=False)
        except np.linalg.LinAlgError:
            jitter = np.eye(self.n) * 1e-6
            U = cholesky((self.n + self.lambda_) * (P + jitter), lower=False)
            
        for k in range(self.n):
            sigmas[k + 1] = x + U[k]
            sigmas[self.n + k + 1] = x - U[k]
            
        return sigmas


class UnscentedKalmanFilter9DoF:
    """
    9-State Unscented Kalman Filter for non-linear Dead Reckoning and Sensor Fusion.
    """

    def __init__(
        self,
        dt: float = 0.01,
        alpha: float = 0.1,
        beta: float = 2.0,
        kappa: float = 0.0,
        q_pos: float = 0.01,
        q_vel: float = 0.1,
        q_att: float = 0.005,
        r_gnss_pos: float = 1.5,
        r_gnss_vel: float = 0.1,
        r_baro: float = 0.5,
        r_mag: float = 0.05
    ) -> None:
        """
        Initialize the 9-DoF UKF.
        """
        self.dt = float(dt)
        self.state_dim = 9
        
        # Mean state vector: [x, y, z, vx, vy, vz, roll, pitch, yaw]^T
        self.x = np.zeros(self.state_dim, dtype=np.float64)
        
        # State covariance P (9x9)
        self.P = np.diag([
            1.0, 1.0, 1.0,
            0.5, 0.5, 0.5,
            0.05, 0.05, 0.1
        ]).astype(np.float64)
        
        # Process noise Q (9x9)
        self.Q = np.diag([
            q_pos, q_pos, q_pos,
            q_vel, q_vel, q_vel,
            q_att, q_att, q_att
        ]).astype(np.float64)
        
        # Measurement noise covariances
        self.R_gnss = np.diag([
            r_gnss_pos, r_gnss_pos, r_gnss_pos * 2.0,
            r_gnss_vel, r_gnss_vel, r_gnss_vel * 2.0
        ]).astype(np.float64)
        
        self.R_baro = np.array([[r_baro]], dtype=np.float64)
        self.R_mag = np.array([[r_mag]], dtype=np.float64)
        
        self.sigma_gen = MerweScaledSigmaPoints(n=self.state_dim, alpha=alpha, beta=beta, kappa=kappa)
        self.sigmas_f: Optional[np.ndarray] = None
        
        self.last_innovation: Optional[np.ndarray] = None
        self.last_innovation_cov: Optional[np.ndarray] = None
        self.last_mahalanobis_dist: float = 0.0

    def get_state(self) -> np.ndarray:
        return np.copy(self.x)

    def set_state(self, state: np.ndarray, cov: Optional[np.ndarray] = None) -> None:
        self.x = np.array(state, dtype=np.float64).flatten()
        if cov is not None:
            self.P = np.array(cov, dtype=np.float64)

    def compute_sigma_points(self) -> np.ndarray:
        """Generate current sigma points."""
        return self.sigma_gen.sigma_points(self.x, self.P)

    def _propagate_single_state(
        self,
        state: np.ndarray,
        acc_body: np.ndarray,
        gyro_body: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Non-linear propagation of a single state point through strapdown inertial equations.
        """
        pos = state[0:3]
        vel = state[3:6]
        roll, pitch, yaw = state[6], state[7], state[8]
        
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
        
        R_bn = euler_to_rotation_matrix(new_roll, new_pitch, new_yaw)
        g_ned = np.array([0.0, 0.0, GRAVITY_STANDARD], dtype=np.float64)
        acc_ned = R_bn @ acc_body + g_ned
        
        new_vel = vel + acc_ned * dt
        new_pos = pos + vel * dt + 0.5 * acc_ned * (dt ** 2)
        
        out = np.zeros(9, dtype=np.float64)
        out[0:3] = new_pos
        out[3:6] = new_vel
        out[6] = new_roll
        out[7] = new_pitch
        out[8] = new_yaw
        return out

    def predict(
        self,
        acc_body: np.ndarray,
        gyro_body: np.ndarray,
        dt: Optional[float] = None
    ) -> np.ndarray:
        """
        Execute UKF Prediction step using 6-axis IMU inputs through Unscented Transform.
        """
        dt = float(dt if dt is not None else self.dt)
        acc_body = np.asarray(acc_body, dtype=np.float64).flatten()
        gyro_body = np.asarray(gyro_body, dtype=np.float64).flatten()
        
        # 1. Compute sigma points
        sigmas = self.compute_sigma_points()
        num_sigmas = self.sigma_gen.num_sigmas
        
        # 2. Propagate each sigma point through non-linear dynamics
        self.sigmas_f = np.zeros_like(sigmas)
        for i in range(num_sigmas):
            self.sigmas_f[i] = self._propagate_single_state(sigmas[i], acc_body, gyro_body, dt)
            
        # 3. Compute predicted mean
        # Linear components (pos, vel)
        x_pred = np.zeros(9, dtype=np.float64)
        for i in range(num_sigmas):
            x_pred[0:6] += self.sigma_gen.Wm[i] * self.sigmas_f[i, 0:6]
            
        # Angular components (circular mean)
        sin_sum = np.zeros(3)
        cos_sum = np.zeros(3)
        for i in range(num_sigmas):
            sin_sum += self.sigma_gen.Wm[i] * np.sin(self.sigmas_f[i, 6:9])
            cos_sum += self.sigma_gen.Wm[i] * np.cos(self.sigmas_f[i, 6:9])
            
        x_pred[6:9] = np.arctan2(sin_sum, cos_sum)
        self.x = x_pred
        
        # 4. Compute predicted covariance P
        P_pred = np.zeros((9, 9), dtype=np.float64)
        for i in range(num_sigmas):
            dx = self.sigmas_f[i] - self.x
            # Wrap angles in error
            dx[6:9] = wrap_to_pi(dx[6:9])
            P_pred += self.sigma_gen.Wc[i] * np.outer(dx, dx)
            
        self.P = P_pred + self.Q * dt
        self.P = 0.5 * (self.P + self.P.T)
        
        return np.copy(self.x)

    def update_gnss(
        self,
        gnss_pos_ned: np.ndarray,
        gnss_vel_ned: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Execute UKF Measurement Update for 6D GNSS (3D pos, 3D vel).
        """
        z = np.hstack([gnss_pos_ned, gnss_vel_ned]).astype(np.float64)
        m_dim = 6
        num_sigmas = self.sigma_gen.num_sigmas
        
        # Re-compute sigma points or use prior predicted sigmas
        sigmas = self.compute_sigma_points()
        
        # Measurement function h(x) maps state directly to [pos, vel]
        sigmas_h = np.zeros((num_sigmas, m_dim), dtype=np.float64)
        for i in range(num_sigmas):
            sigmas_h[i] = sigmas[i, 0:6]
            
        # Predicted measurement mean
        z_pred = np.zeros(m_dim, dtype=np.float64)
        for i in range(num_sigmas):
            z_pred += self.sigma_gen.Wm[i] * sigmas_h[i]
            
        # Innovation covariance S and Cross covariance P_xz
        S = np.zeros((m_dim, m_dim), dtype=np.float64)
        P_xz = np.zeros((9, m_dim), dtype=np.float64)
        
        for i in range(num_sigmas):
            dz = sigmas_h[i] - z_pred
            dx = sigmas[i] - self.x
            dx[6:9] = wrap_to_pi(dx[6:9])
            
            S += self.sigma_gen.Wc[i] * np.outer(dz, dz)
            P_xz += self.sigma_gen.Wc[i] * np.outer(dx, dz)
            
        S += self.R_gnss
        S = 0.5 * (S + S.T)
        
        innovation = z - z_pred
        
        try:
            S_inv = np.linalg.inv(S)
            mahalanobis = float(np.sqrt(innovation.T @ S_inv @ innovation))
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            mahalanobis = 0.0
            
        # Kalman gain K = P_xz * S^-1
        K = P_xz @ S_inv
        
        # State update
        self.x = self.x + K @ innovation
        self.x[6:9] = wrap_to_pi(self.x[6:9])
        
        # Covariance update
        self.P = self.P - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        
        self.last_innovation = innovation
        self.last_innovation_cov = S
        self.last_mahalanobis_dist = mahalanobis
        
        return np.copy(self.x), mahalanobis

    def update_barometer(self, baro_z_ned: float) -> Tuple[np.ndarray, float]:
        """Update UKF with Barometer altitude."""
        z = np.array([baro_z_ned], dtype=np.float64)
        m_dim = 1
        num_sigmas = self.sigma_gen.num_sigmas
        
        sigmas = self.compute_sigma_points()
        sigmas_h = sigmas[:, 2:3] # z index
        
        z_pred = np.sum(self.sigma_gen.Wm[:, None] * sigmas_h, axis=0)
        
        S = np.zeros((1, 1), dtype=np.float64)
        P_xz = np.zeros((9, 1), dtype=np.float64)
        for i in range(num_sigmas):
            dz = sigmas_h[i] - z_pred
            dx = sigmas[i] - self.x
            dx[6:9] = wrap_to_pi(dx[6:9])
            S += self.sigma_gen.Wc[i] * np.outer(dz, dz)
            P_xz += self.sigma_gen.Wc[i] * np.outer(dx, dz)
            
        S += self.R_baro
        innovation = z - z_pred
        S_inv = 1.0 / S[0, 0]
        mahalanobis = float(np.abs(innovation[0]) * np.sqrt(S_inv))
        
        K = P_xz * S_inv
        self.x = self.x + (K @ innovation).flatten()
        self.x[6:9] = wrap_to_pi(self.x[6:9])
        
        self.P = self.P - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        
        return np.copy(self.x), mahalanobis

    def update_magnetometer_yaw(self, mag_yaw: float) -> Tuple[np.ndarray, float]:
        """Update UKF with Magnetometer heading."""
        z = np.array([mag_yaw], dtype=np.float64)
        m_dim = 1
        num_sigmas = self.sigma_gen.num_sigmas
        
        sigmas = self.compute_sigma_points()
        # Circular mean for heading prediction
        sin_sum = np.sum(self.sigma_gen.Wm * np.sin(sigmas[:, 8]))
        cos_sum = np.sum(self.sigma_gen.Wm * np.cos(sigmas[:, 8]))
        z_pred = np.array([np.arctan2(sin_sum, cos_sum)])
        
        S = np.zeros((1, 1), dtype=np.float64)
        P_xz = np.zeros((9, 1), dtype=np.float64)
        for i in range(num_sigmas):
            dz = wrap_to_pi(sigmas[i, 8] - z_pred[0])
            dx = sigmas[i] - self.x
            dx[6:9] = wrap_to_pi(dx[6:9])
            S += self.sigma_gen.Wc[i] * (dz ** 2)
            P_xz += self.sigma_gen.Wc[i] * dx[:, None] * dz
            
        S += self.R_mag
        innovation = wrap_to_pi(z - z_pred)
        S_inv = 1.0 / S[0, 0]
        mahalanobis = float(np.abs(innovation[0]) * np.sqrt(S_inv))
        
        K = P_xz * S_inv
        self.x = self.x + (K @ innovation).flatten()
        self.x[6:9] = wrap_to_pi(self.x[6:9])
        
        self.P = self.P - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        
        return np.copy(self.x), mahalanobis
