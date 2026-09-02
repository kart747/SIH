"""
AI-ML Intelligent Dead Reckoning System - Utilities & Data Processing
Problem Statement: SIH26168 (Smart India Hackathon 2026)

This module provides:
- Geodetic (WGS84) to local Cartesian (NED/ENU) coordinate transformations (vectorized & scalar)
- Quaternion and Euler angle conversions
- High-fidelity 6-DoF synthetic vehicle trajectory generator (Kinematic Bicycle Model)
- Sensor simulation (IMU with bias/drift, Barometer, Magnetometer, GNSS)
- GNSS Outage and Spoofing Attack injectors
- IO-VNBD dataset loader with automatic synthetic fallback
- Performance evaluation metrics (RMSE, CEP50, CEP95, Max Error)
"""

from __future__ import annotations
import os
import math
from typing import Dict, Tuple, Optional, Union, List, Any
import numpy as np


# ---------------------------------------------------------------------------
# Geodetic & Coordinate Transformations (WGS84 <-> Local Cartesian NED)
# ---------------------------------------------------------------------------

WGS84_A = 6378137.0          # Semi-major axis (meters)
WGS84_F = 1.0 / 298.257223563 # Flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F) # Semi-minor axis
WGS84_E2 = 2.0 * WGS84_F - WGS84_F ** 2 # First eccentricity squared
GRAVITY_STANDARD = 9.80665    # Standard gravity m/s^2


def wrap_to_pi(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Wrap angle in radians to the interval [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def geodetic_to_ecef(lat: Union[float, np.ndarray], lon: Union[float, np.ndarray], alt: Union[float, np.ndarray]) -> Tuple[Any, Any, Any]:
    """Convert geodetic coordinates (lat, lon, alt) to ECEF (X, Y, Z). Vectorized."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
    
    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1.0 - WGS84_E2) + alt) * sin_lat
    
    if np.isscalar(lat):
        return float(x), float(y), float(z)
    return x, y, z


def ecef_to_geodetic(x: Union[float, np.ndarray], y: Union[float, np.ndarray], z: Union[float, np.ndarray]) -> Tuple[Any, Any, Any]:
    """Convert ECEF coordinates (X, Y, Z) to geodetic (lat, lon, alt). Vectorized Bowring algorithm."""
    p = np.sqrt(x**2 + y**2)
    
    theta = np.arctan2(z * WGS84_A, np.maximum(p * WGS84_B, 1e-12))
    e_prime2 = (WGS84_A**2 - WGS84_B**2) / (WGS84_B**2)
    
    lat_rad = np.arctan2(
        z + e_prime2 * WGS84_B * (np.sin(theta)**3),
        np.maximum(p - WGS84_E2 * WGS84_A * (np.cos(theta)**3), 1e-12)
    )
    lon_rad = np.arctan2(y, x)
    
    sin_lat = np.sin(lat_rad)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    alt = p / np.cos(lat_rad) - N
    
    lat_deg = np.degrees(lat_rad)
    lon_deg = np.degrees(lon_rad)
    
    if np.isscalar(x):
        return float(lat_deg), float(lon_deg), float(alt)
    return lat_deg, lon_deg, alt


def geodetic_to_ned(
    lat: Union[float, np.ndarray], lon: Union[float, np.ndarray], alt: Union[float, np.ndarray],
    lat0: float, lon0: float, alt0: float
) -> Tuple[Any, Any, Any]:
    """Convert geodetic coordinates to local North-East-Down (NED). Vectorized."""
    x, y, z = geodetic_to_ecef(lat, lon, alt)
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, alt0)
    
    dx = x - x0
    dy = y - y0
    dz = z - z0
    
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    
    sin_lat = np.sin(lat0_rad)
    cos_lat = np.cos(lat0_rad)
    sin_lon = np.sin(lon0_rad)
    cos_lon = np.cos(lon0_rad)
    
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    e = -sin_lon * dx + cos_lon * dy
    d = -cos_lat * cos_lon * dx - cos_lat * sin_lon * dy - sin_lat * dz
    
    if np.isscalar(lat):
        return float(n), float(e), float(d)
    return n, e, d


def ned_to_geodetic(
    n: Union[float, np.ndarray], e: Union[float, np.ndarray], d: Union[float, np.ndarray],
    lat0: float, lon0: float, alt0: float
) -> Tuple[Any, Any, Any]:
    """Convert local North-East-Down (NED) coordinates to geodetic (WGS84). Vectorized."""
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, alt0)
    
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    
    sin_lat = np.sin(lat0_rad)
    cos_lat = np.cos(lat0_rad)
    sin_lon = np.sin(lon0_rad)
    cos_lon = np.cos(lon0_rad)
    
    dx = -sin_lat * cos_lon * n - sin_lon * e - cos_lat * cos_lon * d
    dy = -sin_lat * sin_lon * n + cos_lon * e - cos_lat * sin_lon * d
    dz = cos_lat * n - sin_lat * d
    
    x = x0 + dx
    y = y0 + dy
    z = z0 + dz
    
    return ecef_to_geodetic(x, y, z)


# ---------------------------------------------------------------------------
# Orientation & Rotation Matrix Helpers
# ---------------------------------------------------------------------------

def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Compute rotation matrix from body frame to navigation (NED) frame: C_b^n."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    R = np.array([
        [cp * cy,  sr * sp * cy - cr * sy,  cr * sp * cy + sr * sy],
        [cp * sy,  sr * sp * sy + cr * cy,  cr * sp * sy - sr * cy],
        [-sp,      sr * cp,                 cr * cp]
    ], dtype=np.float64)
    return R


def euler_rates_to_body_rates(roll: float, pitch: float, roll_rate: float, pitch_rate: float, yaw_rate: float) -> np.ndarray:
    """Convert Euler angle rates to body angular velocity (p, q, r)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp = np.cos(pitch)
    
    p = roll_rate - np.sin(pitch) * yaw_rate
    q = cr * pitch_rate + sr * cp * yaw_rate
    r = -sr * pitch_rate + cr * cp * yaw_rate
    return np.array([p, q, r], dtype=np.float64)


# ---------------------------------------------------------------------------
# High-Fidelity Synthetic Trajectory & Sensor Data Generation
# ---------------------------------------------------------------------------

def generate_synthetic_vehicle_trajectory(
    duration_sec: float = 600.0,
    dt: float = 0.01,
    origin_lat: float = 28.6139,
    origin_lon: float = 77.2090,
    origin_alt: float = 216.0,
    outage_start: float = 180.0,
    outage_duration: float = 120.0,
    spoof_start: float = 420.0,
    spoof_duration: float = 60.0,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generate a 10-minute realistic ground vehicle trajectory with 6-DoF dynamics,
    IMU, Barometer, Magnetometer, and GNSS observations with simulated Outages
    and Spoofing attacks. Fully vectorized for high performance.
    """
    np.random.seed(seed)
    n_samples = int(duration_sec / dt)
    t = np.linspace(0.0, duration_sec, n_samples)
    
    v_base = 12.0 # ~43 km/h
    v_forward = v_base + 4.0 * np.sin(2.0 * np.pi * t / 120.0) + 2.0 * np.sin(2.0 * np.pi * t / 45.0)
    v_forward = np.clip(v_forward, 2.0, 22.0)
    
    yaw_rate = np.zeros(n_samples)
    # Turn 1: right turn at 60s
    m1 = (t >= 60.0) & (t <= 70.0)
    yaw_rate[m1] = np.pi / 20.0
    # Turn 2: left turn at 140s
    m2 = (t >= 140.0) & (t <= 150.0)
    yaw_rate[m2] = -np.pi / 20.0
    # Turn 3: Roundabout (360 deg) at 220s (during GNSS outage)
    m3 = (t >= 220.0) & (t <= 250.0)
    yaw_rate[m3] = (2.0 * np.pi) / 30.0
    # Turn 4: S-curve slalom at 350s
    m4 = (t >= 340.0) & (t <= 380.0)
    yaw_rate[m4] = (np.pi / 10.0) * np.sin(2.0 * np.pi * (t[m4] - 340.0) / 20.0)
    # Turn 5: 90 deg turn during spoofing attack at 440s
    m5 = (t >= 440.0) & (t <= 450.0)
    yaw_rate[m5] = np.pi / 20.0
    # Turn 6: Gentle turn at 520s
    m6 = (t >= 520.0) & (t <= 540.0)
    yaw_rate[m6] = -np.pi / 40.0
    
    yaw = wrap_to_pi(np.cumsum(yaw_rate * dt))
    
    z_ned = - (15.0 * np.sin(2.0 * np.pi * t / 250.0) + 5.0 * np.sin(2.0 * np.pi * t / 80.0))
    vz_ned = np.gradient(z_ned, dt)
    pitch = -np.arctan2(vz_ned, np.maximum(v_forward, 1e-3))
    
    a_lat = v_forward * yaw_rate
    roll = np.clip(a_lat / 9.81 * 0.15, -np.radians(12.0), np.radians(12.0))
    
    vx_ned = v_forward * np.cos(yaw) * np.cos(pitch)
    vy_ned = v_forward * np.sin(yaw) * np.cos(pitch)
    
    x_ned = np.cumsum(vx_ned * dt)
    y_ned = np.cumsum(vy_ned * dt)
    
    ax_ned = np.gradient(vx_ned, dt)
    ay_ned = np.gradient(vy_ned, dt)
    az_ned = np.gradient(vz_ned, dt)
    
    # Vectorized body frame transformation
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    g_ned = np.array([0.0, 0.0, GRAVITY_STANDARD])
    fn_x = ax_ned - g_ned[0]
    fn_y = ay_ned - g_ned[1]
    fn_z = az_ned - g_ned[2]
    
    # Body acceleration f_b = R_nb * f_n
    # R_nb = R_bn.T
    fb_x = (cp * cy) * fn_x + (cp * sy) * fn_y - sp * fn_z
    fb_y = (sr * sp * cy - cr * sy) * fn_x + (sr * sp * sy + cr * cy) * fn_y + (sr * cp) * fn_z
    fb_z = (cr * sp * cy + sr * sy) * fn_x + (cr * sp * sy - sr * cy) * fn_y + (cr * cp) * fn_z
    acc_body_true = np.column_stack([fb_x, fb_y, fb_z])
    
    d_roll = np.gradient(roll, dt)
    d_pitch = np.gradient(pitch, dt)
    
    p = d_roll - np.sin(pitch) * yaw_rate
    q = cr * d_pitch + sr * cp * yaw_rate
    r = -sr * d_pitch + cr * cp * yaw_rate
    gyro_body_true = np.column_stack([p, q, r])
    
    # Vectorized geodetic conversion
    gt_lat, gt_lon, gt_alt = ned_to_geodetic(x_ned, y_ned, z_ned, origin_lat, origin_lon, origin_alt)

    # -----------------------------------------------------------------------
    # Sensor Noise Models
    # -----------------------------------------------------------------------
    acc_noise_std = 0.08
    gyro_noise_std = 0.002
    acc_bias_drift = np.cumsum(np.random.normal(0, 1e-4, (n_samples, 3)), axis=0) + np.array([0.02, -0.015, 0.03])
    gyro_bias_drift = np.cumsum(np.random.normal(0, 1e-6, (n_samples, 3)), axis=0) + np.array([0.0005, -0.0003, 0.0008])
    
    imu_acc = acc_body_true + acc_bias_drift + np.random.normal(0, acc_noise_std, (n_samples, 3))
    imu_gyro = gyro_body_true + gyro_bias_drift + np.random.normal(0, gyro_noise_std, (n_samples, 3))
    
    baro_noise_std = 0.4
    baro_alt = (-z_ned) + origin_alt + np.random.normal(0, baro_noise_std, n_samples)
    
    # Magnetometer
    B_ned = np.array([28.0, 3.0, 20.0])
    mag_bx = (cp * cy) * B_ned[0] + (cp * sy) * B_ned[1] - sp * B_ned[2]
    mag_by = (sr * sp * cy - cr * sy) * B_ned[0] + (sr * sp * sy + cr * cy) * B_ned[1] + (sr * cp) * B_ned[2]
    mag_bz = (cr * sp * cy + sr * sy) * B_ned[0] + (cr * sp * sy - sr * cy) * B_ned[1] + (cr * cp) * B_ned[2]
    mag_body = np.column_stack([mag_bx, mag_by, mag_bz]) + np.random.normal(0, 0.6, (n_samples, 3))

    # GNSS
    gnss_noise_pos = 1.2
    gnss_noise_vel = 0.08
    
    n_err = np.random.normal(0, gnss_noise_pos, n_samples)
    e_err = np.random.normal(0, gnss_noise_pos, n_samples)
    d_err = np.random.normal(0, gnss_noise_pos * 1.5, n_samples)
    
    gnss_lat, gnss_lon, gnss_alt = ned_to_geodetic(
        x_ned + n_err, y_ned + e_err, z_ned + d_err, origin_lat, origin_lon, origin_alt
    )
    gnss_vx = vx_ned + np.random.normal(0, gnss_noise_vel, n_samples)
    gnss_vy = vy_ned + np.random.normal(0, gnss_noise_vel, n_samples)
    gnss_vz = vz_ned + np.random.normal(0, gnss_noise_vel * 1.5, n_samples)
    
    cn0_series = np.random.normal(42.0, 3.0, n_samples)
    clock_drift_series = np.random.normal(1e-8, 2e-9, n_samples)
    multipath_indicator = np.random.exponential(0.5, n_samples)
    
    is_available = np.ones(n_samples, dtype=bool)
    is_spoofed_gt = np.zeros(n_samples, dtype=bool)
    
    outage_mask = (t >= outage_start) & (t < (outage_start + outage_duration))
    spoof_mask = (t >= spoof_start) & (t < (spoof_start + spoof_duration))
    
    # Apply Outage
    is_available[outage_mask] = False
    gnss_lat[outage_mask] = np.nan
    gnss_lon[outage_mask] = np.nan
    gnss_alt[outage_mask] = np.nan
    gnss_vx[outage_mask] = np.nan
    gnss_vy[outage_mask] = np.nan
    gnss_vz[outage_mask] = np.nan
    cn0_series[outage_mask] = 0.0
    
    # Apply Spoofing Attack
    is_spoofed_gt[spoof_mask] = True
    spoof_indices = np.where(spoof_mask)[0]
    for idx_rel, idx in enumerate(spoof_indices):
        spoof_t = idx_rel * dt
        diverge_n = 35.0 * (spoof_t / spoof_duration) ** 1.5 + 20.0
        diverge_e = -45.0 * (spoof_t / spoof_duration) ** 1.5 - 25.0
        diverge_d = 10.0 * np.sin(spoof_t / 10.0)
        
        sla, slo, sal = ned_to_geodetic(
            x_ned[idx] + diverge_n, y_ned[idx] + diverge_e, z_ned[idx] + diverge_d,
            origin_lat, origin_lon, origin_alt
        )
        gnss_lat[idx] = sla
        gnss_lon[idx] = slo
        gnss_alt[idx] = sal
        
        gnss_vx[idx] += 8.5 * np.cos(spoof_t / 5.0)
        gnss_vy[idx] -= 7.0 * np.sin(spoof_t / 5.0)
        cn0_series[idx] = 52.5 + np.random.normal(0, 0.2)
        clock_drift_series[idx] = 8.5e-7 + np.random.normal(0, 1e-8)
        multipath_indicator[idx] = 4.5 + np.random.normal(0, 0.4)

    dataset = {
        "time": t,
        "dt": dt,
        "origin": (origin_lat, origin_lon, origin_alt),
        "ground_truth": {
            "x_ned": x_ned,
            "y_ned": y_ned,
            "z_ned": z_ned,
            "vx_ned": vx_ned,
            "vy_ned": vy_ned,
            "vz_ned": vz_ned,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "lat": gt_lat,
            "lon": gt_lon,
            "alt": gt_alt,
            "acc_body": acc_body_true,
            "gyro_body": gyro_body_true,
        },
        "sensors": {
            "imu_acc": imu_acc,
            "imu_gyro": imu_gyro,
            "baro_alt": baro_alt,
            "mag_body": mag_body,
            "gnss_lat": gnss_lat,
            "gnss_lon": gnss_lon,
            "gnss_alt": gnss_alt,
            "gnss_vx": gnss_vx,
            "gnss_vy": gnss_vy,
            "gnss_vz": gnss_vz,
            "gnss_cn0": cn0_series,
            "gnss_clock_drift": clock_drift_series,
            "gnss_multipath": multipath_indicator,
            "gnss_available": is_available,
        },
        "attack_labels": {
            "is_outage": outage_mask,
            "is_spoofed": is_spoofed_gt,
        }
    }
    return dataset


def load_iovnbd_or_synthetic(
    data_dir: str = "data/io_vnbd",
    duration_sec: float = 600.0,
    dt: float = 0.01
) -> Dict[str, Any]:
    """Load data from IO-VNBD dataset or generate high-fidelity synthetic trajectory."""
    os.makedirs(data_dir, exist_ok=True)
    csv_file = os.path.join(data_dir, "vehicle_trajectory_sample.npz")
    
    if os.path.exists(csv_file):
        try:
            loaded = np.load(csv_file, allow_pickle=True)
            data = {key: loaded[key].item() if loaded[key].shape == () else loaded[key] for key in loaded.files}
            if "ground_truth" in data:
                return data
        except Exception:
            pass
            
    dataset = generate_synthetic_vehicle_trajectory(duration_sec=duration_sec, dt=dt)
    
    try:
        np.savez_compressed(
            csv_file,
            time=dataset["time"],
            dt=dataset["dt"],
            origin=dataset["origin"],
            ground_truth=dataset["ground_truth"],
            sensors=dataset["sensors"],
            attack_labels=dataset["attack_labels"]
        )
    except Exception as e:
        print(f"Warning: Could not cache dataset to {csv_file}: {e}")
        
    return dataset


def compute_metrics(
    gt_pos: np.ndarray,
    est_pos: np.ndarray,
    gt_yaw: Optional[np.ndarray] = None,
    est_yaw: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute standard navigation metrics (RMSE, CEP50, CEP95, Max Error)."""
    gt_pos = np.asarray(gt_pos)
    est_pos = np.asarray(est_pos)
    
    err_2d = np.linalg.norm(gt_pos[:, :2] - est_pos[:, :2], axis=1)
    rmse_2d = float(np.sqrt(np.mean(err_2d**2)))
    max_err_2d = float(np.max(err_2d))
    cep50 = float(np.percentile(err_2d, 50))
    cep95 = float(np.percentile(err_2d, 95))
    
    metrics = {
        "rmse_2d_m": rmse_2d,
        "max_err_2d_m": max_err_2d,
        "cep50_m": cep50,
        "cep95_m": cep95,
    }
    
    if gt_pos.shape[1] >= 3 and est_pos.shape[1] >= 3:
        err_3d = np.linalg.norm(gt_pos[:, :3] - est_pos[:, :3], axis=1)
        metrics["rmse_3d_m"] = float(np.sqrt(np.mean(err_3d**2)))
        metrics["max_err_3d_m"] = float(np.max(err_3d))
        
    if gt_yaw is not None and est_yaw is not None:
        yaw_err = wrap_to_pi(gt_yaw - est_yaw)
        metrics["rmse_yaw_deg"] = float(np.degrees(np.sqrt(np.mean(yaw_err**2))))
        
    return metrics
