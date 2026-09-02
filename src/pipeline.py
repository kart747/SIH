"""
Main Intelligent Dead Reckoning Pipeline
Problem Statement: SIH26168 (Smart India Hackathon 2026)

Architecture:
    1. Ingests 6-DoF IMU, Barometer, Magnetometer, and GNSS observations in real-time.
    2. Runs GNSS integrity & Spoofing Detector (detects C/N0 anomalies, clock jumps, Doppler mismatches).
    3. Executes Adaptive Sensor Fusion Engine (dynamically switching between EKF and UKF based on motion non-linearity).
    4. Concurrently updates Deep Neural Dead Reckoning model (GRU / TCN) over sliding sensor windows.
    5. Fuses Kinematic Filter state with Neural displacement predictions using dynamic confidence weighting.
    6. Performance: Target < 10ms per timestep inference latency on standard CPU/GPU.
    7. Outputs: Geodetic coordinates (lat, lon, alt), linear velocities, Euler attitude, and trajectory logging.
"""

from __future__ import annotations
import os
import sys
import time
from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import (
    geodetic_to_ned,
    ned_to_geodetic,
    wrap_to_pi,
    compute_metrics,
    GRAVITY_STANDARD
)
from src.sensor_fusion.fusion import AdaptiveFusionEngine
from src.gnss.gnss_handler import GNSSHandler
from src.neural.gru_model import GRUDeadReckoning
from src.neural.tcn_model import TCNDeadReckoning


class IntelligentDeadReckoningPipeline:
    """
    Unified Real-Time Hybrid Dead Reckoning System.
    """

    def __init__(
        self,
        dt: float = 0.01,
        origin_lat: float = 28.6139,
        origin_lon: float = 77.2090,
        origin_alt: float = 216.0,
        neural_model_type: str = "gru", # "gru" or "tcn"
        neural_weights_path: Optional[str] = None,
        window_size: int = 100,
        device: Optional[str] = None,
        log_trajectory: bool = True
    ) -> None:
        """
        Initialize the complete Dead Reckoning Pipeline.
        """
        self.dt = float(dt)
        self.origin_lat = float(origin_lat)
        self.origin_lon = float(origin_lon)
        self.origin_alt = float(origin_alt)
        self.window_size = int(window_size)
        self.log_trajectory = log_trajectory
        
        # 1. Kinematic Adaptive Fusion Engine (EKF/UKF switcher)
        self.fusion_engine = AdaptiveFusionEngine(dt=dt)
        
        # 2. GNSS Anomaly & Spoofing Handler
        self.gnss_handler = GNSSHandler(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            origin_alt=origin_alt,
            spoof_threshold=0.5
        )
        
        # 3. Neural Dead Reckoning Model (GRU or TCN)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.neural_model_type = neural_model_type.lower()
        self.has_trained_weights = False
        
        if self.neural_model_type == "gru":
            self.neural_model = GRUDeadReckoning(input_dim=7, hidden_dim=256, num_layers=3, dropout=0.0)
        else:
            self.neural_model = TCNDeadReckoning(input_dim=7, num_channels=[128, 128, 128, 128, 128, 128], dropout=0.0)
            
        if neural_weights_path is not None and os.path.exists(neural_weights_path):
            ckpt = torch.load(neural_weights_path, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            self.neural_model.load_state_dict(state_dict, strict=False)
            self.has_trained_weights = True
            
        self.neural_model.to(self.device)
        self.neural_model.eval()
        
        # Sliding feature window buffer for neural network
        self.feature_buffer: List[np.ndarray] = []
        self.last_baro_alt: Optional[float] = None
        
        # State tracking
        self.is_initialized = False
        self.fused_pos_ned = np.zeros(3, dtype=np.float64)
        self.fused_vel_ned = np.zeros(3, dtype=np.float64)
        self.fused_attitude = np.zeros(3, dtype=np.float64)
        self.neural_pos_ned = np.zeros(3, dtype=np.float64)
        
        self.current_time: float = 0.0
        self.outage_duration_sec: float = 0.0
        self.trajectory_log: List[Dict[str, Any]] = []

    def set_neural_weights(self, state_dict: Dict[str, Any]) -> None:
        """Load trained neural weights into the active model."""
        self.neural_model.load_state_dict(state_dict, strict=False)
        self.has_trained_weights = True
        self.neural_model.eval()

    def reset(
        self,
        init_pos_ned: Optional[np.ndarray] = None,
        init_vel_ned: Optional[np.ndarray] = None,
        init_att: Optional[np.ndarray] = None
    ) -> None:
        """Reset internal states."""
        p0 = init_pos_ned if init_pos_ned is not None else np.zeros(3)
        v0 = init_vel_ned if init_vel_ned is not None else np.zeros(3)
        a0 = init_att if init_att is not None else np.zeros(3)
        
        init_state = np.hstack([p0, v0, a0])
        self.fusion_engine.set_initial_state(init_state)
        
        self.fused_pos_ned = np.copy(p0)
        self.fused_vel_ned = np.copy(v0)
        self.fused_attitude = np.copy(a0)
        self.neural_pos_ned = np.copy(p0)
        
        self.feature_buffer.clear()
        self.last_baro_alt = None
        self.current_time = 0.0
        self.outage_duration_sec = 0.0
        self.is_initialized = (init_vel_ned is not None)
        self.gnss_handler.set_origin(self.origin_lat, self.origin_lon, self.origin_alt)
        self.trajectory_log.clear()

    def step(
        self,
        imu_acc: np.ndarray,
        imu_gyro: np.ndarray,
        baro_alt: float,
        mag_body: Optional[np.ndarray] = None,
        gnss_lat: Optional[float] = None,
        gnss_lon: Optional[float] = None,
        gnss_alt: Optional[float] = None,
        gnss_vx: Optional[float] = None,
        gnss_vy: Optional[float] = None,
        gnss_vz: Optional[float] = None,
        gnss_cn0: float = 0.0,
        gnss_clock_drift: float = 0.0,
        gnss_multipath: float = 0.0,
        dt: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute a single real-time inference step. Target latency < 10ms.
        """
        t_start = time.perf_counter()
        dt_val = float(dt if dt is not None else self.dt)
        self.current_time += dt_val
        
        imu_acc = np.asarray(imu_acc, dtype=np.float64)
        imu_gyro = np.asarray(imu_gyro, dtype=np.float64)
        
        # 1. Barometer feature prep
        delta_baro = 0.0
        if self.last_baro_alt is not None:
            delta_baro = float(baro_alt - self.last_baro_alt)
        self.last_baro_alt = float(baro_alt)
        
        feature_vec = np.hstack([imu_acc, imu_gyro, [delta_baro]])
        self.feature_buffer.append(feature_vec)
        if len(self.feature_buffer) > self.window_size:
            self.feature_buffer.pop(0)
            
        # 2. Kinematic Filter Prediction (Adaptive EKF / UKF)
        filter_pred, active_filter, nonlin_score = self.fusion_engine.predict(imu_acc, imu_gyro, dt_val)
        
        # 3. Barometer and Magnetometer Measurement Updates
        baro_z_ned = -(baro_alt - self.origin_alt)
        self.fusion_engine.update_barometer(baro_z_ned)
        
        if mag_body is not None:
            mag_yaw = float(np.arctan2(-mag_body[1], mag_body[0]))
            self.fusion_engine.update_magnetometer_yaw(mag_yaw)
            
        # 4. GNSS Ingestion & Spoofing Detection
        current_filter_vel = self.fusion_engine.get_state()[3:6]
        
        if gnss_lat is not None and not np.isnan(gnss_lat):
            gnss_result = self.gnss_handler.process_gnss_sample(
                lat=gnss_lat, lon=gnss_lon, alt=gnss_alt,
                vx=gnss_vx, vy=gnss_vy, vz=gnss_vz,
                cn0=gnss_cn0, clock_drift=gnss_clock_drift,
                multipath=gnss_multipath,
                ins_vel_ned=current_filter_vel,
                dt=dt_val
            )
        else:
            gnss_result = {
                "status": "OUTAGE",
                "is_valid": False,
                "trust_weight": 0.0,
                "pos_ned": None,
                "vel_ned": None,
                "spoof_confidence": 0.0,
                "diagnostics": {"reason": "No GNSS measurement provided"}
            }
            
        # Auto-initialize state from initial valid GNSS fix if uninitialized
        if not self.is_initialized and gnss_result["is_valid"]:
            p_init = gnss_result["pos_ned"]
            v_init = gnss_result["vel_ned"]
            current_att = self.fusion_engine.get_state()[6:9]
            # Initial yaw from velocity vector if moving
            speed = float(np.linalg.norm(v_init[:2]))
            if speed > 1.0:
                init_yaw = float(np.arctan2(v_init[1], v_init[0]))
                current_att[2] = init_yaw
            init_st = np.hstack([p_init, v_init, current_att])
            self.fusion_engine.set_initial_state(init_st)
            self.fused_pos_ned = np.copy(p_init)
            self.fused_vel_ned = np.copy(v_init)
            self.neural_pos_ned = np.copy(p_init)
            self.is_initialized = True
            
        # 5. GNSS Measurement Update (if healthy)
        if gnss_result["is_valid"]:
            self.fusion_engine.update_gnss(gnss_result["pos_ned"], gnss_result["vel_ned"])
            self.outage_duration_sec = 0.0
        else:
            self.outage_duration_sec += dt_val
            
        filter_state = self.fusion_engine.get_state()
        filter_pos_ned = filter_state[0:3]
        filter_vel_ned = filter_state[3:6]
        filter_att = filter_state[6:9]
        
        # 6. Neural Dead Reckoning Step
        neural_delta_pos = None
        if self.has_trained_weights and len(self.feature_buffer) >= min(20, self.window_size):
            window_arr = np.array(self.feature_buffer, dtype=np.float32)
            if len(window_arr) < self.window_size:
                pad_len = self.window_size - len(window_arr)
                pad_block = np.repeat(window_arr[:1], pad_len, axis=0)
                window_arr = np.vstack([pad_block, window_arr])
                
            window_t = torch.from_numpy(window_arr).unsqueeze(0).to(self.device)
            with torch.no_grad():
                delta_pos_pred, _ = self.neural_model.predict_step(window_t)
                neural_delta_pos = delta_pos_pred.cpu().numpy().astype(np.float64)
                
        # 7. Multi-Layer Hybrid Fusion
        gnss_trust = float(gnss_result["trust_weight"])
        
        if gnss_result["is_valid"] and gnss_trust > 0.5:
            # Healthy GNSS
            self.fused_pos_ned = np.copy(filter_pos_ned)
            self.fused_vel_ned = np.copy(filter_vel_ned)
            self.fused_attitude = np.copy(filter_att)
            self.neural_pos_ned = np.copy(filter_pos_ned)
        else:
            # Outage / Spoofing Dead Reckoning
            if neural_delta_pos is not None:
                # Blend Kinematic Inertial Integration with Trained Neural Odometry
                tau = 15.0
                alpha = float(np.exp(-self.outage_duration_sec / tau))
                neural_vel = neural_delta_pos / dt_val
                self.fused_vel_ned = alpha * filter_vel_ned + (1.0 - alpha) * neural_vel
                self.fused_pos_ned += self.fused_vel_ned * dt_val
            else:
                # Pure Kinematic Dead Reckoning from strapdown mechanization
                self.fused_pos_ned = np.copy(filter_pos_ned)
                self.fused_vel_ned = np.copy(filter_vel_ned)
                
            self.fused_attitude = np.copy(filter_att)
            
        # 8. Geodetic output
        fused_lat, fused_lon, fused_alt = ned_to_geodetic(
            self.fused_pos_ned[0], self.fused_pos_ned[1], self.fused_pos_ned[2],
            self.origin_lat, self.origin_lon, self.origin_alt
        )
        
        t_end = time.perf_counter()
        step_latency_ms = (t_end - t_start) * 1000.0
        
        step_output = {
            "time": self.current_time,
            "lat": fused_lat,
            "lon": fused_lon,
            "alt": fused_alt,
            "pos_ned": np.copy(self.fused_pos_ned),
            "vel_ned": np.copy(self.fused_vel_ned),
            "attitude_rpy": np.copy(self.fused_attitude),
            "heading_deg": float(np.degrees(self.fused_attitude[2])),
            "gnss_status": gnss_result["status"],
            "spoof_confidence": float(gnss_result["spoof_confidence"]),
            "is_spoofed": bool(gnss_result["status"] == "SPOOFED"),
            "active_filter": active_filter,
            "nonlinearity_score": nonlin_score,
            "latency_ms": step_latency_ms
        }
        
        if self.log_trajectory:
            self.trajectory_log.append(step_output)
            
        return step_output

    def run_batch(self, dataset_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Run the full pipeline offline on a dataset dictionary.
        """
        time_arr = dataset_dict["time"]
        sensors = dataset_dict["sensors"]
        n_samples = len(time_arr)
        
        origin = dataset_dict.get("origin", (self.origin_lat, self.origin_lon, self.origin_alt))
        self.origin_lat, self.origin_lon, self.origin_alt = origin
        self.gnss_handler.set_origin(*origin)
        
        # Initialize initial state from ground truth or first fix if available
        gt = dataset_dict.get("ground_truth", {})
        if "vx_ned" in gt and len(gt["vx_ned"]) > 0:
            init_p = np.array([gt["x_ned"][0], gt["y_ned"][0], gt["z_ned"][0]])
            init_v = np.array([gt["vx_ned"][0], gt["vy_ned"][0], gt["vz_ned"][0]])
            init_a = np.array([gt["roll"][0], gt["pitch"][0], gt["yaw"][0]])
            self.reset(init_pos_ned=init_p, init_vel_ned=init_v, init_att=init_a)
        else:
            self.reset()
            
        records = []
        for i in range(n_samples):
            dt_step = dataset_dict["dt"]
            out = self.step(
                imu_acc=sensors["imu_acc"][i],
                imu_gyro=sensors["imu_gyro"][i],
                baro_alt=sensors["baro_alt"][i],
                mag_body=sensors["mag_body"][i],
                gnss_lat=sensors["gnss_lat"][i],
                gnss_lon=sensors["gnss_lon"][i],
                gnss_alt=sensors["gnss_alt"][i],
                gnss_vx=sensors["gnss_vx"][i],
                gnss_vy=sensors["gnss_vy"][i],
                gnss_vz=sensors["gnss_vz"][i],
                gnss_cn0=sensors["gnss_cn0"][i],
                gnss_clock_drift=sensors["gnss_clock_drift"][i],
                gnss_multipath=sensors["gnss_multipath"][i],
                dt=dt_step
            )
            records.append({
                "time": out["time"],
                "lat": out["lat"],
                "lon": out["lon"],
                "alt": out["alt"],
                "x_ned": out["pos_ned"][0],
                "y_ned": out["pos_ned"][1],
                "z_ned": out["pos_ned"][2],
                "vx_ned": out["vel_ned"][0],
                "vy_ned": out["vel_ned"][1],
                "vz_ned": out["vel_ned"][2],
                "roll": out["attitude_rpy"][0],
                "pitch": out["attitude_rpy"][1],
                "yaw": out["attitude_rpy"][2],
                "heading_deg": out["heading_deg"],
                "gnss_status": out["gnss_status"],
                "spoof_confidence": out["spoof_confidence"],
                "is_spoofed": out["is_spoofed"],
                "active_filter": out["active_filter"],
                "nonlinearity_score": out["nonlinearity_score"],
                "latency_ms": out["latency_ms"]
            })
            
        df = pd.DataFrame(records)
        return df

    def save_trajectory_csv(self, file_path: str) -> None:
        """Save logged trajectory records to CSV."""
        if not self.trajectory_log:
            return
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        flat_records = []
        for r in self.trajectory_log:
            flat_records.append({
                "time": r["time"],
                "lat": r["lat"],
                "lon": r["lon"],
                "alt": r["alt"],
                "x_ned": r["pos_ned"][0],
                "y_ned": r["pos_ned"][1],
                "z_ned": r["pos_ned"][2],
                "vx_ned": r["vel_ned"][0],
                "vy_ned": r["vel_ned"][1],
                "vz_ned": r["vel_ned"][2],
                "roll": r["attitude_rpy"][0],
                "pitch": r["attitude_rpy"][1],
                "yaw": r["attitude_rpy"][2],
                "heading_deg": r["heading_deg"],
                "gnss_status": r["gnss_status"],
                "spoof_confidence": r["spoof_confidence"],
                "is_spoofed": r["is_spoofed"],
                "active_filter": r["active_filter"],
                "latency_ms": r["latency_ms"]
            })
        df = pd.DataFrame(flat_records)
        df.to_csv(file_path, index=False)
