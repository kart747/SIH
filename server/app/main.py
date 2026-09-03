from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import sys
import os
import numpy as np

# ensure project src is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.sensor_fusion.ekf import ExtendedKalmanFilter9DoF
from src.pipeline import IntelligentDeadReckoningPipeline
from src.utils import geodetic_to_ned
import torch

app = FastAPI()

# per-device pipeline instances and anchors
pipeline_store: Dict[str, IntelligentDeadReckoningPipeline] = {}
# store GNSS origin anchor per device once first valid fix arrives
origin_store: Dict[str, tuple] = {}


def get_pipeline_for(device_id: str) -> IntelligentDeadReckoningPipeline:
    if device_id not in pipeline_store:
        weights_path = os.path.join(ROOT, "models", "pretrained_tcn.pth")
        p = IntelligentDeadReckoningPipeline(dt=0.01, neural_model_type="tcn", neural_weights_path=weights_path, window_size=200)
        pipeline_store[device_id] = p
    return pipeline_store[device_id]


class IMU(BaseModel):
    acc: list[float] = Field(..., description="Accelerometer [ax,ay,az] m/s^2")
    gyro: list[float] = Field(..., description="Gyroscope [p,q,r] rad/s")
    dt: Optional[float] = Field(0.01, description="time delta")


class GNSS(BaseModel):
    # Accept either geodetic fixes (lat/lon/alt) or pre-converted NED values
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt: Optional[float] = None
    speed: Optional[float] = None
    pos_ned: Optional[list[float]] = None
    vel_ned: Optional[list[float]] = None


class IngestPayload(BaseModel):
    device_id: str
    imu: IMU
    gnss: Optional[GNSS] = None


@app.get("/health")
def health():
    return {"status": "ok"}


def get_ekf_for(device_id: str) -> ExtendedKalmanFilter9DoF:
    if device_id not in ekf_store:
        ekf_store[device_id] = ExtendedKalmanFilter9DoF(dt=0.01)
    return ekf_store[device_id]


@app.post("/ingest")
def ingest(payload: IngestPayload):
    pipeline = get_pipeline_for(payload.device_id)

    imu = payload.imu
    acc = np.asarray(imu.acc, dtype=np.float64)
    gyro = np.asarray(imu.gyro, dtype=np.float64)
    dt = float(imu.dt or pipeline.dt)

    # Run pipeline IMU prediction step (no GNSS passed yet)
    try:
        step_out = pipeline.step(
            imu_acc=acc,
            imu_gyro=gyro,
            baro_alt=0.0,
            mag_body=None,
            gnss_lat=None,
            gnss_lon=None,
            gnss_alt=None,
            dt=dt
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"pipeline.step error: {e}")

    # If GNSS provided, set origin anchor (first valid fix) and convert to NED
    mahalanobis = None
    reverted = False
    used_velocity = None
    ai_velocity = None
    ekf_velocity = pipeline.fusion_engine.get_state()[3:6].copy()

    if payload.gnss is not None:
        # payload.gnss may carry lat/lon/alt or pos_ned; accept either
        gnss_obj = payload.gnss
        # If raw geodetic lat/lon/alt fields present (compatibility with Android sample)
        lat = getattr(gnss_obj, "lat", None)
        lon = getattr(gnss_obj, "lon", None)
        alt = getattr(gnss_obj, "alt", None)

        if lat is not None and lon is not None and alt is not None:
            # Initialize origin if not set
            if payload.device_id not in origin_store:
                origin_store[payload.device_id] = (float(lat), float(lon), float(alt))
                # set pipeline origin and reset internal GNSS handler
                pipeline.origin_lat, pipeline.origin_lon, pipeline.origin_alt = origin_store[payload.device_id]
                pipeline.gnss_handler.set_origin(*origin_store[payload.device_id])
                # Reset pipeline state using this initial GNSS fix
                pos0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                vel0 = np.zeros(3, dtype=np.float64)
                pipeline.reset(init_pos_ned=pos0, init_vel_ned=vel0)

            lat0, lon0, alt0 = origin_store[payload.device_id]
            # Convert to NED relative to stored origin
            pos_ned = np.array(geodetic_to_ned(float(lat), float(lon), float(alt), lat0, lon0, alt0), dtype=np.float64)

            # Determine GNSS velocity if provided in payload as vel_ned
            vel_ned = None
            if gnss_obj.vel_ned is not None:
                vel_ned = np.asarray(gnss_obj.vel_ned, dtype=np.float64)

            # Update fusion engine with converted NED measurement
            try:
                updated_state, mahalanobis = pipeline.fusion_engine.update_gnss(pos_ned, vel_ned)
            except Exception as e:
                shp_pos = getattr(pos_ned, "shape", None)
                shp_vel = getattr(vel_ned, "shape", None)
                raise HTTPException(status_code=500, detail=f"fusion update_gnss error: {e} pos={repr(pos_ned)} pos.shape={shp_pos} vel={repr(vel_ned)} vel.shape={shp_vel}")

        elif gnss_obj.pos_ned is not None:
            # Direct NED sample provided
            pos_ned = np.asarray(gnss_obj.pos_ned, dtype=np.float64)
            vel_ned = np.asarray(gnss_obj.vel_ned, dtype=np.float64) if gnss_obj.vel_ned is not None else None
            try:
                updated_state, mahalanobis = pipeline.fusion_engine.update_gnss(pos_ned, vel_ned)
            except Exception as e:
                # include shapes in error for debugging
                shp_pos = getattr(pos_ned, "shape", None)
                shp_vel = getattr(vel_ned, "shape", None)
                raise HTTPException(status_code=500, detail=f"fusion update_gnss error: {e} pos.shape={shp_pos} vel.shape={shp_vel}")

    # Compute AI (neural) predicted velocity from pipeline neural model if available
    try:
        if pipeline.has_trained_weights and len(pipeline.feature_buffer) >= min(20, pipeline.window_size):
            window_arr = np.array(pipeline.feature_buffer, dtype=np.float32)
            if len(window_arr) < pipeline.window_size:
                pad_len = pipeline.window_size - len(window_arr)
                pad_block = np.repeat(window_arr[:1], pad_len, axis=0)
                window_arr = np.vstack([pad_block, window_arr])

            window_t = torch.from_numpy(window_arr).unsqueeze(0).to(pipeline.device)
            with torch.no_grad():
                delta_pos_pred, _ = pipeline.neural_model.predict_step(window_t)
            ai_velocity = (delta_pos_pred.cpu().numpy().astype(np.float64).flatten() / float(dt))
        else:
            ai_velocity = np.zeros(3, dtype=np.float64)
    except Exception:
        ai_velocity = np.zeros(3, dtype=np.float64)

    ekf_velocity = pipeline.fusion_engine.get_state()[3:6].copy()
    deviation = float(np.linalg.norm(ai_velocity - ekf_velocity))
    threshold = 2.0
    if deviation > threshold:
        reverted = True
        # ensure fused velocity uses filter baseline
        pipeline.fused_vel_ned = ekf_velocity.copy()
        used_velocity = ekf_velocity.copy()
    else:
        # trust neural velocity and inject into fused pipeline velocity
        pipeline.fused_vel_ned = ai_velocity.copy()
        used_velocity = ai_velocity.copy()

    resp: Dict[str, Any] = {
        "device_id": payload.device_id,
        "ai_velocity": ai_velocity.tolist() if ai_velocity is not None else None,
        "ekf_velocity": ekf_velocity.tolist(),
        "used_velocity": used_velocity.tolist(),
        "reverted_to_ekf": reverted,
        "pipeline_state_time": float(pipeline.current_time)
    }
    if mahalanobis is not None:
        resp["gnss_mahalanobis"] = float(mahalanobis)

    return resp
