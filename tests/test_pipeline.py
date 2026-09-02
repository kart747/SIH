"""
Integration tests for the complete Intelligent Dead Reckoning Pipeline
"""

import pytest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pipeline import IntelligentDeadReckoningPipeline
from src.utils import generate_synthetic_vehicle_trajectory, compute_metrics, GRAVITY_STANDARD


def test_pipeline_initialization():
    pipeline_gru = IntelligentDeadReckoningPipeline(neural_model_type="gru")
    assert pipeline_gru.neural_model_type == "gru"
    
    pipeline_tcn = IntelligentDeadReckoningPipeline(neural_model_type="tcn")
    assert pipeline_tcn.neural_model_type == "tcn"


def test_pipeline_step_latency():
    pipeline = IntelligentDeadReckoningPipeline(dt=0.01)
    
    # Warm up step
    acc = np.array([0.0, 0.0, -GRAVITY_STANDARD])
    gyro = np.array([0.0, 0.0, 0.0])
    pipeline.step(imu_acc=acc, imu_gyro=gyro, baro_alt=216.0)
    
    latencies = []
    for _ in range(50):
        out = pipeline.step(
            imu_acc=acc + np.random.normal(0, 0.01, 3),
            imu_gyro=gyro + np.random.normal(0, 0.001, 3),
            baro_alt=216.0 + np.random.normal(0, 0.1),
            gnss_lat=28.6139,
            gnss_lon=77.2090,
            gnss_alt=216.0,
            gnss_vx=0.0, gnss_vy=0.0, gnss_vz=0.0,
            gnss_cn0=42.0, gnss_clock_drift=1e-8, gnss_multipath=0.2
        )
        latencies.append(out["latency_ms"])
        
    avg_latency = float(np.mean(latencies))
    # Target is < 10ms
    assert avg_latency < 10.0, f"Average latency {avg_latency:.2f}ms exceeds 10ms requirement"


def test_pipeline_batch_run_and_accuracy():
    # 60-second test trajectory with a short outage (t=20 to 30) and spoofing (t=40 to 50)
    dataset = generate_synthetic_vehicle_trajectory(
        duration_sec=60.0,
        dt=0.05,
        outage_start=20.0,
        outage_duration=10.0,
        spoof_start=40.0,
        spoof_duration=10.0,
        seed=101
    )
    
    pipeline = IntelligentDeadReckoningPipeline(dt=0.05, origin_lat=28.6139, origin_lon=77.2090, origin_alt=216.0)
    df_results = pipeline.run_batch(dataset)
    
    assert len(df_results) == len(dataset["time"])
    assert "x_ned" in df_results.columns
    assert "is_spoofed" in df_results.columns
    
    # Check spoofing detection during attack
    spoofed_flags = df_results.loc[(df_results["time"] >= 40.0) & (df_results["time"] < 50.0), "is_spoofed"]
    assert np.mean(spoofed_flags) > 0.8, "Spoofing detector should identify attack interval"
    
    # Compute metrics against ground truth
    gt_pos = np.column_stack([
        dataset["ground_truth"]["x_ned"],
        dataset["ground_truth"]["y_ned"],
        dataset["ground_truth"]["z_ned"]
    ])
    est_pos = df_results[["x_ned", "y_ned", "z_ned"]].to_numpy()
    
    metrics = compute_metrics(gt_pos, est_pos)
    # The intelligent fused estimate should maintain low error (< 15m RMSE over mixed 60s scenario with outages)
    assert metrics["rmse_2d_m"] < 15.0
