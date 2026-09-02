"""
Unit tests for GNSS Spoofing Detection and Handler
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gnss.spoof_detector import GNSSFeatureExtractor, GNSSSpoofDetector
from src.gnss.gnss_handler import GNSSHandler


def test_feature_extractor():
    extractor = GNSSFeatureExtractor()
    pos = np.array([100.0, 50.0, -10.0])
    vel = np.array([15.0, 0.0, 0.0])
    
    feats = extractor.extract(
        gnss_pos_ned=pos,
        gnss_vel_ned=vel,
        cn0=43.0,
        clock_drift=1e-8,
        multipath=0.4,
        ins_vel_ned=np.array([15.1, -0.1, 0.0]),
        dt=0.1
    )
    assert feats.shape == (6,)
    # Nominal cn0 dev should be near 1.0 dB
    assert np.isclose(feats[0], 1.0, atol=0.5)


def test_spoof_detector_clean_signal():
    detector = GNSSSpoofDetector(confidence_threshold=0.5)
    pos = np.array([10.0, 10.0, 0.0])
    vel = np.array([10.0, 0.0, 0.0])
    
    conf, is_spoofed, diag = detector.detect(
        gnss_pos_ned=pos,
        gnss_vel_ned=vel,
        cn0=42.0,
        clock_drift=1e-8,
        multipath=0.2,
        ins_vel_ned=vel,
        dt=0.1
    )
    assert not is_spoofed
    assert conf < 0.5


def test_spoof_detector_spoofed_attack():
    detector = GNSSSpoofDetector(confidence_threshold=0.5)
    pos = np.array([500.0, -400.0, 50.0])
    vel = np.array([35.0, -20.0, 5.0])
    
    # Abnormal C/N0 (54 dB-Hz), clock drift jump, huge velocity mismatch vs INS
    conf, is_spoofed, diag = detector.detect(
        gnss_pos_ned=pos,
        gnss_vel_ned=vel,
        cn0=54.0,
        clock_drift=9e-7,
        multipath=5.0,
        ins_vel_ned=np.array([10.0, 0.0, 0.0]),
        dt=0.1
    )
    assert is_spoofed
    assert conf >= 0.5


def test_gnss_handler_outage_and_spoof_fallback():
    handler = GNSSHandler(origin_lat=28.6139, origin_lon=77.2090, origin_alt=216.0)
    
    # 1. Healthy fix
    res_clean = handler.process_gnss_sample(
        lat=28.6140, lon=77.2091, alt=216.0,
        vx=10.0, vy=0.0, vz=0.0,
        cn0=42.5, clock_drift=1e-8, multipath=0.3,
        ins_vel_ned=np.array([10.0, 0.0, 0.0]),
        dt=0.1
    )
    assert res_clean["status"] == "HEALTHY"
    assert res_clean["is_valid"]
    assert res_clean["trust_weight"] > 0.5
    
    # 2. Outage (NaN)
    res_outage = handler.process_gnss_sample(
        lat=np.nan, lon=np.nan, alt=np.nan,
        vx=np.nan, vy=np.nan, vz=np.nan,
        cn0=0.0, clock_drift=0.0, multipath=0.0
    )
    assert res_outage["status"] == "OUTAGE"
    assert not res_outage["is_valid"]
    assert res_outage["trust_weight"] == 0.0
    
    # 3. Spoofed
    res_spoof = handler.process_gnss_sample(
        lat=28.6250, lon=77.2200, alt=220.0,
        vx=40.0, vy=-25.0, vz=0.0,
        cn0=54.0, clock_drift=1e-6, multipath=6.0,
        ins_vel_ned=np.array([10.0, 0.0, 0.0]),
        dt=0.1
    )
    assert res_spoof["status"] == "SPOOFED"
    assert not res_spoof["is_valid"]
    assert res_spoof["trust_weight"] == 0.0
