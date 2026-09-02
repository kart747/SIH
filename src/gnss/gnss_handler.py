"""
GNSS Data Handler & Fallback Manager
Problem Statement: SIH26168 (Smart India Hackathon 2026)

Responsibilities:
    - Ingests raw GNSS geodetic & kinematic measurements
    - Maintains local navigation reference origin (lat0, lon0, alt0)
    - Manages cold-start initialization and anchor synchronization
    - Detects signal outages (loss-of-lock, NaN values, zero satellites)
    - Coordinates with GNSSSpoofDetector for real-time cyber-attack mitigation
    - Computes continuous GNSS trust weight [0.0, 1.0] for graceful fallback
"""

from __future__ import annotations
import os
import sys
from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import geodetic_to_ned, ned_to_geodetic
from src.gnss.spoof_detector import GNSSSpoofDetector


class GNSSHandler:
    """
    GNSS stream sanitizer, validator, and fallback coordinator.
    """

    def __init__(
        self,
        origin_lat: Optional[float] = None,
        origin_lon: Optional[float] = None,
        origin_alt: Optional[float] = None,
        spoof_threshold: float = 0.5,
        init_grace_samples: int = 5
    ) -> None:
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.origin_alt = origin_alt
        self.init_grace_samples = init_grace_samples
        self.valid_fix_count = 0
        
        self.spoof_detector = GNSSSpoofDetector(confidence_threshold=spoof_threshold)
        self.last_valid_pos_ned: Optional[np.ndarray] = None
        self.last_valid_vel_ned: Optional[np.ndarray] = None
        self.outage_counter: int = 0

    def set_origin(self, lat0: float, lon0: float, alt0: float) -> None:
        """Explicitly configure the local tangent origin."""
        self.origin_lat = float(lat0)
        self.origin_lon = float(lon0)
        self.origin_alt = float(alt0)
        self.valid_fix_count = 0
        self.spoof_detector.extractor.reset()

    def process_gnss_sample(
        self,
        lat: float,
        lon: float,
        alt: float,
        vx: float,
        vy: float,
        vz: float,
        cn0: float,
        clock_drift: float,
        multipath: float,
        ins_vel_ned: Optional[np.ndarray] = None,
        dt: float = 0.1
    ) -> Dict[str, Any]:
        """
        Process a single GNSS reading and determine validity and spoofing status.
        """
        # 1. Check for Signal Outage / Invalid Numbers
        has_nans = (
            np.isnan(lat) or np.isnan(lon) or np.isnan(alt) or
            np.isnan(vx) or np.isnan(vy) or np.isnan(vz) or
            np.isinf(lat) or np.isinf(lon) or np.isinf(alt)
        )
        is_zero_fix = (abs(lat) < 1e-4 and abs(lon) < 1e-4)
        is_zero_power = (cn0 <= 1.0)
        
        if has_nans or is_zero_fix or is_zero_power:
            self.outage_counter += 1
            return {
                "status": "OUTAGE",
                "is_valid": False,
                "trust_weight": 0.0,
                "pos_ned": None,
                "vel_ned": None,
                "spoof_confidence": 0.0,
                "diagnostics": {"reason": "GNSS signal loss or invalid fix"}
            }
            
        # Initialize origin if not yet set
        if self.origin_lat is None:
            self.origin_lat = float(lat)
            self.origin_lon = float(lon)
            self.origin_alt = float(alt)
            
        # Convert Geodetic (lat, lon, alt) to NED (North, East, Down)
        n, e, d = geodetic_to_ned(
            lat, lon, alt, self.origin_lat, self.origin_lon, self.origin_alt
        )
        pos_ned = np.array([n, e, d], dtype=np.float64)
        vel_ned = np.array([vx, vy, vz], dtype=np.float64)
        
        # Cold-start initialization grace period
        self.valid_fix_count += 1
        if self.valid_fix_count <= self.init_grace_samples:
            # During first few fixes, bypass Doppler comparison to allow filter convergence
            check_ins_vel = vel_ned
        else:
            check_ins_vel = ins_vel_ned
            
        # 2. Run Spoofing & Anomaly Detection
        spoof_conf, is_spoofed, diagnostics = self.spoof_detector.detect(
            gnss_pos_ned=pos_ned,
            gnss_vel_ned=vel_ned,
            cn0=cn0,
            clock_drift=clock_drift,
            multipath=multipath,
            ins_vel_ned=check_ins_vel,
            dt=dt
        )
        
        if is_spoofed:
            trust_weight = 0.0
            status = "SPOOFED"
            is_valid = False
        else:
            trust_weight = float(np.clip(1.0 - spoof_conf * 1.2, 0.0, 1.0))
            status = "HEALTHY"
            is_valid = True
            self.last_valid_pos_ned = np.copy(pos_ned)
            self.last_valid_vel_ned = np.copy(vel_ned)
            self.outage_counter = 0
            
        return {
            "status": status,
            "is_valid": is_valid,
            "trust_weight": trust_weight,
            "pos_ned": pos_ned,
            "vel_ned": vel_ned,
            "spoof_confidence": spoof_conf,
            "diagnostics": diagnostics
        }
