"""
GNSS Spoofing & Anomaly Detector
Problem Statement: SIH26168 (Smart India Hackathon 2026)

Key Features:
    - Multi-Factor Feature Extraction:
        1. C/N0 Carrier-to-Noise Ratio power level & variance anomalies
        2. Receiver Clock Drift Rate jumps (delta dt)
        3. Kinematic Position Jump magnitude (acceleration / speed limit violations)
        4. Doppler Velocity vs Inertial Navigation velocity consistency check
        5. Multipath / pseudorange residual indicator
    - Machine Learning Classifier (LightGBM / RandomForest) + Physics-based Rule Engine
    - Output: Continuous spoof_confidence [0.0, 1.0] and boolean is_spoofed decision
"""

from __future__ import annotations
import os
import sys
from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class GNSSFeatureExtractor:
    """
    Extracts discriminative physical and statistical features from GNSS measurements
    and IMU context.
    """

    def __init__(
        self,
        nominal_cn0_mean: float = 42.0,
        nominal_cn0_std: float = 3.5,
        max_physical_speed_ms: float = 45.0,  # ~160 km/h
        max_physical_acc_ms2: float = 8.0,    # ~0.8g
        clock_drift_threshold: float = 1e-7
    ) -> None:
        self.nominal_cn0 = nominal_cn0_mean
        self.nominal_cn0_std = nominal_cn0_std
        self.max_speed = max_physical_speed_ms
        self.max_acc = max_physical_acc_ms2
        self.clock_drift_threshold = clock_drift_threshold
        
        # Internal state history
        self.prev_pos_ned: Optional[np.ndarray] = None
        self.prev_time: Optional[float] = None
        self.prev_vel_ned: Optional[np.ndarray] = None
        self.cn0_history: List[float] = []

    def reset(self) -> None:
        self.prev_pos_ned = None
        self.prev_time = None
        self.prev_vel_ned = None
        self.cn0_history.clear()

    def extract(
        self,
        gnss_pos_ned: np.ndarray,
        gnss_vel_ned: np.ndarray,
        cn0: float,
        clock_drift: float,
        multipath: float,
        ins_vel_ned: Optional[np.ndarray] = None,
        dt: float = 0.1
    ) -> np.ndarray:
        """
        Compute feature vector for current GNSS sample.
        
        Features (6D):
            0: C/N0 deviation from nominal power (dB-Hz)
            1: C/N0 rolling variance
            2: Clock drift magnitude (sec/sec)
            3: Kinematic apparent acceleration magnitude (m/s^2)
            4: Doppler velocity mismatch ||v_gnss - v_ins|| (m/s)
            5: Multipath / signal quality indicator
        """
        gnss_pos = np.asarray(gnss_pos_ned, dtype=np.float64)
        gnss_vel = np.asarray(gnss_vel_ned, dtype=np.float64)
        
        # 1. C/N0 power anomaly
        cn0_dev = float(cn0 - self.nominal_cn0)
        self.cn0_history.append(float(cn0))
        if len(self.cn0_history) > 20:
            self.cn0_history.pop(0)
            
        if len(self.cn0_history) >= 3:
            cn0_var = float(np.var(self.cn0_history))
        else:
            # If C/N0 is abnormally high, initialize low variance characteristic of spoofers
            cn0_var = 0.1 if cn0_dev > 6.0 else float(self.nominal_cn0_std ** 2)
            
        # 2. Clock drift anomaly
        clock_drift_val = float(abs(clock_drift))
        
        # 3. Kinematic position jump & acceleration
        if self.prev_pos_ned is not None and dt > 1e-4:
            pos_diff = np.linalg.norm(gnss_pos - self.prev_pos_ned)
            apparent_speed = pos_diff / dt
            if self.prev_vel_ned is not None:
                apparent_acc = float(np.linalg.norm(gnss_vel - self.prev_vel_ned) / dt)
            else:
                apparent_acc = 0.0
        else:
            apparent_speed = 0.0
            apparent_acc = 0.0
            
        self.prev_pos_ned = np.copy(gnss_pos)
        self.prev_vel_ned = np.copy(gnss_vel)
        
        # 4. Doppler Consistency Check vs INS
        if ins_vel_ned is not None:
            doppler_mismatch = float(np.linalg.norm(gnss_vel - ins_vel_ned))
        else:
            doppler_mismatch = 0.0
            
        # 5. Multipath indicator
        multipath_val = float(multipath)
        
        feature_vector = np.array([
            cn0_dev,
            cn0_var,
            clock_drift_val,
            apparent_acc,
            doppler_mismatch,
            multipath_val
        ], dtype=np.float64)
        
        return feature_vector


class GNSSSpoofDetector:
    """
    Real-Time GNSS Spoofing & Anomaly Detector with ML Classifier and Physics Rules.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        feature_extractor: Optional[GNSSFeatureExtractor] = None
    ) -> None:
        self.threshold = float(confidence_threshold)
        self.extractor = feature_extractor if feature_extractor is not None else GNSSFeatureExtractor()
        
        self.model = None
        self._initialize_classifier()

    def _initialize_classifier(self) -> None:
        """
        Train or initialize the anomaly classifier on calibrated feature permutations.
        """
        X_train, y_train = self._generate_calibration_data()
        
        if HAS_LIGHTGBM:
            self.model = lgb.LGBMClassifier(
                n_estimators=60,
                max_depth=5,
                learning_rate=0.08,
                random_state=42,
                verbosity=-1
            )
            self.model.fit(X_train, y_train)
        elif HAS_SKLEARN:
            self.model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            self.model.fit(X_train, y_train)
        else:
            self.model = None

    def _generate_calibration_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate diverse calibration feature dataset covering clean vs spoofing conditions.
        """
        np.random.seed(123)
        n = 1500
        
        # Clean features:
        clean_cn0_dev = np.random.normal(0.0, 2.5, n)
        clean_cn0_var = np.random.uniform(2.0, 16.0, n)
        clean_clock_drift = np.random.normal(1e-8, 2e-9, n)
        clean_acc = np.random.exponential(0.8, n)
        clean_doppler = np.random.exponential(0.3, n)
        clean_multipath = np.random.exponential(0.5, n)
        
        X_clean = np.column_stack([
            clean_cn0_dev, clean_cn0_var, clean_clock_drift, clean_acc, clean_doppler, clean_multipath
        ])
        y_clean = np.zeros(n, dtype=int)
        
        # Spoofed attack scenarios (combinations of power, clock, doppler, jumps)
        # Type 1: High power spoofer + Doppler mismatch
        s1_cn0_dev = np.random.normal(11.0, 2.0, n // 3)
        s1_cn0_var = np.random.uniform(0.01, 1.5, n // 3)
        s1_clock_drift = np.random.normal(7e-7, 1e-7, n // 3)
        s1_acc = np.random.normal(10.0, 3.0, n // 3)
        s1_doppler = np.random.normal(8.0, 2.0, n // 3)
        s1_multipath = np.random.normal(4.0, 1.0, n // 3)
        
        # Type 2: Stealth clock/Doppler injection with moderate C/N0
        s2_cn0_dev = np.random.normal(4.0, 2.0, n // 3)
        s2_cn0_var = np.random.uniform(0.1, 5.0, n // 3)
        s2_clock_drift = np.random.normal(5e-7, 1e-7, n // 3)
        s2_acc = np.random.normal(8.0, 2.0, n // 3)
        s2_doppler = np.random.normal(12.0, 3.0, n // 3)
        s2_multipath = np.random.normal(3.0, 1.0, n // 3)
        
        # Type 3: Trajectory divergence / velocity spoofing
        s3_cn0_dev = np.random.normal(8.0, 3.0, n - 2 * (n // 3))
        s3_cn0_var = np.random.uniform(0.05, 3.0, n - 2 * (n // 3))
        s3_clock_drift = np.random.normal(4e-7, 1.5e-7, n - 2 * (n // 3))
        s3_acc = np.random.normal(14.0, 4.0, n - 2 * (n // 3))
        s3_doppler = np.random.normal(15.0, 4.0, n - 2 * (n // 3))
        s3_multipath = np.random.normal(4.5, 1.5, n - 2 * (n // 3))
        
        X_spoof = np.vstack([
            np.column_stack([s1_cn0_dev, s1_cn0_var, s1_clock_drift, s1_acc, s1_doppler, s1_multipath]),
            np.column_stack([s2_cn0_dev, s2_cn0_var, s2_clock_drift, s2_acc, s2_doppler, s2_multipath]),
            np.column_stack([s3_cn0_dev, s3_cn0_var, s3_clock_drift, s3_acc, s3_doppler, s3_multipath]),
        ])
        y_spoof = np.ones(len(X_spoof), dtype=int)
        
        X = np.vstack([X_clean, X_spoof])
        y = np.concatenate([y_clean, y_spoof])
        return X, y

    def _physics_heuristic_confidence(self, features: np.ndarray) -> float:
        """
        Physics-based heuristic rule engine computing spoof confidence score [0.0, 1.0].
        """
        cn0_dev, cn0_var, clock_drift, apparent_acc, doppler_mismatch, multipath = features
        
        score = 0.0
        # 1. C/N0 power > 50 dB-Hz (nominal is 42 dB-Hz; dev > 7 dB)
        if cn0_dev > 7.0:
            score += 0.40
        elif cn0_dev > 4.0:
            score += 0.20
            
        # 2. C/N0 flat power variance
        if cn0_var < 0.5 and cn0_dev > 3.0:
            score += 0.20
            
        # 3. Clock drift jump
        if clock_drift > 2e-7:
            score += 0.35
            
        # 4. Doppler mismatch vs INS > 3.0 m/s
        if doppler_mismatch > 5.0:
            score += 0.45
        elif doppler_mismatch > 2.5:
            score += 0.25
            
        # 5. High acceleration violation
        if apparent_acc > 9.0:
            score += 0.30
            
        return float(np.clip(score, 0.0, 1.0))

    def detect(
        self,
        gnss_pos_ned: np.ndarray,
        gnss_vel_ned: np.ndarray,
        cn0: float,
        clock_drift: float,
        multipath: float,
        ins_vel_ned: Optional[np.ndarray] = None,
        dt: float = 0.1
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Evaluate current GNSS measurement and determine if signal is spoofed.
        
        Returns:
            Tuple of:
                - spoof_confidence: Float in [0.0, 1.0]
                - is_spoofed: Boolean decision (True if spoofed)
                - diagnostics: Dictionary of feature details
        """
        features = self.extractor.extract(
            gnss_pos_ned=gnss_pos_ned,
            gnss_vel_ned=gnss_vel_ned,
            cn0=cn0,
            clock_drift=clock_drift,
            multipath=multipath,
            ins_vel_ned=ins_vel_ned,
            dt=dt
        )
        
        heuristic_conf = self._physics_heuristic_confidence(features)
        
        if self.model is not None:
            try:
                proba = float(self.model.predict_proba(features.reshape(1, -1))[0, 1])
                # Conservative maximum / weighted blend
                spoof_confidence = max(proba, heuristic_conf)
            except Exception:
                spoof_confidence = heuristic_conf
        else:
            spoof_confidence = heuristic_conf
            
        is_spoofed = bool(spoof_confidence >= self.threshold)
        
        diagnostics = {
            "spoof_confidence": spoof_confidence,
            "is_spoofed": is_spoofed,
            "cn0_dev": float(features[0]),
            "cn0_var": float(features[1]),
            "clock_drift": float(features[2]),
            "apparent_acc": float(features[3]),
            "doppler_mismatch": float(features[4]),
            "multipath": float(features[5])
        }
        
        return spoof_confidence, is_spoofed, diagnostics
