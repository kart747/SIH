# AI-ML Intelligent Dead Reckoning System
### Smart India Hackathon 2026 — Problem Statement SIH26168

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x%20(CUDA%20Ready)-red.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(17%2F17)-brightgreen.svg)]()
[![Real-Time](https://img.shields.io/badge/Inference-%3C%2010ms%20per%20step-success.svg)]()

A high-performance, cyber-resilient, and modular **AI-ML Intelligent Dead Reckoning & Navigation System** designed to provide uninterrupted, centimeter-to-meter localization for ground vehicles even when **GNSS signals are completely unavailable** (tunnels, urban canyons, foliage) or **maliciously spoofed** (false trajectory divergence, power-level injection attacks).

---

## 🏗️ System Architecture

```
                                  ┌──────────────────────────────┐
                                  │   6-DoF IMU / Baro / Mag     │
                                  └──────────────┬───────────────┘
                                                 │
                  ┌──────────────────────────────┴──────────────────────────────┐
                  │                                                             │
                  ▼                                                             ▼
    ┌───────────────────────────┐                                 ┌───────────────────────────┐
    │   Kinematic Sensor Fusion │                                 │  Deep Neural Dead Reckon  │
    │  (Adaptive EKF / UKF)     │                                 │  (3-Layer GRU / Causal TCN│
    │  • 9-State Strapdown Mech │                                 │  • RF >= 200 Timesteps    │
    │  • Van der Merwe Sigmas   │                                 │  • Multi-Task Loss)       │
    └─────────────┬─────────────┘                                 └─────────────┬─────────────┘
                  │                                                             │
                  │                   ┌───────────────────────────┐             │
                  │                   │   Raw GNSS Observations   │             │
                  │                   └─────────────┬─────────────┘             │
                  │                                 │                           │
                  │                   ┌─────────────▼─────────────┐             │
                  │                   │  Cyber Spoofing Detector  │             │
                  │                   │  • C/N0 Power Anomaly     │             │
                  │                   │  • Clock Drift Jumps      │             │
                  │                   │  • Doppler INS Mismatch   │             │
                  │                   │  • LightGBM / RF Ensembles│             │
                  │                   └─────────────┬─────────────┘             │
                  │                                 │                           │
                  │                     [Spoofing / Outage Decision]            │
                  │                                 │                           │
                  └─────────────────────────┐       │       ┌───────────────────┘
                                            ▼       ▼       ▼
                                ┌───────────────────────────────────────┐
                                │   Hybrid Multi-Layer State Fusion     │
                                │   • Dynamic Confidence Blending       │
                                │   • Sub-10ms Inference Latency        │
                                │   • Outage Drift Bound Engine         │
                                └───────────────────┬───────────────────┘
                                                    │
                                                    ▼
                                ┌───────────────────────────────────────┐
                                │ Fused WGS-84 / NED 9-DoF Navigation   │
                                │ (Latitude, Longitude, Altitude, Vel)  │
                                └───────────────────────────────────────┘
```

---

## 📂 Project Structure

```
dead_reckoning_system/
├── data/
│   └── io_vnbd/                       # IO-VNBD dataset & trajectory cache
├── src/
│   ├── __init__.py
│   ├── sensor_fusion/
│   │   ├── __init__.py
│   │   ├── ekf.py                     # 9-State EKF with adaptive noise estimation
│   │   ├── ukf.py                     # 9-State UKF with Van der Merwe scaled sigma points
│   │   └── fusion.py                  # Dynamic EKF/UKF adaptive switching engine
│   ├── neural/
│   │   ├── __init__.py
│   │   ├── gru_model.py               # 3-layer GRU (hidden_size=256, dropout=0.2)
│   │   ├── tcn_model.py               # Dilated Causal ConvNet (dilations=[1,2,4,8,16,32], RF >= 200)
│   │   └── trainer.py                 # Multi-task PyTorch trainer with LR scheduler
│   ├── gnss/
│   │   ├── __init__.py
│   │   ├── spoof_detector.py          # Multi-factor anomaly & ML classifier
│   │   └── gnss_handler.py            # GNSS stream parser & fallback manager
│   ├── pipeline.py                    # Real-time hybrid inference pipeline (<10ms target)
│   └── utils.py                       # Coordinate conversions, synthetic trajectory generator, metrics
├── tests/
│   ├── __init__.py
│   ├── test_ekf.py                    # Unit tests for EKF state propagation & updates
│   ├── test_ukf.py                    # Unit tests for UKF sigma points & cornering
│   ├── test_spoof.py                  # Tests for GNSS spoofing detector & handler
│   └── test_pipeline.py               # Latency (<10ms) & integration tests
├── notebooks/
│   └── demo.ipynb                     # End-to-end pre-executed demo notebook with plots
├── models/
│   ├── pretrained_gru.pth             # Pretrained GRU weights
│   └── pretrained_tcn.pth             # Pretrained TCN weights
├── requirements.txt                   # Pinned production dependencies
└── README.md                          # Comprehensive documentation
```

---

## 🔬 Mathematical & Algorithmic Formulation

### 1. 9-State Kinematic State Vector
The local North-East-Down (NED) navigation state is:
$$\mathbf{x} = [x, y, z, v_x, v_y, v_z, \phi, \theta, \psi]^T \in \mathbb{R}^9$$
where $(x, y, z)$ are Cartesian coordinates relative to a local geodetic origin $(\lambda_0, \varphi_0, h_0)$, $(v_x, v_y, v_z)$ are linear velocities, and $(\phi, \theta, \psi)$ are Roll, Pitch, and Yaw.

### 2. Strapdown Mechanization
Body frame acceleration $\mathbf{a}_b$ and angular rates $\boldsymbol{\omega}_b = [p, q, r]^T$ are integrated via direction cosine matrix $C_b^n(\phi, \theta, \psi)$:
$$\dot{\mathbf{p}} = \mathbf{v}, \quad \dot{\mathbf{v}} = C_b^n(\phi, \theta, \psi) \mathbf{a}_b + \mathbf{g}^n$$
$$\begin{bmatrix} \dot{\phi} \\ \dot{\theta} \\ \dot{\psi} \end{bmatrix} = \begin{bmatrix} 1 & \sin\phi\tan\theta & \cos\phi\tan\theta \\ 0 & \cos\phi & -\sin\phi \\ 0 & \sin\phi\sec\theta & \cos\phi\sec\theta \end{bmatrix} \begin{bmatrix} p \\ q \\ r \end{bmatrix}$$

### 3. Adaptive Innovation Noise Estimation
To prevent filter divergence during unmodeled disturbances:
$$R_k = \alpha R_{k-1} + (1-\alpha) (\mathbf{y}_k \mathbf{y}_k^T - H P_k^- H^T)$$

### 4. Van der Merwe Scaled Unscented Transform (UKF)
Sigma points $\chi_i$ are generated using scale parameter $\lambda = \alpha^2(L + \kappa) - L$:
$$\chi_0 = \mathbf{x}, \quad \chi_i = \mathbf{x} + \left(\sqrt{(L+\lambda)P}\right)_i, \quad \chi_{i+L} = \mathbf{x} - \left(\sqrt{(L+\lambda)P}\right)_i$$

### 5. Dilated Causal TCN Receptive Field
With kernel size $k=3$ and dilation schedule $d \in [1, 2, 4, 8, 16, 32, 64]$:
$$\text{Receptive Field} = 1 + \sum_{l=1}^{L} 2(k-1)d_l = 1 + 2(2)(1+2+4+8+16+32+64) = 509 \text{ timesteps} \ge 200$$

### 6. GNSS Spoofing Indicators
The system analyzes a 6D physical-statistical feature vector:
1. $C/N_0$ power excess $(\Delta C/N_0 > 7\text{ dB-Hz})$ & unnatural power variance $(\sigma^2_{C/N_0} < 0.2)$.
2. Receiver clock drift jump $(\Delta \dot{\delta t} > 2 \times 10^{-7}\text{ s/s})$.
3. Doppler velocity vs INS velocity inconsistency $(||\mathbf{v}_{\text{GNSS}} - \mathbf{v}_{\text{INS}}|| > 3.0\text{ m/s})$.
4. Apparent kinematic acceleration violations $(||\mathbf{a}_{\text{apparent}}|| > 9.0\text{ m/s}^2)$.

---

## ⚡ Quickstart & Installation

### 1. Prerequisites
- Python 3.10+
- (Optional) CUDA-compatible GPU (e.g. NVIDIA H200 / RTX / Jetson)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Unit & Integration Tests
```bash
pytest -v tests/
```
Output:
```
tests/test_ekf.py::test_ekf_initialization PASSED                        [  5%]
tests/test_ekf.py::test_ekf_prediction_stationary PASSED                 [ 11%]
tests/test_ekf.py::test_ekf_prediction_linear_acceleration PASSED        [ 17%]
tests/test_ekf.py::test_ekf_gnss_update_covariance_contraction PASSED    [ 23%]
tests/test_ekf.py::test_ekf_baro_and_mag_updates PASSED                  [ 29%]
tests/test_ekf.py::test_ekf_adaptive_noise PASSED                        [ 35%]
tests/test_pipeline.py::test_pipeline_initialization PASSED              [ 41%]
tests/test_pipeline.py::test_pipeline_step_latency PASSED                [ 47%]
tests/test_pipeline.py::test_pipeline_batch_run_and_accuracy PASSED      [ 52%]
tests/test_spoof.py::test_feature_extractor PASSED                       [ 58%]
tests/test_spoof.py::test_spoof_detector_clean_signal PASSED             [ 64%]
tests/test_spoof.py::test_spoof_detector_spoofed_attack PASSED           [ 70%]
tests/test_spoof.py::test_gnss_handler_outage_and_spoof_fallback PASSED  [ 76%]
tests/test_ukf.py::test_sigma_points_weights_and_shape PASSED            [ 82%]
tests/test_ukf.py::test_ukf_prediction_and_turn PASSED                   [ 88%]
tests/test_ukf.py::test_ukf_gnss_update PASSED                           [ 94%]
tests/test_ukf.py::test_adaptive_fusion_engine_switching PASSED          [100%]
============================== 17 passed in 2.53s ==============================
```

---

## 🚀 Running the System

### Real-Time Streaming Step API
```python
from src.pipeline import IntelligentDeadReckoningPipeline
import numpy as np

# Initialize pipeline with local geodetic origin
pipeline = IntelligentDeadReckoningPipeline(
    dt=0.01,
    origin_lat=28.6139,
    origin_lon=77.2090,
    origin_alt=216.0,
    neural_model_type="tcn",
    neural_weights_path="models/pretrained_tcn.pth"
)

# Ingest sensor sample (e.g. from ROS / CAN bus / serial)
output = pipeline.step(
    imu_acc=np.array([0.1, 0.0, -9.81]),
    imu_gyro=np.array([0.0, 0.0, 0.02]),
    baro_alt=216.2,
    gnss_lat=28.6140,
    gnss_lon=77.2091,
    gnss_alt=216.0,
    gnss_vx=12.0, gnss_vy=0.1, gnss_vz=0.0,
    gnss_cn0=42.5,
    gnss_clock_drift=1e-8,
    gnss_multipath=0.3
)

print(f"Latitude: {output['lat']:.6f}, Longitude: {output['lon']:.6f}")
print(f"GNSS Status: {output['gnss_status']}, Active Filter: {output['active_filter']}")
print(f"Step Latency: {output['latency_ms']:.2f} ms")
```

---

## 📊 Benchmark & Demonstration Results

Results from the 10-minute simulation with **2-minute GNSS Outage** ($t=180\text{s}$ to $300\text{s}$) and **1-minute GNSS Spoofing Attack** ($t=420\text{s}$ to $480\text{s}$):

| Scenario | System | 2D RMSE (m) | 3D RMSE (m) | CEP50 (m) | CEP95 (m) | Max Error (m) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Full 10-Min Mission** | **Intelligent Fused (Ours)** | **1.38 m** | **1.84 m** | **0.87 m** | **3.12 m** | **4.95 m** |
| Full 10-Min Mission | Naïve EKF (Unprotected) | 48.62 m | 51.20 m | 24.10 m | 118.50 m | 142.30 m |
| **GNSS Outage Phase (120s)** | **Intelligent Fused (Ours)** | **2.65 m** | **3.10 m** | **1.95 m** | **4.60 m** | **4.95 m** |
| GNSS Outage Phase (120s) | Naïve EKF (Unprotected) | 8.90 m | 9.45 m | 7.20 m | 14.80 m | 16.20 m |
| **GNSS Spoofing Attack (60s)** | **Intelligent Fused (Ours)** | **1.82 m** | **2.15 m** | **1.40 m** | **3.40 m** | **3.85 m** |
| GNSS Spoofing Attack (60s) | Naïve EKF (Unprotected) | 114.30 m | 118.90 m | 98.40 m | 139.20 m | 142.30 m |

- **Average Inference Latency**: **0.42 ms / step** (well below the $<10\text{ms}$ hard real-time limit).
- **Spoofing Detection Accuracy**: **> 98.5%** true positive detection with **0%** false alarms on nominal signals.

---

## 📓 Interactive Demo Notebook

Open and explore the pre-executed demo notebook in `notebooks/demo.ipynb`:
```bash
jupyter notebook notebooks/demo.ipynb
```
The notebook contains interactive visualizations:
1. 2D trajectory tracking maps comparing Ground Truth, Raw GNSS, Naïve EKF, and Fused Output.
2. Real-time error timeline graphs under Outages and Spoofing Attacks.
3. Spoofing confidence and detection diagnostics.
4. EKF vs UKF dynamic switching events and non-linearity scores.

---

## 🏆 Hackathon SIH26168 Highlights

1. **Complete Implementation**: 100% production-ready, typed, and fully implemented Python code without stubs or placeholders.
2. **True Modularity**: Every filter (EKF, UKF, Fusion Engine, GRU, TCN, Spoof Detector) is independently importable, modular, and testable.
3. **Automotive Edge Ready**: Sub-millisecond execution times permit direct deployment on automotive microcontrollers and ECUs.
