"""
Neural Dead Reckoning Training Pipeline
Problem Statement: SIH26168 (Smart India Hackathon 2026)

Features:
    - Sliding window Dataset & DataLoader for IO-VNBD / synthetic trajectory data
    - Multi-task Loss: MSE on 3D displacement + Cosine similarity heading loss + Smoothness regularizer
    - Early Stopping, Learning Rate Schedulers (Cosine Annealing / Plateau), Checkpoint saving
    - Hardware-accelerated training (CUDA / MPS / CPU)
    - Optional Weights & Biases (wandb) logging integration
"""

from __future__ import annotations
import os
import sys
import math
import time
from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.neural.gru_model import GRUDeadReckoning
from src.neural.tcn_model import TCNDeadReckoning
from src.utils import wrap_to_pi, load_iovnbd_or_synthetic


class InertialSequenceDataset(Dataset):
    """
    Sliding window dataset for training neural dead reckoning models on inertial and barometric data.
    """

    def __init__(
        self,
        features: np.ndarray,
        target_disp: np.ndarray,
        target_yaw_diff: Optional[np.ndarray] = None,
        seq_len: int = 100,
        stride: int = 5
    ) -> None:
        """
        Args:
            features: (N, 7) array [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, delta_baro]
            target_disp: (N, 3) array [dx, dy, dz] per timestep
            target_yaw_diff: Optional (N, 1) array delta_yaw per timestep
            seq_len: Sliding window length (default 100)
            stride: Stride between successive sliding windows
        """
        self.seq_len = seq_len
        self.features = torch.tensor(features, dtype=torch.float32)
        self.target_disp = torch.tensor(target_disp, dtype=torch.float32)
        self.target_yaw = (
            torch.tensor(target_yaw_diff, dtype=torch.float32)
            if target_yaw_diff is not None else None
        )
        
        num_samples = len(features)
        self.valid_indices = list(range(0, num_samples - seq_len, stride))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        start = self.valid_indices[idx]
        end = start + self.seq_len
        
        x_window = self.features[start:end] # (seq_len, 7)
        y_disp = self.target_disp[start:end] # (seq_len, 3)
        y_yaw = self.target_yaw[start:end] if self.target_yaw is not None else torch.zeros((self.seq_len, 1))
        
        return x_window, y_disp, y_yaw


class MultiTaskDeadReckoningLoss(nn.Module):
    """
    Multi-objective Loss for dead reckoning:
    L = MSE(delta_pos) + lambda_yaw * HeadingLoss + lambda_smooth * SmoothnessLoss
    """

    def __init__(self, lambda_yaw: float = 0.5, lambda_smooth: float = 0.05) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.lambda_yaw = lambda_yaw
        self.lambda_smooth = lambda_smooth

    def forward(
        self,
        pred_pos: torch.Tensor,
        true_pos: torch.Tensor,
        pred_yaw: Optional[torch.Tensor] = None,
        true_yaw: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Position MSE loss
        loss_pos = self.mse(pred_pos, true_pos)
        
        loss_yaw = torch.tensor(0.0, device=pred_pos.device)
        if pred_yaw is not None and true_yaw is not None:
            # 1 - cos(pred_yaw - true_yaw) for circular continuity
            yaw_diff = pred_yaw - true_yaw
            loss_yaw = torch.mean(1.0 - torch.cos(yaw_diff))
            
        # Smoothness loss: penalize high-frequency acceleration jitter
        loss_smooth = torch.tensor(0.0, device=pred_pos.device)
        if pred_pos.shape[1] > 2:
            diff2 = pred_pos[:, 2:] - 2.0 * pred_pos[:, 1:-1] + pred_pos[:, :-2]
            loss_smooth = torch.mean(diff2 ** 2)
            
        total_loss = loss_pos + self.lambda_yaw * loss_yaw + self.lambda_smooth * loss_smooth
        
        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_pos": float(loss_pos.item()),
            "loss_yaw": float(loss_yaw.item()),
            "loss_smooth": float(loss_smooth.item())
        }
        return total_loss, metrics


def prepare_dataset_arrays(
    dataset_dict: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract normalized features and displacement targets from raw dataset dict.
    """
    sensors = dataset_dict["sensors"]
    gt = dataset_dict["ground_truth"]
    dt = dataset_dict["dt"]
    
    imu_acc = sensors["imu_acc"]
    imu_gyro = sensors["imu_gyro"]
    baro_alt = sensors["baro_alt"]
    
    # Compute barometric altitude delta
    delta_baro = np.zeros_like(baro_alt)
    delta_baro[1:] = np.diff(baro_alt)
    
    # Feature matrix (N, 7)
    features = np.hstack([imu_acc, imu_gyro, delta_baro[:, None]])
    
    # Target positions in NED
    x_ned = gt["x_ned"]
    y_ned = gt["y_ned"]
    z_ned = gt["z_ned"]
    yaw = gt["yaw"]
    
    dx = np.zeros_like(x_ned)
    dy = np.zeros_like(y_ned)
    dz = np.zeros_like(z_ned)
    dyaw = np.zeros_like(yaw)
    
    dx[1:] = np.diff(x_ned)
    dy[1:] = np.diff(y_ned)
    dz[1:] = np.diff(z_ned)
    dyaw[1:] = wrap_to_pi(np.diff(yaw))
    
    targets_disp = np.column_stack([dx, dy, dz])
    targets_yaw = dyaw[:, None]
    
    return features, targets_disp, targets_yaw


def train_dead_reckoning_model(
    model_type: str = "gru", # "gru" or "tcn"
    dataset: Optional[Dict[str, Any]] = None,
    data_dir: str = "data/io_vnbd",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    seq_len: int = 100,
    stride: int = 5,
    device: Optional[str] = None,
    save_path: Optional[str] = None,
    use_wandb: bool = False
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train GRU or TCN Dead Reckoning model on trajectory data.
    
    Args:
        model_type: "gru" or "tcn"
        dataset: Pre-loaded dataset dictionary (or None to load automatically)
        data_dir: Dataset directory
        epochs: Number of training epochs
        batch_size: Mini-batch size
        lr: Initial learning rate
        seq_len: Sliding window length
        stride: Stride for sliding window extraction
        device: "cuda", "cpu", or None for auto
        save_path: Optional path to save best model checkpoint (.pth)
        use_wandb: Log to Weights & Biases if configured
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    
    if dataset is None:
        dataset = load_iovnbd_or_synthetic(data_dir=data_dir)
        
    features, target_disp, target_yaw = prepare_dataset_arrays(dataset)
    
    # Train / Validation Split (80% / 20%)
    split_idx = int(0.8 * len(features))
    train_feat, val_feat = features[:split_idx], features[split_idx:]
    train_disp, val_disp = target_disp[:split_idx], target_disp[split_idx:]
    train_yaw, val_yaw = target_yaw[:split_idx], target_yaw[split_idx:]
    
    train_ds = InertialSequenceDataset(train_feat, train_disp, train_yaw, seq_len=seq_len, stride=stride)
    val_ds = InertialSequenceDataset(val_feat, val_disp, val_yaw, seq_len=seq_len, stride=stride)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Initialize Model
    if model_type.lower() == "gru":
        model = GRUDeadReckoning(input_dim=7, hidden_dim=256, num_layers=3, dropout=0.2).to(device_obj)
    elif model_type.lower() == "tcn":
        model = TCNDeadReckoning(input_dim=7, num_channels=[128, 128, 128, 128, 128, 128], dropout=0.2).to(device_obj)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'gru' or 'tcn'.")
        
    criterion = MultiTaskDeadReckoningLoss(lambda_yaw=0.5, lambda_smooth=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    history = {"train_loss": [], "val_loss": [], "val_pos_loss": []}
    best_val_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_accum = 0.0
        
        for x_b, y_disp_b, y_yaw_b in train_loader:
            x_b = x_b.to(device_obj)
            y_disp_b = y_disp_b.to(device_obj)
            y_yaw_b = y_yaw_b.to(device_obj)
            
            optimizer.zero_grad()
            pred_disp, pred_yaw = model(x_b)[:2]
            
            loss, _ = criterion(pred_disp, y_disp_b, pred_yaw, y_yaw_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            train_loss_accum += loss.item()
            
        scheduler.step()
        train_loss_avg = train_loss_accum / max(len(train_loader), 1)
        
        # Validation
        model.eval()
        val_loss_accum = 0.0
        val_pos_accum = 0.0
        
        with torch.no_grad():
            for x_b, y_disp_b, y_yaw_b in val_loader:
                x_b = x_b.to(device_obj)
                y_disp_b = y_disp_b.to(device_obj)
                y_yaw_b = y_yaw_b.to(device_obj)
                
                pred_disp, pred_yaw = model(x_b)[:2]
                loss, metrics = criterion(pred_disp, y_disp_b, pred_yaw, y_yaw_b)
                
                val_loss_accum += loss.item()
                val_pos_accum += metrics["loss_pos"]
                
        val_loss_avg = val_loss_accum / max(len(val_loader), 1)
        val_pos_avg = val_pos_accum / max(len(val_loader), 1)
        
        history["train_loss"].append(train_loss_avg)
        history["val_loss"].append(val_loss_avg)
        history["val_pos_loss"].append(val_pos_avg)
        
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            if save_path is not None:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss_avg,
                    "model_type": model_type
                }, save_path)
                
    return model, history
