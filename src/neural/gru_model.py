"""
GRU-based Deep Neural Dead Reckoning Model
Problem Statement: SIH26168 (Smart India Hackathon 2026)

Architecture:
    - 3-Layer Gated Recurrent Unit (GRU)
    - Hidden size: 256
    - Dropout: 0.2
    - Input sequence: (batch, seq_len=100, in_features=7)
      [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, delta_baro_alt]
    - Output: delta_position (dx, dy, dz) and optional delta_yaw
    - Fully PyTorch 2.x and CUDA/GPU accelerated
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any


class GRUDeadReckoning(nn.Module):
    """
    3-Layer Recurrent Neural Network for Deep Inertial Odometry and Dead Reckoning.
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        output_dim: int = 3,
        predict_heading: bool = True
    ) -> None:
        """
        Initialize the GRU Dead Reckoning Network.
        
        Args:
            input_dim: Input feature dimensions (default 7: 6-DoF IMU + Baro Delta)
            hidden_dim: Hidden state size in GRU layers (256)
            num_layers: Number of stacked GRU layers (3)
            dropout: Dropout probability between GRU layers (0.2)
            output_dim: Spatial displacement dimensions (3 for dx, dy, dz)
            predict_heading: Whether to predict auxiliary heading delta
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.predict_heading = predict_heading
        
        # Input normalization / linear projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2.0)
        )
        
        # 3-Layer Stacked GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        
        # Output displacement regression head
        self.pos_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
        
        # Optional heading delta regression head
        if self.predict_heading:
            self.yaw_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass for GRU sequence model.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            hidden: Optional initial hidden state (num_layers, batch, hidden_dim)
            
        Returns:
            Tuple of:
                - delta_pos: (batch, seq_len, 3) or (batch, 3) displacement predictions
                - delta_yaw: (batch, seq_len, 1) heading angle delta (if predict_heading=True)
                - next_hidden: updated GRU hidden state
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear feature embedding
        feat = self.input_proj(x) # (batch, seq_len, hidden_dim)
        
        # Recurrent propagation
        gru_out, next_hidden = self.gru(feat, hidden) # (batch, seq_len, hidden_dim)
        
        # Position displacement prediction
        delta_pos = self.pos_head(gru_out) # (batch, seq_len, 3)
        
        delta_yaw = None
        if self.predict_heading:
            delta_yaw = self.yaw_head(gru_out) # (batch, seq_len, 1)
            
        return delta_pos, delta_yaw, next_hidden

    def predict_step(
        self,
        window_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Inference on a single sliding window tensor. Returns the displacement of the last timestep.
        
        Args:
            window_features: Tensor of shape (1, window_len, 7) or (window_len, 7)
            
        Returns:
            Tuple of (delta_pos [3], delta_yaw [1 or None])
        """
        self.eval()
        with torch.no_grad():
            if window_features.ndim == 2:
                window_features = window_features.unsqueeze(0)
            delta_pos, delta_yaw, _ = self.forward(window_features)
            # Take last timestep prediction
            last_pos = delta_pos[0, -1] # (3,)
            last_yaw = delta_yaw[0, -1] if delta_yaw is not None else None
            return last_pos, last_yaw
