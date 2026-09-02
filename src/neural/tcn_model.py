"""
Temporal Convolutional Network (TCN) for Inertial Dead Reckoning
Problem Statement: SIH26168 (Smart India Hackathon 2026)

Architecture:
    - Dilated Causal 1D Convolutions strictly preventing future information leakage
    - Dilation schedule: [1, 2, 4, 8, 16] (with residual multi-block stacking)
    - Receptive field: >= 200 timesteps
    - Weight Normalization / BatchNorm + GELU activations + Residual Skip Connections
    - Output: delta_position (dx, dy, dz) and delta_yaw per timestep
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any


class Chomp1d(nn.Module):
    """
    Trims the right padding of a 1D convolution to enforce causality.
    """

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Residual block containing two dilated causal convolution layers with Chomp1d and Dropout.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2
        )
        
        # 1x1 conv residual projection if input and output dimensions differ
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Multi-layer Temporal Convolutional Network backbone.
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.2
    ) -> None:
        super().__init__()
        
        if dilations is None:
            dilations = [1, 2, 4, 8, 16]
            
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = dilations[i % len(dilations)]
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size, padding=padding, dropout=dropout
                )
            )
            
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, seq_len)
        return self.network(x)


class TCNDeadReckoning(nn.Module):
    """
    Full TCN Dead Reckoning Network for vehicle displacement estimation.
    """

    def __init__(
        self,
        input_dim: int = 7,
        num_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.2,
        output_dim: int = 3,
        predict_heading: bool = True
    ) -> None:
        """
        Initialize TCN Dead Reckoning Network.
        
        Receptive Field calculation:
        RF = 1 + sum_{i}( 2 * (kernel_size - 1) * dilation_i )
        With dilations [1, 2, 4, 8, 16, 32, 64] or stacked [1,2,4,8,16]x2 with k=3, RF > 250 timesteps.
        """
        super().__init__()
        
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 64]
            
        if num_channels is None:
            num_channels = [128, 128, 128, 128, 128, 128, 128]
            
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.predict_heading = predict_heading
        
        # Calculate theoretical receptive field
        rf = 1 + sum(2 * (kernel_size - 1) * d for d in dilations[:len(num_channels)])
        self.receptive_field = rf
        
        self.tcn = TemporalConvNet(
            num_inputs=input_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout
        )
        
        feature_dim = num_channels[-1]
        self.pos_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
        
        if predict_heading:
            self.yaw_head = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            )

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for TCN.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Tuple of:
                - delta_pos: (batch, seq_len, 3) displacement predictions
                - delta_yaw: (batch, seq_len, 1) heading angle delta (if enabled)
        """
        # Permute for 1D convolution: (batch, input_dim, seq_len)
        x_trans = x.transpose(1, 2)
        
        features = self.tcn(x_trans) # (batch, channels, seq_len)
        features = features.transpose(1, 2) # (batch, seq_len, channels)
        
        delta_pos = self.pos_head(features)
        delta_yaw = self.yaw_head(features) if self.predict_heading else None
        
        return delta_pos, delta_yaw

    def predict_step(
        self,
        window_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run inference on a single sliding window tensor.
        """
        self.eval()
        with torch.no_grad():
            if window_features.ndim == 2:
                window_features = window_features.unsqueeze(0)
            delta_pos, delta_yaw = self.forward(window_features)
            last_pos = delta_pos[0, -1]
            last_yaw = delta_yaw[0, -1] if delta_yaw is not None else None
            return last_pos, last_yaw
