"""Physics helpers used by the public wave propagation path."""

import torch
from typing import Dict, Any, Optional, Literal, Tuple
from torch import nn
import gin
from abc import ABC, abstractmethod

from . import wavetypings

@gin.configurable
def calculate_depth_dispersion_derivative(
    Cg: torch.Tensor, 
    wavenum: torch.Tensor, 
    freq: torch.Tensor, 
    depth: torch.Tensor,
    eps: float = 1e-9
) -> torch.Tensor:
    """Compute the depth-dispersion derivative without in-place updates."""
    mk_dsdd = depth * wavenum >= 5

    safe_depth = torch.clamp(depth, min=eps)
    DsDd = (Cg * wavenum - 0.5 * freq * 2 * torch.pi) / safe_depth

    DsDd = torch.nan_to_num(DsDd, nan=0.0, posinf=0.0, neginf=0.0)
    DsDd = torch.where(mk_dsdd, 0.0, DsDd)
    DsDd = torch.clamp(DsDd, min=0.0)

    assert DsDd.min() >= 0, 'The value of DsDd should be positive. See line 1047 in w3pro1md.F90'

    return DsDd

@gin.configurable
class WaveNumSolver(nn.Module):
    """Base class for wave number solvers."""
    def __init__(self, max_iter: int = 50, tol: float = 1e-6):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
    
    def forward(
        self, 
        freq: torch.Tensor,
        depth: wavetypings.Depth,
        metadata: wavetypings.MetadataDict = None
    ) -> torch.Tensor:
        """Estimate wave numbers with a binary search solver."""
        g = metadata.get('gravity', 9.806)
        omega = 2 * torch.pi * freq
        depth = depth.unsqueeze(-1)

        k_lower = torch.full_like(freq, 1e-4)
        k_upper = torch.full_like(freq, 10.0)

        for _ in range(self.max_iter):
            k_mid = (k_lower + k_upper) / 2
            kh_mid = k_mid * depth
            f_mid = torch.tanh(kh_mid) - (omega**2 * depth) / (g * kh_mid)

            k_upper = torch.where(f_mid > 0, k_mid, k_upper)
            k_lower = torch.where(f_mid <= 0, k_mid, k_lower)

        k = (k_lower + k_upper) / 2
        k = torch.where(torch.isnan(k), 0.0, k)
        return torch.round(k, decimals=5).squeeze(-2)

@gin.configurable
class GroupVelocityCalculator(nn.Module):
    """Calculate the group velocity of the wave field."""
    def __init__(self, method: str = "standard"):
        super().__init__()
        self.method = method

    def forward(
        self, 
        freq: wavetypings.Freqs,
        wavenum: wavetypings.Wavenumbers,
        depth: wavetypings.Depth,
    ) -> wavetypings.GroupVelocity:
        """Calculate group velocity following the WW3 formulation."""
        sigma = 2 * torch.pi * freq

        safe_wavenum = torch.clamp(wavenum, min=1e-8)
        sinh_term = torch.sinh(2 * wavenum * depth)
        safe_sinh_term = torch.clamp(sinh_term, min=1e-8)

        n = 0.5 + (wavenum * depth) / safe_sinh_term
        Cg = n * (sigma / safe_wavenum)

        Cg = torch.where(wavenum > 1e-7, Cg, 0.0)

        return Cg
