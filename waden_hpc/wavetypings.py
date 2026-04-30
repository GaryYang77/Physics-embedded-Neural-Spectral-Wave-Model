"""Type aliases used by the public wave propagation modules."""

import torch
from typing import Tuple, Dict, Callable


# Basic Variable Definitions
FieldSpectra = torch.Tensor  # [B, H, W, D, F]
StationSpectra = torch.Tensor  # [B, <mask(H*W)>, D, F]
GroupVelocity = torch.Tensor
ValidMask = torch.Tensor  # [B, H, W], mask including boundary and land/sea mask
SeaMask = torch.Tensor  # [B, H, W], mask for sea areas, 1 for sea, 0 for land
BoundaryCondition = torch.Tensor  # [B, H, W, D, F], boundary condition spectra.

StotSpectra = torch.Tensor  # [B, H, W, D, F], source and sinks

SpatialCoords = Tuple[torch.Tensor, torch.Tensor]  # (lon, lat)
SpectraCoords = Tuple[torch.Tensor, torch.Tensor]  # (freqs, wavenumbers)

# Environment data dictionary for additional information: 
# e.g., wind (unnormed, std, mean,...), bathymetry (unnormed, std, mean,...), time, ...
EnvironmentDataDict = Dict[str, torch.Tensor]
MetadataDict = Dict[str, torch.Tensor]  # e.g., baseinfo from gins
LossDict = Dict[str, torch.Tensor]
MetricsDict = Dict[str, float]

# Additional type definitions for gin configuration
Lon = torch.Tensor  # Longitude array
Lat = torch.Tensor  # Latitude array
Theta = torch.Tensor  # Direction array (radians)
Freqs = torch.Tensor  # Frequency array
Wavenumbers = torch.Tensor  # Wavenumber array
Depth = torch.Tensor  # Water depth

# Functional types for various operations
FluxCalculator = Callable[[FieldSpectra, ValidMask], FieldSpectra]
BoundaryUpdater = Callable[[FieldSpectra], FieldSpectra]
TendencyCalculator = Callable[[FieldSpectra, torch.Tensor, float, MetadataDict], FieldSpectra]