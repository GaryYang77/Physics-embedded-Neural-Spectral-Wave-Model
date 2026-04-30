"""Spectrum and spatial mapper building blocks for public inference."""

import torch
from typing import Dict, Any, Optional
from torch import nn
import gin

import waden_hpc.wavetypings as wavetypings

@gin.register
class BaseSpecMapper(nn.Module):
    """
    Base class for wave spectra mappers.
    This class serves as a no-op mapper, passing the input spectra through unchanged.
    """
    
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        spectra_valid: wavetypings.StationSpectra, 
        environdata: wavetypings.EnvironmentDataDict
    ) -> wavetypings.FieldSpectra:
        """
        Apply the mapper to the input wave spectra.
        
        Args:
            spectra: Input wave spectra [B, <mask(H*W)>, D, F]
            environdata: Optional environdata (e.g., wind, bathymetry)
        
        Returns:
            spectra: Mapped wave spectra [B, <mask(H*W)>, D, F]
        """
        return spectra_valid

@gin.register
class BaseSpatialMapper(nn.Module):
    """
    Base class for spatial mappers.
    This class serves as a no-op mapper, passing the input spectra through unchanged.
    """
    
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        spectra: wavetypings.FieldSpectra,
        environdata: wavetypings.EnvironmentDataDict = None,
        mask: Optional[wavetypings.ValidMask] = None
    ) -> wavetypings.FieldSpectra:
        """
        Apply the spatial mapper to the input wave spectra.
        
        Args:
            spectra: Input wave spectra [B, <mask(H*W)>, D, F]
        
        Returns:
            spectra: Mapped wave spectra [B, <mask(H*W)>, D, F]
        """
        return spectra