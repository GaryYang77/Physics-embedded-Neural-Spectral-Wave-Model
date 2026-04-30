"""Spectral transform helpers used by the public inference path."""

import torch
from typing import Dict, Any, Optional
from torch import nn
import gin

import waden_hpc.wavetypings as wavetypings

@gin.register
class NoEnTransform(nn.Module):
    """
    This class serves as a no-op transformation, passing the input spectra through unchanged.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        spectra: wavetypings.FieldSpectra, 
        mask: wavetypings.ValidMask
    ) -> wavetypings.StationSpectra:
        """
        Apply the transformation to the input wave spectra.
        Args:
            spectra: Input wave spectra [B, H, W, D, F]
            mask: Optional valid mask [B, H, W]
        Returns:
            spectra: Transformed wave spectra [B, <mask(H*W)>, D, F]
        """
        b, h, w, d, f = spectra.shape
        spectra = spectra.view(b, h * w, d, f)
        return spectra

@gin.register
class NoDeTransform(nn.Module):
    """
    This class serves as a no-op de-transformation, passing the input spectra through unchanged.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        spectra_valid: wavetypings.StationSpectra, 
        mask: wavetypings.ValidMask
    ) -> wavetypings.FieldSpectra:
        """
        Apply the de-transformation to the input wave spectra.
        Args:
            spectra: Input wave spectra [B, <mask(H*W)>, D, F]
            mask: Optional valid mask [B, H, W]
        Returns:
            spectra: De-transformed wave spectra [B, H, W, D, F]
        """
        b, h, w  = mask.shape
        d, f = spectra_valid.shape[-2:]
        spectra = spectra_valid.view(b, h, w, d, f)
        return spectra

@gin.register
class BaseEnTransform(nn.Module):
    """
    Base class for wave spectra transformations.
    This class serves as a no-op transformation, passing the input spectra through unchanged.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        spectra: wavetypings.FieldSpectra, 
        mask: wavetypings.ValidMask
    ) -> wavetypings.StationSpectra:
        """
        Applies the mask to the input spectra but does NOT change its shape.
        Args:
            spectra: Input wave spectra [B, H, W, D, F]
            mask: Optional valid mask [B, H, W]
        Returns:
            spectra: Masked wave spectra, still of shape [B, H, W, D, F]
        """
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)
        return spectra * mask_expanded

@gin.register
class BaseDeTransform(nn.Module):
    """
    Base class for wave spectra de-transformations.
    This class serves as a no-op de-transformation, passing the input spectra through unchanged.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        spectra: wavetypings.FieldSpectra, 
        mask: wavetypings.ValidMask
    ) -> wavetypings.FieldSpectra:
        """
        In the new workflow, this function is mostly a pass-through.
        We can re-apply the mask to ensure correctness.
        Args:
            spectra: Input wave spectra [B, H, W, D, F]
            mask: Optional valid mask [B, H, W]
        Returns:
            spectra: Masked wave spectra [B, H, W, D, F]
        """
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)
        return spectra * mask_expanded