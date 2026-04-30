"""Encoder layers for mapping spectra into the latent wave representation."""
from typing import Tuple, Dict, Any, Optional
import torch
from torch import nn
import gin

import waden_hpc.wavetypings as wavetypings
from neuralwave.interfaces import WaveEncoder
from neuralwave import mappers
from neuralwave import transforms

@gin.configurable
class BaseEncoder(WaveEncoder):
    """
    Base encoder class for encoding wave spectra to a latent representation.
    """
    
    def __init__(
        self, 
        spectra_entransform: transforms.BaseEnTransform = gin.REQUIRED,
        spectra_detransform: transforms.BaseDeTransform = gin.REQUIRED,
        spectra_mapper: mappers.BaseSpecMapper = gin.REQUIRED,
        spatial_mapper: mappers.BaseSpatialMapper = gin.REQUIRED
    ):
        super().__init__()
        self.spectra_entransform = spectra_entransform
        self.spectra_detransform = spectra_detransform
        self.spectra_mapper = spectra_mapper
        self.spatial_mapper = spatial_mapper

    def forward(
        self, 
        spectra: wavetypings.FieldSpectra, 
        environdata: wavetypings.EnvironmentDataDict,
        mask: wavetypings.ValidMask
    ) -> wavetypings.FieldSpectra:
        # [B, H, W, D, F] -> [B, <mask(H*W)>, D, F]
        spectra_valid = self.spectra_entransform(spectra, mask=mask)
        spectra_valid = self.spectra_mapper(
            spectra_valid,
            environdata=environdata
        )
        
        # [B, <mask(H*W)>, D, F] -> [B, H, W, D, F]
        spectra = self.spectra_detransform(spectra_valid, mask=mask)
        spectra = self.spatial_mapper(
            spectra,
            environdata=environdata
        )

        return spectra
