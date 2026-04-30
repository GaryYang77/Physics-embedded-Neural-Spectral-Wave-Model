"""Top-level NeuralWave model architecture used by public inference."""

from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List, Callable
import math
import gin

from waden_hpc import wavetypings
from waden_hpc import integrator
from waden_hpc import physics_calculation
from waden_hpc import boundary_process

from neuralwave import encoders
from neuralwave import decoders
from neuralwave import parameterizations 

@gin.configurable('NeuralWave')
class ModelArchitecture(nn.Module):
    """Base class for model architectures in NeuralWave."""
    
    def __init__(
        self, 
        config: Any,
        encoder: encoders.BaseEncoder = gin.REQUIRED,
        parameterization: parameterizations.BaseParameterization = gin.REQUIRED,
        integrator: integrator.ModelAdvance = gin.REQUIRED,
        decoder: decoders.BaseDecoder = gin.REQUIRED,
        wavenum_solver: physics_calculation.WaveNumSolver = gin.REQUIRED,
        group_velocity_calculator: physics_calculation.GroupVelocityCalculator = gin.REQUIRED,
        boundary_processor: boundary_process.update_absorb_boundary = gin.REQUIRED,
        apply_boundary_func: Callable = gin.REQUIRED
    ):
        super().__init__()
        self.config = config
        self.metadata = config.get('metadata', {})

        # Configure components through gin
        self.encoder = encoder
        self.parameterization = parameterization
        self.integrator = integrator
        self.decoder = decoder
        
        # Physics calculation components
        self.wavenum_solver = wavenum_solver
        self.group_velocity_calculator = group_velocity_calculator
        self.boundary_processor = boundary_processor
        self.apply_boundary_func = apply_boundary_func
    
    def to(self, *args, **kwargs):
        """
        Overrides the nn.Module.to() method to also move tensors in self.metadata.
        """
        model_on_device = super().to(*args, **kwargs)
        for key, value in model_on_device.metadata.items():
            if isinstance(value, torch.Tensor):
                model_on_device.metadata[key] = value.to(*args, **kwargs)
        return model_on_device

    def set_wave_environment(
        self, 
        lon: wavetypings.Lon,
        lat: wavetypings.Lat,
        theta: wavetypings.Theta,
        freq: wavetypings.Freqs
    ):
        """Set the wave environment data."""
        self.metadata['lon'] = lon.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.metadata['lat'] = lat.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.metadata['theta'] = theta.unsqueeze(-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.metadata['freq'] = freq.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        if self.metadata['coordinate_type'] == 'Spherical':
            self.metadata['lon'] = self.metadata['lon'] / 180 * torch.pi
            self.metadata['lat'] = self.metadata['lat'] / 180 * torch.pi

        self.metadata['Ntheta'] = len(theta)
        self.metadata['Nk'] = len(freq)
        self.metadata['DTH'] = torch.tensor(2 * torch.pi / self.metadata['Ntheta'], dtype=torch.float32)  # If the theta distributed uniformly.
        freqc = torch.nn.functional.pad(freq, (0, 1), value=0)
        freqc[..., -1] = 1  # Pad the last frequency with 1, a parameter still to be tuned.
        self.metadata['DFreq'] = torch.diff(freqc, dim=-1)

        lonc = torch.cat((self.metadata['lon'][:, -2:-1], self.metadata['lon'], self.metadata['lon'][:, 0:1]), dim=1)
        latc = torch.cat((self.metadata['lat'][:, :, -2:-1], self.metadata['lat'], self.metadata['lat'][:, :, 0:1]), dim=2)
        self.metadata['Dlon'] = torch.abs(0.5 * (torch.diff(lonc[:, 1:], dim=1) + torch.diff(lonc[:, :-1], dim=1)))
        self.metadata['Dlat'] = torch.abs(0.5 * (torch.diff(latc[:, :, 1:], dim=2) + torch.diff(latc[:, :, :-1], dim=2)))
        self.metadata['Dlon'][:, 0] = self.metadata['Dlon'][:, 1]
        self.metadata['Dlon'][:, -1] = self.metadata['Dlon'][:, -2]
        self.metadata['Dlat'][:, :, 0] = self.metadata['Dlat'][:, :, 1]
        self.metadata['Dlat'][:, :, -1] = self.metadata['Dlat'][:, :, -2]
        
        self.metadata['sinTheta'] = torch.round(torch.sin(self.metadata['theta']) * 1e+5) * 1e-5
        self.metadata['cosTheta'] = torch.round(torch.cos(self.metadata['theta']) * 1e+5) * 1e-5
        self.metadata['cosLAT'] = torch.round(torch.cos(self.metadata['lat']) * 1e+5) * 1e-5

    def forward(
        self,
        spectra: wavetypings.FieldSpectra,
        environments: wavetypings.EnvironmentDataDict,
        mask: wavetypings.ValidMask,
        boundary_condition: wavetypings.BoundaryCondition,
        **kwargs
    ) -> wavetypings.FieldSpectra:
        """Forward pass through the model architecture.
        
        Framework: input -> prepare physics -> encoder -> model advance -> decoder -> output
        
        Args:
            spectra: Input wave spectra [B, H, W, D, F]
            environments: Environment data dictionary (wind, depth, etc.)
            mask: Valid mask for sea/land areas [B, H, W]
            
        Returns:
            Updated wave spectra [B, H, W, D, F]
        """
        # Step 1: Prepare physics - calculate wavenumbers and group velocity
        freq = self.metadata.get('freq')
        depth = environments.get('depth').unsqueeze(-1).unsqueeze(-1)

        if freq is None or depth is None:
            raise ValueError("Frequency and depth must be provided in metadata and environments")
            
        # Calculate wavenumbers and group velocity
        wavenum = self.wavenum_solver(
            freq=freq, 
            depth=depth, 
            metadata=self.metadata
        )
        
        group_velocity = self.group_velocity_calculator(
            freq=freq,
            wavenum=wavenum,
            depth=depth
        )
        
        # Update metadata with calculated physics
        physics_metadata = self.metadata.copy()
        # wavenum[~mask.expand_as(wavenum)] = 0
        # group_velocity[~mask.expand_as(group_velocity)] = 0
        physics_metadata.update({
            'wavenum': wavenum,
            'Cg': group_velocity
        })
        
        # Step 2: Encoder - enhance spectral representation
        enhanced_spectra = self.encoder(
            spectra=spectra,
            environdata=environments,
            mask=mask
        )
        
        # Step 3: Parameterization - calculate source/sink terms
        stot = self.parameterization(
            spectra=enhanced_spectra,
            environdata=environments,
            metadata=physics_metadata,
            mask=mask,
            **kwargs
        )

        # Step 4: Model advance - integrate dynamics
        integrated_spectra = self.integrator(
            spectra=enhanced_spectra,
            stot=stot,
            depth=depth,
            metadata=physics_metadata,
            mask=mask,
            boundary_condition=boundary_condition,
            **kwargs
        )
        
        # Step 5: Decoder - map back to output representation
        output_spectra = self.decoder(
            spectra=integrated_spectra,
            environdata=environments,
            mask=mask
        )
        output_spectra = self.apply_boundary_func(output_spectra, boundary_condition, mask)

        return output_spectra
    
    def get_physics_info(
        self,
        environments: wavetypings.EnvironmentDataDict
    ) -> Dict[str, torch.Tensor]:
        """Get physics information for debugging or analysis."""
        freq = self.metadata.get('freq')
        depth = environments.get('depth')
        
        wavenum = self.wavenum_solver(
            freq=freq, 
            depth=depth, 
            metadata=self.metadata
        )
        
        group_velocity = self.group_velocity_calculator(
            freq=freq,
            wavenum=wavenum,
            depth=depth
        )
        
        return {
            'wavenum': wavenum,
            'group_velocity': group_velocity,
            'freq': freq,
            'depth': depth
        }

