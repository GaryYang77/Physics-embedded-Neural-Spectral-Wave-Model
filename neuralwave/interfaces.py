"""
Core interfaces and abstract base classes for NeuralWave components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import gin

import waden_hpc.wavetypings as wavetypings


class WaveEncoder(nn.Module, ABC):
    """
    Abstract base class for wave encoders.

    Encoders should be decorated with @gin.register to enable
    flexible configuration of architecture parameters.
    """
    
    @abstractmethod
    def forward(
        self,
        spectra: wavetypings.FieldSpectra,
        environment: wavetypings.EnvironmentDataDict = None,
        mask: Optional[wavetypings.ValidMask] = None
    ) -> wavetypings.FieldSpectra:
        """
        Encode wave spectra to latent representation.
        
        Args:
            spectra: Wave spectra [B, H, W, D, F]
            environment: Optional environment (wind, bathymetry, etc.)
            mask: Optional valid mask [B, H, W]
            
        Returns:
            spectra: Latent spectra [B, H, W, D, F]
        """
        pass

class WaveDecoder(nn.Module, ABC):
    """Abstract base class for wave decoders."""
    
    @abstractmethod
    def forward(
        self, 
        latent: torch.Tensor,
        target_shape: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        """
        Decode latent representation to wave spectra.
        
        Args:
            spectra: Wave spectra [B, H, W, D, F]
            environment: Optional environment (wind, bathymetry, etc.)
            mask: Optional valid mask [B, H, W]
            
        Returns:
            spectra: Latent spectra [B, H, W, D, F]
        """
        pass


class WavePreprocessor(nn.Module, ABC):
    """Abstract base class for wave data preprocessors."""
    
    @abstractmethod
    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """Preprocess raw wave data."""
        pass
    
    @abstractmethod
    def postprocess(self, data: torch.Tensor) -> torch.Tensor:
        """Postprocess model output."""
        pass


class PhysicsConstraint(nn.Module, ABC):
    """Abstract base class for physics constraints."""
    
    @abstractmethod
    def apply_constraint(self, spectra: torch.Tensor) -> torch.Tensor:
        """Apply physics constraint to wave spectra."""
        pass
    
    @abstractmethod
    def compute_physics_loss(self, predicted: torch.Tensor, 
                           target: torch.Tensor) -> torch.Tensor:
        """Compute physics-based loss."""
        pass


class WaveLoss(nn.Module, ABC):
    """Abstract base class for wave-specific loss functions."""
    
    @abstractmethod
    def forward(self, predicted: torch.Tensor, 
                target: torch.Tensor,
                environment: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Compute loss between predicted and target."""
        pass


class WaveMetrics(ABC):
    """Abstract base class for wave-specific metrics."""
    
    @abstractmethod
    def compute(self, predicted: torch.Tensor, 
                target: torch.Tensor) -> Dict[str, float]:
        """Compute metrics between predicted and target."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset metric state."""
        pass


class WaveDataProcessor(ABC):
    """Abstract base class for data processing pipelines."""
    
    @abstractmethod
    def load_data(self, data_path: str, **kwargs) -> Any:
        """Load wave data from file."""
        pass
    
    @abstractmethod
    def create_batches(self, data: Any, batch_size: int) -> Any:
        """Create training batches from data."""
        pass
    
    @abstractmethod
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the loaded data."""
        pass


class ModelArchitecture(nn.Module, ABC):
    """
    Abstract base class for complete model architectures.
    
    Should be decorated with @gin.configurable to enable flexible
    configuration of the complete model pipeline.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder: Optional[WaveEncoder] = None
        self.decoder: Optional[WaveDecoder] = None
        self.preprocessor: Optional[WavePreprocessor] = None
        self.physics_constraint: Optional[PhysicsConstraint] = None
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through the complete architecture."""
        pass

    def get_components(self) -> Dict[str, nn.Module]:
        """Get all model components."""
        return {
            "encoder": self.encoder,
            "decoder": self.decoder, 
            "preprocessor": self.preprocessor,
            "physics_constraint": self.physics_constraint
        }


class WaveTrainer(ABC):
    """Abstract base class for model trainers."""
    
    @abstractmethod
    def train_epoch(self, dataloader: Any) -> Dict[str, float]:
        """Train for one epoch."""
        pass
    
    @abstractmethod
    def validate_epoch(self, dataloader: Any) -> Dict[str, float]:
        """Validate for one epoch."""
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str, **kwargs) -> None:
        """Save training checkpoint."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load training checkpoint."""
        pass
