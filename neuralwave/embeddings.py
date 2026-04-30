from abc import ABC, abstractmethod
import gin
import torch
from torch import nn
from typing import Dict, Optional

import waden_hpc.wavetypings as wavetypings
from waden_hpc.flux_calculation import get_stencils


class BaseFeatureGenerator(nn.Module, ABC):
    """Base class for feature generators."""
    def __init__(self, spec_shape: tuple, scale_factor: float = 1.0):
        super().__init__()
        self.spec_D, self.spec_F = spec_shape
        initial_embedding = torch.randn(1, 1, self.spec_D, self.spec_F)
        self.register_buffer('embedding', initial_embedding)
        self.scale_factor = scale_factor

    @abstractmethod
    def forward(
        self, 
        spectra_points: wavetypings.StationSpectra,
        environdata: wavetypings.EnvironmentDataDict, 
        metadata: Optional[wavetypings.MetadataDict] = None,
        num_spatial_points: int = None
    ) -> torch.Tensor:
        """Generate feature tensors."""
        raise NotImplementedError
    
@gin.register
class ConstEmbeddings(BaseFeatureGenerator):
    """Use a constant feature map."""
    def __init__(self, spec_shape: tuple = (24, 30), scale_factor: float = 1.0):
        super().__init__(spec_shape)
        self.scale_factor = scale_factor
    
    def forward(
        self, 
        spectra_points: wavetypings.StationSpectra,
        environdata: wavetypings.EnvironmentDataDict, 
        metadata: Optional[wavetypings.MetadataDict] = None,
        num_spatial_points: int = None
    ) -> torch.Tensor:
        weights = torch.ones((num_spatial_points, 1, self.spec_D, self.spec_F), device=self.embedding.device)

        return weights

@gin.register
class BaseWindEmbeddings(BaseFeatureGenerator):
    """Use raw wind components as input features."""
    def __init__(self, spec_shape: tuple = (24, 30), scale_factor: float = 1.0):
        super().__init__(spec_shape)
        self.scale_factor = scale_factor
    
    def forward(
        self, 
        spectra_points: wavetypings.StationSpectra,
        environdata: wavetypings.EnvironmentDataDict, 
        metadata: Optional[wavetypings.MetadataDict] = None,
        num_spatial_points: int = None
    ) -> torch.Tensor:
        wind_u = environdata.get('wind_u').unsqueeze(-1).unsqueeze(-1)  # [b, h, w, 1, 1]
        wind_v = environdata.get('wind_v').unsqueeze(-1).unsqueeze(-1)  # [b, h, w, 1, 1]
        weights_u = wind_u.repeat(1,1,1,self.spec_D,self.spec_F).view(num_spatial_points, 1, self.spec_D, self.spec_F)
        weights_v = wind_v.repeat(1,1,1,self.spec_D,self.spec_F).view(num_spatial_points, 1, self.spec_D, self.spec_F)
        weights = torch.cat([weights_u, weights_v], dim=1)  # [b, 2, d, f]

        return weights

@gin.register
class BaseBathyEmbeddings(BaseFeatureGenerator):
    """Use bathymetry as an input feature."""
    def __init__(self, spec_shape: tuple = (24, 30), scale_factor: float = 100.0):
        super().__init__(spec_shape)
        self.scale_factor = scale_factor

    def forward(
        self, 
        spectra_points: wavetypings.StationSpectra,
        environdata: wavetypings.EnvironmentDataDict, 
        metadata: Optional[wavetypings.MetadataDict] = None,
        num_spatial_points: int = None
    ) -> torch.Tensor:
        bathy = environdata.get('bathymetry').unsqueeze(-1).unsqueeze(-1)  # [b, h, w, 1, 1]
        weights = bathy.repeat(1,1,1,self.spec_D,self.spec_F).view(num_spatial_points, 1, self.spec_D, self.spec_F)  # [b, 1, d, f]

        weights = weights / self.scale_factor

        return weights

@gin.register
class WindEmbeddings(BaseFeatureGenerator):
    """Project wind features onto the spectral direction grid."""
    def __init__(self, spec_shape: tuple = (24, 30), scale_factor: float = 20.0):
        super().__init__(spec_shape)
        self.scale_factor = scale_factor

    def forward(
        self, 
        spectra_points: wavetypings.StationSpectra,
        environdata: wavetypings.EnvironmentDataDict, 
        metadata: Optional[wavetypings.MetadataDict] = None,
        num_spatial_points: int = None
    ) -> torch.Tensor:
        theta_reg = metadata.get('theta') * 180.0 / torch.pi
        theta_rad = ((180 - theta_reg) % 360) * torch.pi / 180.0
        wind_u = environdata.get('wind_u').unsqueeze(-1).unsqueeze(-1)  # [b, h, w, 1, 1]
        wind_v = environdata.get('wind_v').unsqueeze(-1).unsqueeze(-1)  # [b, h, w, 1, 1]
        U_proj = wind_u * torch.sin(theta_rad) + wind_v * torch.cos(theta_rad)
        weights = U_proj * metadata.get('Cg')

        weights = weights / self.scale_factor

        return weights.view(num_spatial_points, 1, self.spec_D, self.spec_F)

@gin.register
class NewWindEmbeddings(BaseFeatureGenerator):
    """Alternative wind projection feature map."""
    def __init__(self, spec_shape: tuple = (24, 30), scale_factor: float = 0.5):
        super().__init__(spec_shape)
        self.scale_factor = scale_factor

    def forward(
        self, 
        spectra_points: wavetypings.StationSpectra,
        environdata: wavetypings.EnvironmentDataDict, 
        metadata: Optional[wavetypings.MetadataDict] = None,
        num_spatial_points: int = None
    ) -> torch.Tensor:
        theta_reg = metadata.get('theta') * 180.0 / torch.pi
        theta_rad = ((180 - theta_reg) % 360) * torch.pi / 180.0
        wind_u = environdata.get('wind_u').unsqueeze(-1).unsqueeze(-1)  # [b, h, w, 1, 1]
        wind_v = environdata.get('wind_v').unsqueeze(-1).unsqueeze(-1)  # [b, h, w, 1, 1]
        U_proj = wind_u * torch.sin(theta_rad) + wind_v * torch.cos(theta_rad)
        weights = U_proj / metadata.get('Cg')

        weights = weights / self.scale_factor

        return weights.view(num_spatial_points, 1, self.spec_D, self.spec_F)


@gin.register
class SdnEmbeddings(BaseFeatureGenerator):
    """Bathymetry-gradient feature map for dissipation-related effects."""
    def __init__(self, spec_shape: tuple = (24, 30), scale_factor: float = 20.0):
        super().__init__(spec_shape)
        self.scale_factor = scale_factor
        
    def forward(
        self, 
        spectra_points: wavetypings.StationSpectra,
        environdata: wavetypings.EnvironmentDataDict, 
        metadata: Optional[wavetypings.MetadataDict] = None,
        num_spatial_points: int = None
    ) -> torch.Tensor:
        depth = environdata.get('bathymetry')
        depth_3d = depth.squeeze(-1).squeeze(-1)
        depthx_3d = nn.functional.pad(depth_3d.permute(0, 2, 1), (1,1), mode='circular').permute(0, 2, 1)
        depthy_3d = nn.functional.pad(depth_3d, (1,1), mode='replicate')
        depthx = depthx_3d.unsqueeze(-1).unsqueeze(-1)
        depthy = depthy_3d.unsqueeze(-1).unsqueeze(-1)
        DdDx = 0.5 * ((depthx[:, 2:, ...] - depthx[:, 1:-1, ...]) + (depthx[:, 1:-1, ...] - depthx[:, :-2, ...])) / metadata['Dlon']
        DdDy = 0.5 * ((depthy[:, :, 2:, ...] - depthy[:, :, 1:-1, ...]) + (depthy[:, :, 1:-1, ...] - depthy[:, :, :-2, ...])) / metadata['Dlat']
        theta_reg = metadata.get('theta') * 180.0 / torch.pi
        theta_rad = ((180 - theta_reg) % 360) * torch.pi / 180.0

        bathy_proj = DdDx * torch.sin(theta_rad) + DdDy * torch.cos(theta_rad)
        bathy_proj = bathy_proj * metadata.get('DFreq')[None,None,None,None,:]

        weights = bathy_proj.reshape(num_spatial_points, 1, self.spec_D, self.spec_F) * spectra_points

        weights = weights / self.scale_factor

        return weights

@gin.register
class SdnEmbeddings2(BaseFeatureGenerator):
    """Bathymetry-gradient feature map without spectrum weighting."""
    def __init__(self, spec_shape: tuple = (24, 30), scale_factor: float = 20.0):
        super().__init__(spec_shape)
        self.scale_factor = scale_factor
        
    def forward(
        self, 
        spectra_points: wavetypings.StationSpectra,
        environdata: wavetypings.EnvironmentDataDict, 
        metadata: Optional[wavetypings.MetadataDict] = None,
        num_spatial_points: int = None
    ) -> torch.Tensor:
        depth = environdata.get('bathymetry')
        depth_3d = depth.squeeze(-1).squeeze(-1)
        depthx_3d = nn.functional.pad(depth_3d.permute(0, 2, 1), (1,1), mode='circular').permute(0, 2, 1)
        depthy_3d = nn.functional.pad(depth_3d, (1,1), mode='replicate')
        depthx = depthx_3d.unsqueeze(-1).unsqueeze(-1)
        depthy = depthy_3d.unsqueeze(-1).unsqueeze(-1)
        DdDx = 0.5 * ((depthx[:, 2:, ...] - depthx[:, 1:-1, ...]) + (depthx[:, 1:-1, ...] - depthx[:, :-2, ...])) / metadata['Dlon']
        DdDy = 0.5 * ((depthy[:, :, 2:, ...] - depthy[:, :, 1:-1, ...]) + (depthy[:, :, 1:-1, ...] - depthy[:, :, :-2, ...])) / metadata['Dlat']
        theta_reg = metadata.get('theta') * 180.0 / torch.pi
        theta_rad = ((180 - theta_reg) % 360) * torch.pi / 180.0

        bathy_proj = DdDx * torch.sin(theta_rad) + DdDy * torch.cos(theta_rad)
        bathy_proj = bathy_proj * metadata.get('DFreq')[None,None,None,None,:]

        weights = bathy_proj.reshape(num_spatial_points, 1, self.spec_D, self.spec_F)

        weights = weights / self.scale_factor

        return weights

@gin.register
class WwInterEmbeddings(BaseFeatureGenerator):
    """Feature map for wave-wave interaction cues."""
    def __init__(self, spec_shape: tuple = (24, 30), scale_factor: float = 1.0):
        super().__init__(spec_shape)
        self.scale_factor = scale_factor
        
    def forward(
        self, 
        spectra_points: wavetypings.StationSpectra,
        environdata: wavetypings.EnvironmentDataDict, 
        metadata: Optional[wavetypings.MetadataDict] = None,
        num_spatial_points: int = None
    ) -> torch.Tensor:
        wave_num = metadata.get('wavenum').repeat(1,1,1,self.spec_D,1)  # [b, h, w, d, f]
        wave_freq = metadata.get('freq').expand_as(wave_num)  # [b, h, w, d, f]
        weights = wave_num + wave_freq

        weights = weights / self.scale_factor

        return weights.view(num_spatial_points, 1, self.spec_D, self.spec_F)
    
@gin.register
class NeighborLandEmbeddings(BaseFeatureGenerator):
    """Feature map based on nearby land points."""
    def __init__(self, spec_shape: tuple = (24, 30), scale_factor: float = 1.0):
        super().__init__(spec_shape)
        self.scale_factor = scale_factor
    
    def forward(
        self, 
        spectra_points: wavetypings.StationSpectra,
        environdata: wavetypings.EnvironmentDataDict, 
        metadata: Optional[wavetypings.MetadataDict] = None,
        num_spatial_points: int = None
    ) -> torch.Tensor:
        depth = environdata.get('depth')
        land_mask = depth.unsqueeze(-1).unsqueeze(-1) > 0   # True for Sea, False for Land
        lm_im1, lm_i, lm_ip1 = get_stencils(land_mask.float(), 1, pad_left=1, pad_right=1, mode='replicate')
        neighbor_land_mask_x = ((lm_i == 0).float() + (lm_im1 == 0).float() + (lm_ip1 == 0).float())
        lm_im1, lm_i, lm_ip1 = get_stencils(land_mask.float(), 2, pad_left=1, pad_right=1, mode='replicate')
        neighbor_land_mask_y = ((lm_i == 0).float() + (lm_im1 == 0).float() + (lm_ip1 == 0).float())
        neighbor_land_mask = neighbor_land_mask_x + neighbor_land_mask_y
        neighbor_land_mask[~land_mask] = -1  # Land points set to -1
        weights = neighbor_land_mask.repeat(1,1,1,self.spec_D,self.spec_F)

        weights = weights / self.scale_factor

        return weights.view(num_spatial_points, 1, self.spec_D, self.spec_F)
    
@gin.register
class WindEmbeddings_Vec(BaseFeatureGenerator):
    """Directional wind feature vector."""
    def __init__(self, spec_shape: tuple = (24, 30), scale_factor: float = 1.0):
        super().__init__(spec_shape)
        self.scale_factor = scale_factor

    def forward(
        self, 
        spectra_points: wavetypings.StationSpectra,
        environdata: wavetypings.EnvironmentDataDict, 
        metadata: Optional[wavetypings.MetadataDict] = None,
        num_spatial_points: int = None
    ) -> torch.Tensor:
        theta_reg = metadata.get('theta') * 180.0 / torch.pi
        theta_rad = ((180 - theta_reg) % 360) * torch.pi / 180.0
        wind_u = environdata.get('wind_u').unsqueeze(-1).unsqueeze(-1)  # [b, h, w, 1, 1]
        wind_v = environdata.get('wind_v').unsqueeze(-1).unsqueeze(-1)  # [b, h, w, 1, 1]
        U_proj = wind_u * torch.sin(theta_rad) + wind_v * torch.cos(theta_rad)
        
        weights = U_proj / self.scale_factor

        return weights.reshape(num_spatial_points, 1, self.spec_D, 1)

@gin.register
class SdnEmbeddings_Vec(BaseFeatureGenerator):
    """Vectorized bathymetry-gradient feature map."""
    def __init__(self, spec_shape: tuple = (24, 30), scale_factor: float = 10000.0):
        super().__init__(spec_shape)
        self.scale_factor = scale_factor
        
    def forward(
        self, 
        spectra_points: wavetypings.StationSpectra,
        environdata: wavetypings.EnvironmentDataDict, 
        metadata: Optional[wavetypings.MetadataDict] = None,
        num_spatial_points: int = None
    ) -> torch.Tensor:
        depth = environdata.get('bathymetry')
        depth_3d = depth.squeeze(-1).squeeze(-1)
        depthx_3d = nn.functional.pad(depth_3d.permute(0, 2, 1), (1,1), mode='circular').permute(0, 2, 1)
        depthy_3d = nn.functional.pad(depth_3d, (1,1), mode='replicate')
        depthx = depthx_3d.unsqueeze(-1).unsqueeze(-1)
        depthy = depthy_3d.unsqueeze(-1).unsqueeze(-1)
        DdDx = 0.5 * ((depthx[:, 2:, ...] - depthx[:, 1:-1, ...]) + (depthx[:, 1:-1, ...] - depthx[:, :-2, ...])) / metadata['Dlon']
        DdDy = 0.5 * ((depthy[:, :, 2:, ...] - depthy[:, :, 1:-1, ...]) + (depthy[:, :, 1:-1, ...] - depthy[:, :, :-2, ...])) / metadata['Dlat']
        theta_reg = metadata.get('theta') * 180.0 / torch.pi
        theta_rad = ((180 - theta_reg) % 360) * torch.pi / 180.0

        bathy_proj = DdDx * torch.sin(theta_rad) + DdDy * torch.cos(theta_rad)

        weights = bathy_proj / self.scale_factor

        return weights.reshape(num_spatial_points, 1, self.spec_D, 1)

@gin.register
class WindEmbeddings_Vec2(BaseFeatureGenerator):
    """Two-channel wind feature vector."""
    def __init__(self, spec_shape: tuple = (24, 30), scale_factor: float = 1.0):
        super().__init__(spec_shape)
        self.scale_factor = scale_factor

    def forward(
        self, 
        spectra_points: wavetypings.StationSpectra,
        environdata: wavetypings.EnvironmentDataDict, 
        metadata: Optional[wavetypings.MetadataDict] = None,
        num_spatial_points: int = None
    ) -> torch.Tensor:
        theta_reg = metadata.get('theta') * 180.0 / torch.pi
        theta_rad = ((180 - theta_reg) % 360) * torch.pi / 180.0
        wind_u = environdata.get('wind_u').unsqueeze(-1)  # [b, h, w, 1]
        wind_v = environdata.get('wind_v').unsqueeze(-1)  # [b, h, w, 1]
        U_proj = torch.stack([wind_u, wind_v], dim=-1).unsqueeze(-1)  # [b, h, w, 1, 2, 1]
        
        weights = U_proj / self.scale_factor

        return weights.reshape(num_spatial_points, 1, 2, 1)
    
@gin.register
class SdnEmbeddings_Vec2(BaseFeatureGenerator):
    """Bathymetry-only vector feature map."""
    def __init__(self, spec_shape: tuple = (24, 30), scale_factor: float = 500.0):
        super().__init__(spec_shape)
        self.scale_factor = scale_factor
        
    def forward(
        self, 
        spectra_points: wavetypings.StationSpectra,
        environdata: wavetypings.EnvironmentDataDict, 
        metadata: Optional[wavetypings.MetadataDict] = None,
        num_spatial_points: int = None
    ) -> torch.Tensor:
        depth = environdata.get('bathymetry')

        weights = depth / self.scale_factor

        return weights.reshape(num_spatial_points, 1, 1, 1)

@gin.register
class WindEmbeddings_noCg(BaseFeatureGenerator):
    """Wind projection feature map without group velocity weighting."""
    def __init__(self, spec_shape: tuple = (24, 30), scale_factor: float = 20.0):
        super().__init__(spec_shape)
        self.scale_factor = scale_factor

    def forward(
        self, 
        spectra_points: wavetypings.StationSpectra,
        environdata: wavetypings.EnvironmentDataDict, 
        metadata: Optional[wavetypings.MetadataDict] = None,
        num_spatial_points: int = None
    ) -> torch.Tensor:
        theta_reg = metadata.get('theta') * 180.0 / torch.pi
        theta_rad = ((180 - theta_reg) % 360) * torch.pi / 180.0
        wind_u = environdata.get('wind_u').unsqueeze(-1).unsqueeze(-1)  # [b, h, w, 1, 1]
        wind_v = environdata.get('wind_v').unsqueeze(-1).unsqueeze(-1)  # [b, h, w, 1, 1]
        U_proj = wind_u * torch.sin(theta_rad) + wind_v * torch.cos(theta_rad)
        weights = U_proj * torch.ones_like(metadata.get('Cg'))

        weights = weights / self.scale_factor

        return weights.view(num_spatial_points, 1, self.spec_D, self.spec_F)

@gin.register
class SdnEmbeddings_OnlyDepth(BaseFeatureGenerator):
    """Bathymetry-only feature map on the spectral grid."""
    def __init__(self, spec_shape: tuple = (24, 30), scale_factor: float = 10000.0):
        super().__init__(spec_shape)
        self.scale_factor = scale_factor
        
    def forward(
        self, 
        spectra_points: wavetypings.StationSpectra,
        environdata: wavetypings.EnvironmentDataDict, 
        metadata: Optional[wavetypings.MetadataDict] = None,
        num_spatial_points: int = None
    ) -> torch.Tensor:
        depth = environdata.get('bathymetry')
        bathy_proj = depth.squeeze(-1).squeeze(-1).repeat(1,1,1,self.spec_D,self.spec_F) # [b, h, w, d, f]

        weights = bathy_proj.reshape(num_spatial_points, 1, self.spec_D, self.spec_F)

        weights = weights / self.scale_factor

        return weights
    
@gin.register
class WindEmbeddings_noCg_Square(BaseFeatureGenerator):
    """Wind projection feature map with squared magnitude emphasis."""
    def __init__(self, spec_shape: tuple = (24, 30), scale_factor: float = 10.0):
        super().__init__(spec_shape)
        self.scale_factor = scale_factor

    def forward(
        self, 
        spectra_points: wavetypings.StationSpectra,
        environdata: wavetypings.EnvironmentDataDict, 
        metadata: Optional[wavetypings.MetadataDict] = None,
        num_spatial_points: int = None
    ) -> torch.Tensor:
        theta_reg = metadata.get('theta') * 180.0 / torch.pi
        theta_rad = ((180 - theta_reg) % 360) * torch.pi / 180.0
        wind_u = environdata.get('wind_u').unsqueeze(-1).unsqueeze(-1)  # [b, h, w, 1, 1]
        wind_v = environdata.get('wind_v').unsqueeze(-1).unsqueeze(-1)  # [b, h, w, 1, 1]
        U_proj = wind_u * torch.sin(theta_rad) + wind_v * torch.cos(theta_rad)
        weights = U_proj * torch.ones_like(metadata.get('Cg')) * U_proj.abs()

        weights = weights / self.scale_factor

        return weights.view(num_spatial_points, 1, self.spec_D, self.spec_F)
