"""Wave tendency operators used by the public inference path."""

import torch
from typing import Dict, Any, Optional, Literal, Tuple
from torch import nn
import torch.nn.functional as F
import gin
from abc import ABC, abstractmethod

from . import wavetypings
from . import flux_calculation
from . import physics_calculation

@gin.register
class BaseTendency(nn.Module, ABC):
    """
    Base class for wave physics calculations.
    
    This class provides a template for calculating wave spectra propagation and other physics-related operations.
    It should be inherited by specific physics calculation classes that implement the required methods.
    """
    
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        spectra: wavetypings.FieldSpectra,
        Cg: torch.Tensor,
        dt: float,
        metadata: wavetypings.MetadataDict,
        **kwargs
    ) -> wavetypings.FieldSpectra:
        """
        Forward method to calculate wave spectra propagation.
        
        Args:
            spectra: Input wave spectra [B, W, H, D, F]
            Cg: Group velocity
            dt: Time step for propagation
            metadata: Dictionary containing fixed parameters and grid information
            **kwargs: Additional parameters for specific calculations
            
        Returns:
            Propagated wave spectra [B, W, H, D, F]
        """
        pass

@gin.register
class XTendency(BaseTendency):
    """
    Class for calculating propagation of wave spectra along the x-axis (longitude).
    Refactored to be compiler-friendly.
    """
    
    def __init__(
            self,
            flux_calculator: flux_calculation.QuickestFlux,
            eps: float = 1e-9
    ):
        super().__init__()
        self.flux_calculator = flux_calculator
        self.eps = eps

    def forward(
        self, 
        spectra: wavetypings.FieldSpectra, 
        dt: float,
        metadata: wavetypings.MetadataDict,
        land_mask: Optional[torch.Tensor] = None,
    ) -> wavetypings.FieldSpectra:

        if metadata['coordinate_type'] == 'Spherical':
            spec = metadata['FACHFA'] * spectra.clone()
            dxgrid = metadata['Dlon'] * (metadata['r_earth'] * metadata['cosLAT'])
            lamda = metadata['FACVX'] * metadata['Cg'] * metadata['sinTheta']
        elif metadata['coordinate_type'] == 'Cartesian':
            spec = metadata['FACHFA'] * spectra.clone()
            dxgrid = metadata['Dlon']
            lamda = metadata['FACVX'] * metadata['Cg'] * metadata['sinTheta']
        else:
            raise ValueError('coordinate_type should be Spherical or Cartesian.')

        spec = torch.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)

        if metadata.get('FULCR', False):
            raise NotImplementedError("Current handling is not implemented yet.")

        FLN_negx, FLN_posx = self.flux_calculator.calculate_flux(
            N=spec, Cg=lamda, land_mask=land_mask, Dgrid=dxgrid,
            target_dim=1, dt=dt
        )

        spatialXPro = (FLN_negx - FLN_posx) / dxgrid

        return spatialXPro

@gin.register
class YTendency(BaseTendency):
    """
    Class for calculating propagation of wave spectra along the y-axis (latitude).
    Refactored to be compiler-friendly.
    """
    
    def __init__(
        self,
        flux_calculator: flux_calculation.QuickestFlux,
        eps: float = 1e-9
    ):
        super().__init__()
        self.flux_calculator = flux_calculator
        self.eps = eps

    def forward(
        self, 
        spectra: wavetypings.FieldSpectra, 
        dt: float,
        metadata: wavetypings.MetadataDict,
        land_mask: Optional[torch.Tensor] = None,
    ) -> wavetypings.FieldSpectra:

        if metadata['coordinate_type'] == 'Spherical':
            spec = metadata['FACHFA'] * spectra.clone() / metadata['cosLAT']
            dygrid = metadata['Dlat'] * metadata['r_earth']
            phi = metadata['FACVY'] * metadata['Cg'] * metadata['cosTheta'] * metadata['cosLAT']
        elif metadata['coordinate_type'] == 'Cartesian':
            spec = metadata['FACHFA'] * spectra.clone()
            dygrid = metadata['Dlat']
            phi = metadata['FACVY'] * metadata['Cg'] * metadata['cosTheta']
        else:
            raise ValueError('coordinate_type should be Spherical or Cartesian.')

        spec = torch.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)

        if metadata.get('FULCR', False):
            pass

        FLN_negy, FLN_posy = self.flux_calculator.calculate_flux(
            N=spec, Cg=phi, land_mask=land_mask, Dgrid=dygrid, 
            target_dim=2, dt=dt
        )

        spatialYPro = (FLN_negy - FLN_posy) / dygrid

        return spatialYPro


@gin.register
class ThetaTendency(BaseTendency):
    """
    Class for calculating propagation of wave spectra along the theta-axis (direction).
    Refactored to be compiler-friendly.
    """
    
    def __init__(
        self,
        flux_calculator: flux_calculation.QuickestFlux,
        dsdd_calculator: physics_calculation.calculate_depth_dispersion_derivative,
        eps: float = 1e-9
    ):
        super().__init__()
        self.flux_calculator = flux_calculator
        self.dsdd_calculator = dsdd_calculator
        self.eps = eps
    
    def _get_VCTH(
        self,
        metadata: wavetypings.MetadataDict,
        depth: torch.Tensor,
        DdDx: torch.Tensor,
        DdDy: torch.Tensor
    ) -> torch.Tensor:
        DsDd = self.dsdd_calculator(
            metadata['Cg'], metadata['wavenum'], metadata['freq'], depth
        )

        if metadata.get('FULCR', False):
            pass

        safe_wavenum = torch.clamp(metadata['wavenum'], min=self.eps)
        VCTH = -metadata['FACTH'] * (1.0 / safe_wavenum) * (DsDd * (metadata['cosTheta'] * DdDx - metadata['sinTheta'] * DdDy))
        VCTH = torch.nan_to_num(VCTH, nan=0.0, posinf=0.0, neginf=0.0)

        if metadata['coordinate_type'] == 'Spherical':
            cgt = metadata['FDG'] * (metadata['Cg'] * torch.tan(metadata['lat']) * metadata['cosTheta']) / metadata['r_earth']
            cgt = torch.nan_to_num(cgt, nan=0.0, posinf=0.0, neginf=0.0)
            VCTH = VCTH - cgt

        return VCTH

    def forward(
        self, 
        spectra: wavetypings.FieldSpectra, 
        dt: float,
        metadata: wavetypings.MetadataDict,
        depth: torch.Tensor,
        DdDx: torch.Tensor,
        DdDy: torch.Tensor,
    ) -> wavetypings.FieldSpectra:
        VCTH = self._get_VCTH(metadata, depth, DdDx, DdDy)

        FLN_negth, FLN_posth = self.flux_calculator.calculate_flux(
            N=spectra, Cg=VCTH, Dgrid=metadata['DTH'], target_dim=3, dt=dt
        )

        safe_DTH = torch.clamp(metadata['DTH'], min=self.eps)
        thetaPro = (FLN_negth - FLN_posth) / safe_DTH

        assert FLN_posth[:,:,:,-1].sum() == FLN_negth[:,:,:,0].sum(), \
            f'FLN_posth[:,:,:,-1].sum() = {FLN_posth[:,:,:,-1].sum()}, FLN_negth[:,:,:,0].sum() = {FLN_negth[:,:,:,0].sum()}'

        return thetaPro[:,1:-1,1:-1,:,:]


@gin.register
class KTendency(BaseTendency):
    """
    Class for calculating propagation of wave spectra along the wavenumber (k) axis.
    Refactored to be compiler-friendly.
    """
    
    def __init__(
        self,
        flux_calculator: flux_calculation.QuickestFlux,
        dsdd_calculator: physics_calculation.calculate_depth_dispersion_derivative,
        eps: float = 1e-9
    ):
        super().__init__()
        self.flux_calculator = flux_calculator
        self.dsdd_calculator = dsdd_calculator
        self.eps = eps
    
    def _get_VCWN(
        self,
        metadata: wavetypings.MetadataDict,
        depth: torch.Tensor,
        DdDx: torch.Tensor,
        DdDy: torch.Tensor
    ) -> torch.Tensor:
        DsDd = self.dsdd_calculator(
            metadata['Cg'], metadata['wavenum'], metadata['freq'], depth
        )

        if metadata.get('FULCR', False):
            pass

        safe_Cg = torch.clamp(metadata['Cg'], min=self.eps)
        VCWN = -1.0 * metadata['FACK'] * (1.0 / safe_Cg) * DsDd * (metadata['sinTheta'] * DdDx + metadata['cosTheta'] * DdDy)
        VCWN = torch.nan_to_num(VCWN, nan=0.0, posinf=0.0, neginf=0.0)

        return VCWN

    def forward(
        self, 
        spectra: wavetypings.FieldSpectra, 
        dt: float,
        metadata: wavetypings.MetadataDict,
        depth: torch.Tensor,
        DdDx: torch.Tensor,
        DdDy: torch.Tensor,
    ) -> wavetypings.FieldSpectra:
        VCWN = self._get_VCWN(metadata, depth, DdDx, DdDy)

        FLN_negK, FLN_posK = self.flux_calculator.calculate_flux(
            N=spectra, Cg=VCWN, wavenum=metadata['wavenum'], target_dim=4, land_mask=depth>0,
            khf=metadata.get('khf', metadata['wavenum'].max()), dt=dt, m=metadata.get('m', 5.0)
        )

        wavenumPro = FLN_negK - FLN_posK
        wavenumPro = torch.nan_to_num(wavenumPro, nan=0.0, posinf=0.0, neginf=0.0)

        return wavenumPro[:,1:-1,1:-1,:,:]

@gin.register
class NeuralThetaTendency(ThetaTendency):
    """
    Class for calculating propagation of wave spectra along the theta-axis (direction).
    Using Neural Network to estimate VCTH.
    Refactored to be compiler-friendly.
    """
    
    def __init__(
        self,
        flux_calculator: flux_calculation.QuickestFlux,
        dsdd_calculator: physics_calculation.calculate_depth_dispersion_derivative,
        network: Optional[nn.Module] = None,
        num_intra_batch_block: Optional[int] = None,
        eps: float = 1e-9
    ):
        super().__init__(flux_calculator, dsdd_calculator, eps)
        self.flux_calculator = flux_calculator
        self.dsdd_calculator = dsdd_calculator
        self.network = nn.Sequential(
            nn.Linear(90, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 60)
        ) if network is None else network
        self.num_intra_batch_block = num_intra_batch_block
        self.eps = eps
    
    def _get_VCTH(
        self,
        metadata: wavetypings.MetadataDict,
        depth: torch.Tensor,
        DdDx: torch.Tensor,
        DdDy: torch.Tensor
    ) -> torch.Tensor:
        DsDd = self.dsdd_calculator(
            metadata['Cg'], metadata['wavenum'], metadata['freq'], depth
        )

        if metadata.get('FULCR', False):    
            pass

        safe_wavenum = torch.clamp(metadata['wavenum'], min=self.eps)
        b, h, w, d, f = safe_wavenum.shape
        inputs = torch.cat([
            (safe_wavenum * DdDx).view(b*h*w, d*f),
            (safe_wavenum * DdDy).view(b*h*w, d*f),
            DsDd.view(b*h*w, d*f)
        ], dim=-1)
        if self.num_intra_batch_block is None or self.num_intra_batch_block <= 1:
            hat_A = self.network(inputs)
        else:
            n_points = inputs.shape[0]
            blocks = int(self.num_intra_batch_block)
            block_size = (n_points + blocks - 1) // blocks
            out_chunks = []
            for start in range(0, n_points, block_size):
                end = min(start + block_size, n_points)
                out_chunks.append(self.network(inputs[start:end]))
            hat_A = torch.cat(out_chunks, dim=0)
        hat_A = hat_A.view(b, h, w, d, f, 2)
        VCTH = hat_A[...,0] * metadata['cosTheta'] - hat_A[...,1] * metadata['sinTheta']

        VCTH_per = F.tanh(VCTH)
        return VCTH_per
    
    def forward(
        self, 
        spectra: wavetypings.FieldSpectra, 
        dt: float,
        metadata: wavetypings.MetadataDict,
        depth: torch.Tensor,
        DdDx: torch.Tensor,
        DdDy: torch.Tensor,
    ) -> wavetypings.FieldSpectra:
        Cg_b_th = self._get_VCTH(metadata, depth, DdDx, DdDy)

        N_im1, N_i = flux_calculation.get_stencils(spectra, target_dim=3, pad_left=1, pad_right=0, mode='circular')
        is_positive = Cg_b_th >= 0
        N_upwind = torch.where(is_positive, N_im1, N_i)
        FLN_negth = Cg_b_th * N_upwind / dt if dt else Cg_b_th * N_upwind / 3600

        FLN_posth = flux_calculation.get_stencils(FLN_negth, target_dim=3, pad_left=0, pad_right=1, mode='circular')[1]

        thetaPro = FLN_negth - FLN_posth

        dis_cir = FLN_posth[:,:,:,-1].sum().round(decimals=6) - FLN_negth[:,:,:,0].sum().round(decimals=6)
        jdg = dis_cir / FLN_negth[:,:,:,0].sum().round(decimals=6)

        return thetaPro[:,1:-1,1:-1,:,:]

@gin.register
class NeuralKTendency(KTendency):
    def __init__(
        self,
        flux_calculator: flux_calculation.QuickestFlux,
        dsdd_calculator: physics_calculation.calculate_depth_dispersion_derivative,
        network: Optional[nn.Module] = None,
        num_intra_batch_block: Optional[int] = None,
        eps: float = 1e-9
    ):
        super().__init__(flux_calculator, dsdd_calculator, eps)
        self.flux_calculator = flux_calculator
        self.dsdd_calculator = dsdd_calculator
        self.network = nn.Sequential(
            nn.Linear(90, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 60)
        ) if network is None else network
        self.num_intra_batch_block = num_intra_batch_block
        self.eps = eps
    
    def _get_VCWN(
        self,
        metadata: wavetypings.MetadataDict,
        depth: torch.Tensor,
        DdDx: torch.Tensor,
        DdDy: torch.Tensor
    ) -> torch.Tensor:
        """Reuse the same input construction as NeuralThetaTendency."""
        DsDd = self.dsdd_calculator(
            metadata['Cg'], metadata['wavenum'], metadata['freq'], depth
        )

        if metadata.get('FULCR', False):    
            pass

        safe_wavenum = torch.clamp(metadata['wavenum'], min=self.eps)
        b, h, w, d, f = safe_wavenum.shape
        inputs = torch.cat([
            (safe_wavenum * DdDx).view(b*h*w, d*f),
            (safe_wavenum * DdDy).view(b*h*w, d*f),
            DsDd.view(b*h*w, d*f)
        ], dim=-1)
        if self.num_intra_batch_block is None or self.num_intra_batch_block <= 1:
            hat_A = self.network(inputs)
        else:
            n_points = inputs.shape[0]
            blocks = int(self.num_intra_batch_block)
            block_size = (n_points + blocks - 1) // blocks
            out_chunks = []
            for start in range(0, n_points, block_size):
                end = min(start + block_size, n_points)
                out_chunks.append(self.network(inputs[start:end]))
            hat_A = torch.cat(out_chunks, dim=0)
        hat_A = hat_A.view(b, h, w, d, f, 2)
        VCWN = hat_A[...,0] * metadata['cosTheta'] + hat_A[...,1] * metadata['sinTheta']

        VCWN_per = F.tanh(VCWN)
        return VCWN_per
    
    def forward(
        self, 
        spectra: wavetypings.FieldSpectra, 
        dt: float,
        metadata: wavetypings.MetadataDict,
        depth: torch.Tensor,
        DdDx: torch.Tensor,
        DdDy: torch.Tensor,
    ) -> wavetypings.FieldSpectra:
        Cg_b_k = self._get_VCWN(metadata, depth, DdDx, DdDy)

        N_im1, N_i = flux_calculation.get_stencils(spectra, target_dim=4, pad_left=1, pad_right=0, mode='replicate')
        is_positive = Cg_b_k >= 0
        N_upwind = torch.where(is_positive, N_im1, N_i)
        FLN_negK = Cg_b_k * N_upwind / dt if dt else Cg_b_k * N_upwind / 3600

        FLN_posK = flux_calculation.get_stencils(FLN_negK, target_dim=4, pad_left=0, pad_right=1, mode='replicate')[1]

        wavenumPro = FLN_negK - FLN_posK

        return wavenumPro[:,1:-1,1:-1,:,:]


@gin.register
class SamplingNeuralThetaTendency(NeuralThetaTendency):
    """Sampled theta tendency that limits gradient tracking to sampled ocean points."""

    def __init__(
        self,
        flux_calculator: flux_calculation.QuickestFlux,
        dsdd_calculator: physics_calculation.calculate_depth_dispersion_derivative,
        network: Optional[nn.Module] = None,
        num_intra_batch_block: Optional[int] = None,
        sample_ratio: Optional[float] = None,
        min_samples: int = 1024,
        eps: float = 1e-9,
    ):
        super().__init__(flux_calculator, dsdd_calculator, network, num_intra_batch_block, eps)
        self.sample_ratio = sample_ratio
        self.min_samples = min_samples

    def _run_network_chunked(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.num_intra_batch_block is None or self.num_intra_batch_block <= 1:
            return self.network(inputs)
        n_points = inputs.shape[0]
        blocks = int(self.num_intra_batch_block)
        block_size = (n_points + blocks - 1) // blocks
        out_chunks = []
        for start in range(0, n_points, block_size):
            end = min(start + block_size, n_points)
            out_chunks.append(self.network(inputs[start:end]))
        return torch.cat(out_chunks, dim=0)

    def _get_sampled_VCTH(
        self,
        metadata: wavetypings.MetadataDict,
        depth: torch.Tensor,
        DdDx: torch.Tensor,
        DdDy: torch.Tensor,
    ) -> torch.Tensor:
        safe_wavenum = torch.clamp(metadata['wavenum'], min=self.eps)
        b, h, w, d, f = safe_wavenum.shape
        inputs = torch.cat([
            (safe_wavenum * DdDx).view(b * h * w, d * f),
            (safe_wavenum * DdDy).view(b * h * w, d * f),
            self.dsdd_calculator(metadata['Cg'], metadata['wavenum'], metadata['freq'], depth).view(b * h * w, d * f)
        ], dim=-1)

        depth_flat = depth.reshape(-1)
        valid_idx = torch.nonzero(depth_flat > 0, as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0 or self.sample_ratio is None or self.sample_ratio <= 0:
            return super()._get_VCTH(metadata, depth, DdDx, DdDy)

        n_valid = valid_idx.numel()
        n_sample = max(int(n_valid * self.sample_ratio), self.min_samples)
        n_sample = min(n_sample, n_valid)

        rand_order = torch.randperm(n_valid, device=depth.device)
        sample_idx = valid_idx[rand_order[:n_sample]]
        rest_idx = valid_idx[rand_order[n_sample:]]

        hat_A_full = torch.zeros((b * h * w, d * f * 2), device=inputs.device, dtype=inputs.dtype)
        hat_A_full[sample_idx] = self.network(inputs[sample_idx])
        if rest_idx.numel() > 0:
            with torch.no_grad():
                hat_A_full[rest_idx] = self._run_network_chunked(inputs[rest_idx])

        hat_A = hat_A_full.view(b, h, w, d, f, 2)
        VCTH = hat_A[..., 0] * metadata['cosTheta'] - hat_A[..., 1] * metadata['sinTheta']
        return F.tanh(VCTH)

    def forward(
        self,
        spectra: wavetypings.FieldSpectra,
        dt: float,
        metadata: wavetypings.MetadataDict,
        depth: torch.Tensor,
        DdDx: torch.Tensor,
        DdDy: torch.Tensor,
    ) -> wavetypings.FieldSpectra:
        if self.sample_ratio is None or self.sample_ratio <= 0:
            return super().forward(spectra, dt, metadata, depth, DdDx, DdDy)

        Cg_b_th = self._get_sampled_VCTH(metadata, depth, DdDx, DdDy)
        N_im1, N_i = flux_calculation.get_stencils(spectra, target_dim=3, pad_left=1, pad_right=0, mode='circular')
        is_positive = Cg_b_th >= 0
        N_upwind = torch.where(is_positive, N_im1, N_i)
        FLN_negth = Cg_b_th * N_upwind / dt if dt else Cg_b_th * N_upwind / 3600
        FLN_posth = flux_calculation.get_stencils(FLN_negth, target_dim=3, pad_left=0, pad_right=1, mode='circular')[1]
        thetaPro = FLN_negth - FLN_posth
        return thetaPro[:, 1:-1, 1:-1, :, :]


@gin.register
class SamplingNeuralKTendency(NeuralKTendency):
    """Sampled wavenumber tendency mirroring SamplingNeuralThetaTendency."""

    def __init__(
        self,
        flux_calculator: flux_calculation.QuickestFlux,
        dsdd_calculator: physics_calculation.calculate_depth_dispersion_derivative,
        network: Optional[nn.Module] = None,
        num_intra_batch_block: Optional[int] = None,
        sample_ratio: Optional[float] = None,
        min_samples: int = 1024,
        eps: float = 1e-9,
    ):
        super().__init__(flux_calculator, dsdd_calculator, network, num_intra_batch_block, eps)
        self.sample_ratio = sample_ratio
        self.min_samples = min_samples

    def _run_network_chunked(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.num_intra_batch_block is None or self.num_intra_batch_block <= 1:
            return self.network(inputs)
        n_points = inputs.shape[0]
        blocks = int(self.num_intra_batch_block)
        block_size = (n_points + blocks - 1) // blocks
        out_chunks = []
        for start in range(0, n_points, block_size):
            end = min(start + block_size, n_points)
            out_chunks.append(self.network(inputs[start:end]))
        return torch.cat(out_chunks, dim=0)

    def _get_sampled_VCWN(
        self,
        metadata: wavetypings.MetadataDict,
        depth: torch.Tensor,
        DdDx: torch.Tensor,
        DdDy: torch.Tensor,
    ) -> torch.Tensor:
        safe_wavenum = torch.clamp(metadata['wavenum'], min=self.eps)
        b, h, w, d, f = safe_wavenum.shape
        inputs = torch.cat([
            (safe_wavenum * DdDx).view(b * h * w, d * f),
            (safe_wavenum * DdDy).view(b * h * w, d * f),
            self.dsdd_calculator(metadata['Cg'], metadata['wavenum'], metadata['freq'], depth).view(b * h * w, d * f)
        ], dim=-1)

        depth_flat = depth.reshape(-1)
        valid_idx = torch.nonzero(depth_flat > 0, as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0 or self.sample_ratio is None or self.sample_ratio <= 0:
            return super()._get_VCWN(metadata, depth, DdDx, DdDy)

        n_valid = valid_idx.numel()
        n_sample = max(int(n_valid * self.sample_ratio), self.min_samples)
        n_sample = min(n_sample, n_valid)

        rand_order = torch.randperm(n_valid, device=depth.device)
        sample_idx = valid_idx[rand_order[:n_sample]]
        rest_idx = valid_idx[rand_order[n_sample:]]

        hat_A_full = torch.zeros((b * h * w, d * f * 2), device=inputs.device, dtype=inputs.dtype)
        hat_A_full[sample_idx] = self.network(inputs[sample_idx])
        if rest_idx.numel() > 0:
            with torch.no_grad():
                hat_A_full[rest_idx] = self._run_network_chunked(inputs[rest_idx])

        hat_A = hat_A_full.view(b, h, w, d, f, 2)
        VCWN = hat_A[..., 0] * metadata['cosTheta'] + hat_A[..., 1] * metadata['sinTheta']
        return F.tanh(VCWN)

    def forward(
        self,
        spectra: wavetypings.FieldSpectra,
        dt: float,
        metadata: wavetypings.MetadataDict,
        depth: torch.Tensor,
        DdDx: torch.Tensor,
        DdDy: torch.Tensor,
    ) -> wavetypings.FieldSpectra:
        if self.sample_ratio is None or self.sample_ratio <= 0:
            return super().forward(spectra, dt, metadata, depth, DdDx, DdDy)

        Cg_b_k = self._get_sampled_VCWN(metadata, depth, DdDx, DdDy)
        N_im1, N_i = flux_calculation.get_stencils(spectra, target_dim=4, pad_left=1, pad_right=0, mode='replicate')
        is_positive = Cg_b_k >= 0
        N_upwind = torch.where(is_positive, N_im1, N_i)
        FLN_negK = Cg_b_k * N_upwind / dt if dt else Cg_b_k * N_upwind / 3600
        FLN_posK = flux_calculation.get_stencils(FLN_negK, target_dim=4, pad_left=0, pad_right=1, mode='replicate')[1]
        wavenumPro = FLN_negK - FLN_posK
        return wavenumPro[:, 1:-1, 1:-1, :, :]
