"""Flux calculators used by the public wave propagation path."""

import torch
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from torch import nn
import gin

from . import wavetypings

def get_stencils(
    tensor: torch.Tensor, 
    target_dim: int, 
    pad_left: int = 2,
    pad_right: int = 1,
    mode: str = 'circular'
) -> Tuple[torch.Tensor, ...]:
    """Create stencil views along one dimension without torch.roll."""
    dims = list(range(tensor.dim()))
    dims.pop(target_dim)
    dims.append(target_dim)
    
    inv_dims = [0] * tensor.dim()
    for i, d in enumerate(dims):
        inv_dims[d] = i
        
    padded_tensor = nn.functional.pad(
        tensor.permute(*dims), 
        (pad_left, pad_right, 0, 0, 0, 0), 
        mode=mode
    )
    padded_tensor = padded_tensor.permute(*inv_dims)
    
    original_size = tensor.size(target_dim)
    stencils = tuple(
        padded_tensor.narrow(target_dim, i, original_size) 
        for i in range(pad_left + pad_right + 1)
    )
    return stencils


@gin.register
class BasePhysicsCalculator(ABC):
    @abstractmethod
    def calculate_flux(
        self,
        N: wavetypings.FieldSpectra,
        Cg: wavetypings.GroupVelocity,
        target_dim: int = 1,
        **kwargs
    ) -> Tuple[wavetypings.FieldSpectra, wavetypings.FieldSpectra]:
        """
        Abstract method for calculating flux.
        
        Parameters:
        N : wavetypings.FieldSpectra
            Wave spectrum or field data
        Cg : wavetypings.GroupVelocity
            Group velocity
        target_dim : int
            Target dimension for flux calculation
            
        Returns:
            Tuple of (FLN_minus, FLN_plus)
        """
        pass


@gin.register
class FirstOrderFlux(BasePhysicsCalculator):
    """
    First order flux calculation scheme, refactored with stencils.
    """
    def calculate_flux(
        self,
        N: wavetypings.FieldSpectra,
        Cg: wavetypings.GroupVelocity,
        land_mask: wavetypings.SeaMask,
        target_dim: int = 1
    ) -> Tuple[wavetypings.FieldSpectra, wavetypings.FieldSpectra]:
        """
        Refactored to use stencils instead of torch.roll.
        """
        mode = 'circular' if target_dim in [1] else 'replicate'

        N_im1, N_i = get_stencils(N, target_dim, pad_left=1, pad_right=0, mode=mode)
        Cg_i, Cg_ip1 = get_stencils(Cg, target_dim, pad_left=0, pad_right=1, mode=mode)
        lm_i, lm_ip1 = get_stencils(land_mask.float(), target_dim, pad_left=0, pad_right=1, mode=mode)

        land2sea_mask = (lm_i == 0) & (lm_ip1 == 1)
        Cg_eff = torch.where(land2sea_mask.bool(), Cg_ip1, Cg_i)
        Cg_eff_im1 = get_stencils(Cg_eff, target_dim, pad_left=1, pad_right=0, mode=mode)[0]

        Cg_b = 0.5 * (Cg_eff_im1 + Cg_eff)
        
        is_positive = Cg_b >= 0
        N_upwind = torch.where(is_positive, N_im1, N_i)
        FLN_minus = Cg_b * N_upwind
        
        # FLN_plus is the i+1 view of FLN_minus
        FLN_plus = get_stencils(FLN_minus, target_dim, pad_left=0, pad_right=1, mode=mode)[1]
        
        return FLN_minus, FLN_plus


@gin.register
class QuickestFlux(FirstOrderFlux):
    """
    QUICKEST scheme, refactored to be compiler-friendly while preserving structure.
    """
    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps

    def _apply_tvd_limiter(
        self, 
        N: torch.Tensor, # Note: Pass original N for stencil creation inside
        Cg_b: torch.Tensor, 
        Nb: torch.Tensor, 
        CFL: torch.Tensor, 
        target_dim: int,
        mode: str
    ) -> torch.Tensor:
        """Refactored TVD limiter, numerically robust and compiler-friendly."""
        N_im2, N_im1, N_i, N_ip1 = get_stencils(N, target_dim, pad_left=2, pad_right=1, mode=mode)

        is_positive = Cg_b >= 0
        N_c = torch.where(is_positive, N_im1, N_i)
        N_u = torch.where(is_positive, N_im2, N_ip1)
        N_d = torch.where(is_positive, N_i, N_im1)
        
        DQ = N_d - N_u
        DQNZ = torch.sign(DQ) * torch.clamp(torch.abs(DQ), min=self.eps)
        DQNZ = torch.where(DQ == 0, -self.eps, DQNZ)

        Nc_tilde = torch.clamp((N_c - N_u) / DQNZ, -0.1, 1.1)
        Nb_tilde = torch.maximum((Nb - N_u) / DQNZ, Nc_tilde)
        CFL = torch.clamp(torch.abs(CFL), min=self.eps)
        Nclimit = Nc_tilde / CFL
        Nb_tilde = torch.minimum(Nb_tilde, torch.clamp(Nclimit, max=1.0))
        
        Nclimit2 = ((N_c / CFL) - N_u) / DQNZ
        Nc_tilde = torch.where(DQ < 0,
                    torch.maximum(Nc_tilde, Nclimit2),
                    torch.minimum(Nc_tilde, Nclimit2))
        Nc_rev = N_u + Nc_tilde * DQ
        
        Nb_rev = N_u + Nb_tilde * DQ
        CFAC = torch.clamp(torch.floor(2.0 * torch.abs(Nc_tilde - 0.5)), 0.0, 1.0)

        N_c = torch.where(CFL > 1.0, Nc_rev, N_c)
        return (1.0 - CFAC) * Nb_rev + CFAC * N_c

    def _handle_land_boundaries(
        self,
        N: torch.Tensor,
        Cg: torch.Tensor,
        Cg_b: torch.Tensor,
        Nb: torch.Tensor,
        land_mask: torch.Tensor,
        target_dim: int,
        mode: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refactored land boundary handler, preserving structure."""
        lm_im1, lm_i, lm_ip1 = get_stencils(land_mask.float(), target_dim, pad_left=1, pad_right=1, mode=mode)
        neighbor_land_mask = ((lm_i == 0) | (lm_im1 == 0) | (lm_ip1 == 0)).bool()

        FLN_minus_1order, _ = super().calculate_flux(N, Cg, land_mask, target_dim)
        FLN_minus = torch.where(neighbor_land_mask, FLN_minus_1order, Cg_b * Nb)

        FLN_plus = get_stencils(FLN_minus, target_dim, pad_left=0, pad_right=1, mode=mode)[1]

        sea2land_mask = (lm_i == 1) & (lm_ip1 == 0) & (Cg >= 0)
        FLN_plus = torch.where(sea2land_mask.bool(), Cg * N, FLN_plus)
        
        return FLN_minus, FLN_plus

    def calculate_flux(
        self,
        N: wavetypings.FieldSpectra,
        Cg: wavetypings.GroupVelocity,
        Dgrid: torch.Tensor,
        target_dim: int = 1,
        land_mask: Optional[wavetypings.SeaMask] = None,
        dt: float = 1.0
    ) -> Tuple[wavetypings.FieldSpectra, wavetypings.FieldSpectra]:
        mode = 'circular' if target_dim in [1, 3] else 'replicate'

        N_im2, N_im1, N_i, N_ip1 = get_stencils(N, target_dim, pad_left=2, pad_right=1, mode=mode)
        Cg_im1, Cg_i = get_stencils(Cg, target_dim, pad_left=1, pad_right=0, mode=mode)

        Dgrid_expanded = Dgrid.expand_as(N_i)
        safe_Dgrid = torch.clamp(Dgrid_expanded, min=self.eps)
        
        Cg_b = 0.5 * (Cg_im1 + Cg_i)
        CFL = (Cg_b * dt) / safe_Dgrid

        is_positive = Cg_b >= 0

        CU_positive = (N_im2 - 2.0 * N_im1 + N_i) / (safe_Dgrid**2)
        CU_negative = (N_im1 - 2.0 * N_i + N_ip1) / (safe_Dgrid**2)
        CU = torch.where(is_positive, CU_positive, CU_negative)
        
        Nb = 0.5 * ((1 + CFL) * N_im1 + (1 - CFL) * N_i) - (1 - CFL**2) / 6.0 * CU * Dgrid_expanded**2

        Nb = self._apply_tvd_limiter(N, Cg_b, Nb, CFL, target_dim, mode)
        
        if land_mask is not None:
            return self._handle_land_boundaries(N, Cg, Cg_b, Nb, land_mask, target_dim, mode)
        else:
            FLN_minus = Cg_b * Nb
            FLN_plus = get_stencils(FLN_minus, target_dim, pad_left=0, pad_right=1, mode=mode)[1]
            return FLN_minus, FLN_plus


@gin.register
class ThetaFlux(FirstOrderFlux):
    """
    Theta flux calculator. Third-Order scheme with tailor-made TVD limiter.
    """
    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps

    def _apply_tvd_limiter(
        self, 
        N: torch.Tensor, # Note: Pass original N for stencil creation inside
        Cg_b: torch.Tensor, 
        Nb: torch.Tensor, 
        CFL: torch.Tensor, 
        target_dim: int,
        mode: str
    ) -> torch.Tensor:
        """TVD limiter tailored for theta dimension."""
        N_im2, N_im1, N_i, N_ip1 = get_stencils(N, target_dim, pad_left=2, pad_right=1, mode=mode)

        is_positive = Cg_b >= 0
        N_c = torch.where(is_positive, N_im1, N_i)
        N_u = torch.where(is_positive, N_im2, N_ip1)
        N_d = torch.where(is_positive, N_i, N_im1)
        
        DQ = N_d - N_u
        DQNZ = torch.sign(DQ) * torch.clamp(torch.abs(DQ), min=self.eps)
        DQNZ = torch.where(DQ == 0, -self.eps, DQNZ)

        Nc_tilde = (N_c - N_u) / DQNZ
        Nb_tilde = torch.maximum((Nb - N_u) / DQNZ, torch.clamp(Nc_tilde, min=-0.1, max=1.1))
        CFL = torch.clamp(torch.abs(CFL), min=self.eps)

        Nclimit = ((N_c / CFL) - N_u) / DQNZ
        Nb_tilde = torch.minimum(Nb_tilde, torch.clamp(Nclimit, max=1.0))
        Nb_tilde = torch.where(DQ < 0,
                            torch.maximum(Nb_tilde, Nclimit),
                            torch.minimum(Nb_tilde, Nclimit))

        Nc_tilde = torch.where(DQ < 0,
                    torch.maximum(Nc_tilde, Nclimit),
                    torch.minimum(Nc_tilde, Nclimit))

        Nc_rev = N_u + Nc_tilde * DQ
        Nb_rev = N_u + Nb_tilde * DQ
        CFAC = torch.clamp(torch.floor(2.0 * torch.abs(Nc_tilde - 0.5)), 0.0, 1.0)

        return (1.0 - CFAC) * Nb_rev  + CFAC * Nc_rev

    def calculate_flux(
        self,
        N: wavetypings.FieldSpectra,
        Cg: wavetypings.GroupVelocity,
        Dgrid: torch.Tensor,
        target_dim: int = 1,
        land_mask: Optional[wavetypings.SeaMask] = None,
        dt: float = 1.0
    ) -> Tuple[wavetypings.FieldSpectra, wavetypings.FieldSpectra]:
        mode = 'circular' if target_dim in [1, 3] else 'replicate'

        N_im2, N_im1, N_i, N_ip1 = get_stencils(N, target_dim, pad_left=2, pad_right=1, mode=mode)
        Cg_im1, Cg_i = get_stencils(Cg, target_dim, pad_left=1, pad_right=0, mode=mode)

        Dgrid_expanded = Dgrid.expand_as(N_i)
        safe_Dgrid = torch.clamp(Dgrid_expanded, min=self.eps)
        
        Cg_b = 0.5 * (Cg_im1 + Cg_i)
        CFL = (torch.abs(Cg_b) * dt) / safe_Dgrid
        is_positive = Cg_b >= 0

        CU_positive = (N_im2 - 2.0 * N_im1 + N_i) / (safe_Dgrid**2)
        CU_negative = (N_im1 - 2.0 * N_i + N_ip1) / (safe_Dgrid**2)
        CU = torch.where(is_positive, CU_positive, CU_negative)
        
        Nb = 0.5 * ((1 + CFL) * N_im1 + (1 - CFL) * N_i) - (1 - CFL**2) / 6.0 * CU * Dgrid_expanded**2

        Nb = self._apply_tvd_limiter(N, Cg_b, Nb, CFL, target_dim, mode)
        
        if land_mask is not None:
            return self._handle_land_boundaries(N, Cg, Cg_b, Nb, land_mask, target_dim, mode)
        else:
            FLN_minus = Cg_b * Nb
            FLN_plus = get_stencils(FLN_minus, target_dim, pad_left=0, pad_right=1, mode=mode)[1]
            return FLN_minus, FLN_plus

@gin.register
class WavenumberFlux(ThetaFlux):
    """
    Wavenumber flux calculator, refactored for compiler-friendliness.
    """
    def _calculate_wavenumber_grid(self, wavenum: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Refactored to use stencils and avoid in-place modification."""
        mode = 'replicate'
        k_im1, k_i, k_ip1 = get_stencils(wavenum, target_dim, pad_left=1, pad_right=1, mode=mode)
        
        k_cell_boundary_i = 0.5 * (k_im1 + k_i)
        k_cell_boundary_ip1 = 0.5 * (k_i + k_ip1)
        
        DX1 = k_cell_boundary_ip1 - k_cell_boundary_i
        DX2 = k_ip1 - k_i

        safe_DX2 = torch.where(DX2 == 0, self.eps, DX2)
        return safe_DX2

    def _apply_wavenumber_boundaries(
        self, 
        FLN_minus: torch.Tensor, 
        N: torch.Tensor, 
        Cg: torch.Tensor, 
        wavenum: torch.Tensor, 
        target_dim: int,
        khf: float,
        m: float,
        mode: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refactored to be compiler-friendly."""
        mask_shape = [1] * FLN_minus.dim()
        mask_shape[target_dim] = FLN_minus.size(target_dim)
        low_bdy_mask = torch.zeros(tuple(mask_shape), device=N.device, dtype=torch.bool)
        low_bdy_mask.narrow(target_dim, 0, 1).fill_(True)
        
        FLN_minus = torch.where(low_bdy_mask, 0.0, FLN_minus)

        is_high_freq = (wavenum >= khf)
        if torch.any(is_high_freq):
            khf_index = torch.where(is_high_freq)[-1][-1].item()
            
            Cg_im1, Cg_i = get_stencils(Cg, target_dim, pad_left=1, pad_right=0, mode=mode)
            Cg_b = 0.5 * (Cg_im1 + Cg_i)

            slice_obj_hf = [slice(None)] * N.dim(); slice_obj_hf[target_dim] = khf_index
            N_hf = N[tuple(slice_obj_hf)]
            
            slice_obj_last = [slice(None)] * N.dim(); slice_obj_last[target_dim] = -1
            Cg_b_last = Cg_b[tuple(slice_obj_last)]
            Cg_last = Cg[tuple(slice_obj_last)]

            tail_flux = Cg_b_last * N_hf * (Cg_last / khf) ** (-m - 2)

            high_bdy_mask = torch.zeros(tuple(mask_shape), device=N.device, dtype=torch.bool)
            high_bdy_mask.narrow(target_dim, N.size(target_dim)-1, 1).fill_(True)
            
            FLN_minus = torch.where(high_bdy_mask, tail_flux, FLN_minus)

        FLN_plus = get_stencils(FLN_minus, target_dim, pad_left=0, pad_right=1, mode=mode)[1]
        return FLN_minus, FLN_plus

    def calculate_flux(
        self,
        N: wavetypings.FieldSpectra,
        Cg: wavetypings.GroupVelocity,
        wavenum: torch.Tensor,
        target_dim: int = 4, # Default for wavenumber
        land_mask: Optional[wavetypings.SeaMask] = None,
        dt: float = 1.0,
        khf: float = 1.0,
        m: float = 5.0
    ) -> Tuple[wavetypings.FieldSpectra, wavetypings.FieldSpectra]:
        Dgrid = self._calculate_wavenumber_grid(wavenum, target_dim)

        FLN_minus, FLN_plus = super(WavenumberFlux, self).calculate_flux(
            N=N, Cg=Cg, Dgrid=Dgrid, target_dim=target_dim, land_mask=None, dt=dt
        )

        mode = 'replicate'
        return self._apply_wavenumber_boundaries(FLN_minus, N, Cg, wavenum, target_dim, khf, m, mode)

@gin.register
class WavenumberFlux2(ThetaFlux):
    """
    Wavenumber flux calculator on a non-uniform k-grid (Leonard 1979; Eqs. 3.54–3.58).
    """
    def _k_metrics(
        self,
        wavenum: torch.Tensor,
        target_dim: int,
        mode: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute non-uniform k-grid metrics:
        - Face spacings: Δk_{m-5/2}, Δk_{m-3/2}, Δk_{m-1/2}, Δk_{m+1/2}
        - Cell widths: Δk_{m-2}, Δk_{m-1}, Δk_m
        """
        k_im3, k_im2, k_im1, k_i, k_ip1 = get_stencils(wavenum, target_dim, pad_left=3, pad_right=1, mode=mode)

        dk_m5h = torch.clamp(k_im2 - k_im3, min=self.eps)  # Δk_{m-5/2}
        dk_m3h = torch.clamp(k_im1 - k_im2, min=self.eps)  # Δk_{m-3/2}
        dk_mh  = torch.clamp(k_i   - k_im1, min=self.eps)  # Δk_{m-1/2}
        dk_p1h = torch.clamp(k_ip1 - k_i,   min=self.eps)  # Δk_{m+1/2}

        dk_m2 = 0.5 * torch.clamp(dk_m5h + dk_m3h, min=self.eps)  # Δk_{m-2}
        dk_m1 = 0.5 * torch.clamp(dk_m3h + dk_mh,  min=self.eps)  # Δk_{m-1}
        dk_m  = 0.5 * torch.clamp(dk_mh  + dk_p1h, min=self.eps)  # Δk_{m}

        return dk_m5h, dk_m3h, dk_mh, dk_p1h, dk_m2, dk_m1, dk_m

    def calculate_flux(
        self,
        N: wavetypings.FieldSpectra,
        Cg: wavetypings.GroupVelocity,
        wavenum: torch.Tensor,
        target_dim: int = 4, # Default for wavenumber
        land_mask: Optional[wavetypings.SeaMask] = None,  # not used in k-space
        dt: float = 1.0,
        khf: Optional[float] = None,
        m: float = 5.0,
        khf_index: Optional[int] = None,
    ) -> Tuple[wavetypings.FieldSpectra, wavetypings.FieldSpectra]:
        mode = 'replicate'

        N_im2, N_im1, N_i, N_ip1 = get_stencils(N,  target_dim, pad_left=2, pad_right=1, mode=mode)
        Cg_im1, Cg_i             = get_stencils(Cg, target_dim, pad_left=1, pad_right=0, mode=mode)
        Cg_b = 0.5 * (Cg_im1 + Cg_i)

        dk_m5h, dk_m3h, dk_mh, dk_p1h, dk_m2, dk_m1, dk_m = self._k_metrics(wavenum, target_dim, mode)

        CFL = torch.clamp((Cg_b * dt) / dk_mh, min=self.eps)
        is_pos = Cg_b >= 0

        slope_m   = (N_i   - N_im1) / dk_m1
        slope_m1  = (N_im1 - N_im2) / dk_m2
        CU_pos = (slope_m / dk_mh) - (slope_m1 / dk_m3h)

        slope_p   = (N_ip1 - N_i)   / dk_m
        slope_m   = (N_i   - N_im1) / dk_m1
        CU_neg = (slope_p / dk_p1h) - (slope_m / dk_mh)

        CU = torch.where(is_pos, CU_pos, CU_neg)

        Nb = 0.5 * ((1 + CFL) * N_im1 + (1 - CFL) * N_i) - (1 - CFL**2) * (CU * (dk_mh ** 2)) / 6.0

        Nb = self._apply_tvd_limiter(N, Cg_b, Nb, CFL, target_dim, mode)

        FLN_minus = Cg_b * Nb
        FLN_plus  = get_stencils(FLN_minus, target_dim, pad_left=0, pad_right=1, mode=mode)[1]

        first = [slice(None)] * FLN_minus.dim(); first[target_dim] = 0
        FLN_minus[tuple(first)] = 0.0

        if khf is not None and khf_index is not None:
            last_face = [slice(None)] * FLN_minus.dim(); last_face[target_dim] = -1

            N_hf = N.index_select(target_dim, torch.tensor([khf_index], device=N.device)).squeeze(target_dim)
            k_last = wavenum.select(target_dim, wavenum.size(target_dim) - 1)
            Cg_b_last = 0.5 * (
                Cg.select(target_dim, Cg.size(target_dim) - 2) +
                Cg.select(target_dim, Cg.size(target_dim) - 1)
            )
            ratio = torch.clamp(k_last / khf, min=self.eps)
            tail_flux = Cg_b_last * N_hf * torch.pow(ratio, -(m + 2.0))
            FLN_minus[tuple(last_face)] = tail_flux

        FLN_plus = get_stencils(FLN_minus, target_dim, pad_left=0, pad_right=1, mode=mode)[1]
        last_face = [slice(None)] * FLN_minus.dim(); last_face[target_dim] = -1
        FLN_plus[tuple(last_face)] = 0.0
        return FLN_minus, FLN_plus

@gin.configurable
def create_flux_calculator(
    scheme: str = "first_order",
    **kwargs
) -> BasePhysicsCalculator:
    """
    Factory function to create flux calculators.
    
    Parameters:
    scheme : str
        Type of flux scheme ("first_order", "quickest", "wavenumber")
    **kwargs : dict
        Additional parameters for specific schemes
        
    Returns:
        BasePhysicsCalculator instance
    """
    if scheme.lower() == "first_order":
        return FirstOrderFlux()
    elif scheme.lower() == "quickest":
        return QuickestFlux()
    elif scheme.lower() == "wavenumber":
        return WavenumberFlux()
    else:
        raise ValueError(f"Unknown flux scheme: {scheme}")


@gin.configurable
def calculate_flux(
    N: wavetypings.FieldSpectra,
    Cg: wavetypings.GroupVelocity,
    scheme: str = "first_order",
    target_dim: int = 1,
    **kwargs
) -> Tuple[wavetypings.FieldSpectra, wavetypings.FieldSpectra]:
    """
    Factory function for flux calculation.
    
    Parameters:
    N : wavetypings.FieldSpectra
        Wave spectrum or field data
    Cg : wavetypings.GroupVelocity
        Group velocity
    scheme : str
        Flux calculation scheme ("first_order", "quickest", "wavenumber")
    target_dim : int
        Target dimension for flux calculation
    **kwargs : dict
        Additional parameters specific to each scheme:
        - For quickest: Dgrid, land_mask, dt, CFL
        - For wavenumber: wavenum, land_mask, dt, CFL, khf, m, khf_index
        
    Returns:
        Tuple of (FLN_minus, FLN_plus)
    """
    calculator = create_flux_calculator(scheme)
    return calculator.calculate_flux(N=N, Cg=Cg, target_dim=target_dim, **kwargs)


# Backward compatibility functions
@gin.register
def pr1flux(
    N: wavetypings.FieldSpectra,
    Cg: wavetypings.GroupVelocity,
    land_mask: wavetypings.SeaMask,
    target_dim: int = 1
) -> wavetypings.FieldSpectra:
    """Backward compatibility function for first order flux."""
    return calculate_flux(N, Cg, scheme="first_order", target_dim=target_dim, land_mask=land_mask)


@gin.register
def qckflux(
    N: wavetypings.FieldSpectra,
    Cg: wavetypings.GroupVelocity,
    land_mask: wavetypings.SeaMask,
    Dgrid: torch.Tensor,
    dt: float = 1.0,
    CFL: float = 0.5,
    target_dim: int = 1
) -> wavetypings.FieldSpectra:
    """Backward compatibility function for QUICKEST flux."""
    return calculate_flux(
        N, Cg, scheme="quickest", target_dim=target_dim, 
        land_mask=land_mask, Dgrid=Dgrid, dt=dt, CFL=CFL
    )


def qckflux4k(
    spectra: wavetypings.FieldSpectra,
    Cg: wavetypings.GroupVelocity,
    wavenum: torch.Tensor,
    mask: Optional[wavetypings.SeaMask] = None,
    dt: float = 1.0,
    CFL: float = 0.5,
    khf: float = 1.0,
    m: float = 5.0,
    target_dim: int = 4,
    khf_index: Optional[int] = None
) -> wavetypings.FieldSpectra:
    """Backward compatibility function for wavenumber flux."""
    return calculate_flux(
        spectra, Cg, scheme="wavenumber", target_dim=target_dim,
        wavenum=wavenum, land_mask=mask, dt=dt, CFL=CFL, khf=khf, m=m, khf_index=khf_index
    )