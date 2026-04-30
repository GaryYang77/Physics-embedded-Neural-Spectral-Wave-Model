"""Time integrators used by the public wave propagation path."""
import torch
import torch.nn as nn
import gin
import math
from typing import Optional, Callable, Dict, Any
from abc import ABC, abstractmethod

from . import wavetypings
from . import tendencies
from . import boundary_process

def _greatest_common_divisor(a: int, b: int) -> int:
    """Return the greatest common divisor of two integers."""
    return math.gcd(a, b)

def _least_common_multiple(a: int, b: int) -> int:
    """Return the least common multiple of two integers."""
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // math.gcd(a, b) if a != 0 and b != 0 else 0


@gin.register
class OperatorSplittingIntegrator(nn.Module):
    """
    Operator splitting time integration scheme.
    """
    def __init__(
        self,
        tendency_functions: tendencies.BaseTendency,
        boundary_update_func: Optional[Callable] = None,
    ):
        super().__init__()
        self.tendency_functions = tendency_functions
        self.boundary_update_func = boundary_update_func
    
    def integrate(
        self,
        initial_spectra: wavetypings.FieldSpectra,
        dt: float,
        **kwargs
    ) -> wavetypings.FieldSpectra:
        tendency = self.tendency_functions(initial_spectra, dt, **kwargs)
        spectra = initial_spectra + tendency * dt
        if self.boundary_update_func is not None:
            spectra = self.boundary_update_func(spectra)
        return spectra


@gin.configurable('ModelAdvance')
class ModelAdvance(nn.Module):
    """
    Operator splitting time integration scheme.
    
    Splits the integration into spatial and spectral sub-steps,
    allowing different time steps for different processes.
    """
    def __init__(
        self,
        x_tendency: tendencies.XTendency = gin.REQUIRED,
        y_tendency: tendencies.YTendency = gin.REQUIRED,
        th_tendency: tendencies.ThetaTendency = gin.REQUIRED,
        k_tendency: tendencies.KTendency = gin.REQUIRED,
        stot_advance: nn.Module = gin.REQUIRED,
        boundary_update_func: Callable = gin.REQUIRED,
        apply_boundary_func: Callable = gin.REQUIRED
    ):
        """
        Initialize operator splitting integrator.
        """
        super().__init__()

        self.x_tendency = x_tendency
        self.y_tendency = y_tendency
        self.th_tendency = th_tendency
        self.k_tendency = k_tendency
        self.stot_advance = stot_advance
        self.boundary_update_func = boundary_update_func
        self.apply_boundary_func = apply_boundary_func

    def forward(self,
        spectra: wavetypings.FieldSpectra,
        stot: wavetypings.StotSpectra,
        depth: wavetypings.Depth,
        metadata: wavetypings.MetadataDict,
        mask: wavetypings.ValidMask,
        boundary_condition: wavetypings.BoundaryCondition,
        **kwargs
    ) -> wavetypings.FieldSpectra:
        """
        Integrate using operator splitting with dynamic time stepping.
        
        Args:
            spectra: Initial wave spectra
            metadata: Dictionary containing time step information
            **kwargs: Additional parameters
        
        Returns:
            Updated state after integration
        """

        yfirst_initial = metadata.get('yfirst', True)
        dt_global = metadata.get('dt_global', 3600)
        dt_x = metadata.get('dt_x', 1800)
        dt_y = metadata.get('dt_y', 1800)
        dt_th = metadata.get('dt_th', 600)
        dt_k = metadata.get('dt_k', 600)
        dt_stot = metadata.get('dt_stot', 300)

        # Precompute masks and depth derivatives.
        land_mask = depth > 0

        depth_3d = depth.squeeze(-1).squeeze(-1)

        depthx_3d = nn.functional.pad(depth_3d.permute(0, 2, 1), (1,1), mode='circular').permute(0, 2, 1)
        depthy_3d = nn.functional.pad(depth_3d, (1,1), mode='replicate')
        depthx = depthx_3d.unsqueeze(-1).unsqueeze(-1)
        depthy = depthy_3d.unsqueeze(-1).unsqueeze(-1)

        DdDx = 0.5 * ((depthx[:, 2:, ...] - depthx[:, 1:-1, ...]) + (depthx[:, 1:-1, ...] - depthx[:, :-2, ...])) / metadata['Dlon']
        DdDy = 0.5 * ((depthy[:, :, 2:, ...] - depthy[:, :, 1:-1, ...]) + (depthy[:, :, 1:-1, ...] - depthy[:, :, :-2, ...])) / metadata['Dlat']
        if metadata['coordinate_type'] == 'Spherical':
            DdDx = DdDx / (metadata['r_earth'] * metadata['cosLAT'])
            DdDy = DdDy / metadata['r_earth']
        DdDx = DdDx * land_mask
        DdDy = DdDy * land_mask
        
        wavenum = torch.where(land_mask.expand_as(metadata['wavenum']), metadata['wavenum'], 0.0)
        Cg = torch.where(land_mask.expand_as(metadata['Cg']), metadata['Cg'], 0.0)
        wavenum = torch.nan_to_num(wavenum, nan=0.0, posinf=0.0, neginf=0.0)
        Cg = torch.nan_to_num(Cg, nan=0.0, posinf=0.0, neginf=0.0)
        physics_metadata = metadata.copy()
        physics_metadata.update({'wavenum': wavenum, 'Cg': Cg})

        # Build a static time-step schedule.
        active_dts_int = [int(dt) for dt in [dt_x, dt_y, dt_th, dt_k, dt_stot] if dt > 0]
        if not active_dts_int:
            return spectra
        dt_min = active_dts_int[0]
        for dt in active_dts_int[1:]:
            dt_min = _greatest_common_divisor(dt_min, dt)
        num_steps = int(dt_global / dt_min)
        
        # Precompute per-step execution masks.
        step_indices = torch.arange(1, num_steps + 1, device=spectra.device)
        should_integrate_x = (step_indices * dt_min % dt_x < 1e-5) if dt_x > 0 else torch.zeros_like(step_indices, dtype=torch.bool)
        should_integrate_y = (step_indices * dt_min % dt_y < 1e-5) if dt_y > 0 else torch.zeros_like(step_indices, dtype=torch.bool)
        should_integrate_th = (step_indices * dt_min % dt_th < 1e-5) if dt_th > 0 else torch.zeros_like(step_indices, dtype=torch.bool)
        should_integrate_k = (step_indices * dt_min % dt_k < 1e-5) if dt_k > 0 else torch.zeros_like(step_indices, dtype=torch.bool)
        should_integrate_stot = (step_indices * dt_min % dt_stot < 1e-5) if dt_stot > 0 else torch.zeros_like(step_indices, dtype=torch.bool)
        
        # Fixed-length integration loop.
        for i in range(num_steps):
            
            # --- Spectral integration (half step before spatial) ---
            th_tend = self.th_tendency(spectra, dt=dt_th, metadata=physics_metadata, depth=depth, DdDx=DdDx, DdDy=DdDy)
            update_th = torch.where(should_integrate_th[i], 0.5 * dt_th * th_tend, torch.zeros_like(th_tend))
            spectra_after_th1 = spectra.clone()
            spectra_after_th1[:, 1:-1, 1:-1, :, :] += update_th
            min_val = spectra_after_th1[land_mask.expand_as(spectra_after_th1)].min()
            spectra = torch.clamp(spectra_after_th1, min=0.0)

            k_tend = self.k_tendency(spectra, dt=dt_k, metadata=physics_metadata, depth=depth, DdDx=DdDx, DdDy=DdDy)
            update_k = torch.where(should_integrate_k[i], 0.5 * dt_k * k_tend, torch.zeros_like(k_tend))
            spectra_after_k1 = spectra.clone()
            spectra_after_k1[:, 1:-1, 1:-1, :, :] += update_k
            min_val = spectra_after_k1[land_mask.expand_as(spectra_after_k1)].min()
            spectra = torch.clamp(spectra_after_k1, min=0.0)
            
            # Spatial integration.
            num_x_steps_done = should_integrate_x[:i].sum()
            num_y_steps_done = should_integrate_y[:i].sum()
            yfirst_current = ((num_x_steps_done + num_y_steps_done) % 2 == 0) == yfirst_initial
            
            # --- Path 1: Y-first ---
            # 1.a: Y-step (Apply -> Tendency -> Update -> Clamp -> Update)
            spectra_pre_y = self.apply_boundary_func(spectra, boundary_condition, mask)
            y_tend = self.y_tendency(spectra_pre_y, dt=dt_y, metadata=physics_metadata, land_mask=land_mask)
            update_y = torch.where(should_integrate_y[i], dt_y * y_tend, torch.zeros_like(y_tend))
            spectra_after_y = self.boundary_update_func(torch.clamp(spectra_pre_y + update_y, min=0.0))

            # 1.b: X-step (Apply -> Tendency -> Update -> Clamp -> Update)
            spectra_pre_x_after_y = self.apply_boundary_func(spectra_after_y, boundary_condition, mask)
            x_tend_after_y = self.x_tendency(spectra_pre_x_after_y, dt=dt_x, metadata=physics_metadata, land_mask=land_mask)
            update_x_after_y = torch.where(should_integrate_x[i], dt_x * x_tend_after_y, torch.zeros_like(x_tend_after_y))
            spectra_y_first_final = self.boundary_update_func(torch.clamp(spectra_pre_x_after_y + update_x_after_y, min=0.0))

            # --- Path 2: X-first ---
            # 2.a: X-step (Apply -> Tendency -> Update -> Clamp -> Update)
            spectra_pre_x = self.apply_boundary_func(spectra, boundary_condition, mask)
            x_tend = self.x_tendency(spectra_pre_x, dt=dt_x, metadata=physics_metadata, land_mask=land_mask)
            update_x = torch.where(should_integrate_x[i], dt_x * x_tend, torch.zeros_like(x_tend))
            spectra_after_x = self.boundary_update_func(torch.clamp(spectra_pre_x + update_x, min=0.0))
            
            # 2.b: Y-step (Apply -> Tendency -> Update -> Clamp -> Update)
            spectra_pre_y_after_x = self.apply_boundary_func(spectra_after_x, boundary_condition, mask)
            y_tend_after_x = self.y_tendency(spectra_pre_y_after_x, dt=dt_y, metadata=physics_metadata, land_mask=land_mask)
            update_y_after_x = torch.where(should_integrate_y[i], dt_y * y_tend_after_x, torch.zeros_like(y_tend_after_x))
            spectra_x_first_final = self.boundary_update_func(torch.clamp(spectra_pre_y_after_x + update_y_after_x, min=0.0))
            
            # --- Merge the paths based on yfirst_current ---
            spectra = torch.where(yfirst_current, spectra_y_first_final, spectra_x_first_final)
            
            # --- Spectral integration (half step after spatial) ---
            th_tend = self.th_tendency(spectra, dt=dt_th, metadata=physics_metadata, depth=depth, DdDx=DdDx, DdDy=DdDy)
            update_th = torch.where(should_integrate_th[i], 0.5 * dt_th * th_tend, torch.zeros_like(th_tend))
            spectra_after_th2 = spectra.clone()
            spectra_after_th2[:, 1:-1, 1:-1, :, :] += update_th
            min_val = spectra_after_th2[land_mask.expand_as(spectra_after_th2)].min()
            spectra = torch.clamp(spectra_after_th2, min=0.0)

            k_tend = self.k_tendency(spectra, dt=dt_k, metadata=physics_metadata, depth=depth, DdDx=DdDx, DdDy=DdDy)
            update_k = torch.where(should_integrate_k[i], 0.5 * dt_k * k_tend, torch.zeros_like(k_tend))
            spectra_after_k2 = spectra.clone()
            spectra_after_k2[:, 1:-1, 1:-1, :, :] += update_k
            min_val = spectra_after_k2[land_mask.expand_as(spectra_after_k2)].min()
            spectra = torch.clamp(spectra_after_k2, min=0.0)
            
            if should_integrate_stot[i]:
                spectra = self.boundary_update_func(spectra)
                spectra = self.apply_boundary_func(spectra, boundary_condition, mask)
                spectra = self.stot_advance(spectra, stot, dt=dt_stot, metadata=physics_metadata, mask=mask)
                spectra = torch.clamp(spectra, min=0.0)

        spectra = self.boundary_update_func(spectra)
        spectra = self.apply_boundary_func(spectra, boundary_condition, mask)
        spectra = torch.clamp(spectra, min=0.0)
        
        return spectra

@gin.register
class SimpleStotIntegrator(nn.Module):
    """
    Simple source term integrator using a semi-implicit scheme.
    """

    def __init__(self, method: str = "explicit"):
        super().__init__()
        self.method = method

    def forward(
        self, 
        spectra: wavetypings.FieldSpectra,
        stot: wavetypings.FieldSpectra,
        dt: float,
        metadata: wavetypings.MetadataDict,
        mask: wavetypings.ValidMask
    ) -> wavetypings.FieldSpectra:
        update_term = stot * (dt * 0.01)
        updated_spectra = spectra + update_term
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand_as(spectra)
        return torch.where(mask_expanded, updated_spectra, spectra)


@gin.register
class StotIntegrator(nn.Module):
    def __init__(self, epsilon=1.0, dt_min=0.05, xp=0.15, xr=0.10, xf=0.05, alpha=0.62e-4):
        """
        Non-ice source term integrator implementing semi-implicit integration scheme
        based on WAVEWATCH III methodology.
        
        Args:
            epsilon: Offset parameter for semi-implicit scheme (default: 1.0)
            dt_min: Minimum time step (default: 0.05)
            xp, xr, xf: User-defined constants for dynamic time stepping
            alpha: PM spectrum energy level parameter
        """
        super(StotIntegrator, self).__init__()
        self.epsilon = epsilon
        self.dt_min = dt_min
        self.xp = xp
        self.xr = xr
        self.xf = xf
        self.alpha = alpha
    
    def integrate(self, spectra, stot, dt_global=1.0, freq=None, max_substeps=10):
        """
        Integrate source terms using semi-implicit scheme with dynamic time stepping.
        
        Args:
            spectra: Wave action density [B, W, H, D, F] where D=directions, F=frequencies
            stot: Source term [B, W, H, D, F] in units of ∂N/∂t
            dt_global: Global time step (default: 1.0)
            freq: Frequency array [F] for parametric calculations
            max_substeps: Maximum number of dynamic substeps
            
        Returns:
            Updated spectra after source term integration
        """
        device = spectra.device
        B, W, H, D, F = spectra.shape
        
        # Initialize output
        N_new = spectra.clone()
        
        # If no frequencies provided, create a default array
        if freq is None:
            freq = torch.linspace(0.04, 1.0, F, device=device)
        
        # Calculate wavenumber (deep water approximation: k = ω²/g)
        g = 9.81  # gravity
        omega = 2 * torch.pi * freq
        k = omega**2 / g  # [F]
        k = k.view(1, 1, 1, 1, F).expand(B, W, H, D, F)
        
        # Calculate diagonal derivative terms D (simplified approximation)
        # In practice, this would be computed from the specific source term formulations
        D = torch.abs(stot) / (torch.abs(N_new) + 1e-8)  # Approximate diagonal derivative
        
        # Dynamic time stepping loop
        remaining_time = torch.full((B, W, H), dt_global, device=device)
        
        for substep in range(max_substeps):
            # Check if any points need further integration
            active_mask = remaining_time > 1e-6
            if not active_mask.any():
                break
            
            # Calculate parametric change limit ΔNp (Eq. 3.67)
            # ΔNp = Xp * (α/π) * (2π)^4 / g^2 * 1/σk^3
            sigma_k = omega.view(1, 1, 1, 1, F).expand(B, W, H, D, F)
            delta_Np = self.xp * (self.alpha / torch.pi) * (2*torch.pi)**4 / g**2 * 1.0 / (sigma_k * k**3)
            delta_Np = torch.clamp(delta_Np, min=1e-12)
            
            # Calculate relative change limit ΔNr (Eq. 3.68)
            N_filter = torch.clamp(N_new, min=1e-12)
            N_max = torch.max(N_new.view(B, W, H, -1), dim=-1)[0]  # Max over D,F dimensions
            N_filter_level = torch.maximum(
                delta_Np.max(dim=-1)[0].max(dim=-1)[0],  # max over frequencies and directions
                self.xf * N_max.unsqueeze(-1).unsqueeze(-1)
            )
            delta_Nr = self.xr * torch.maximum(N_new, N_filter_level.unsqueeze(-1).unsqueeze(-1))
            
            # Maximum change ΔNm (Eq. 3.66)
            delta_Nm = torch.minimum(delta_Np, delta_Nr)
            
            # Calculate dynamic time step (Eq. 3.62)
            S_abs = torch.abs(stot) + 1e-12
            dt_dynamic = delta_Nm / S_abs
            
            # Apply epsilon correction for semi-implicit scheme
            dt_dynamic = dt_dynamic / (1 + self.epsilon * D * dt_dynamic + 1e-12)
            
            # Take minimum over all spectral components for each grid point
            dt_dynamic = torch.min(dt_dynamic.view(B, W, H, -1), dim=-1)[0]  # [B, W, H]
            
            # Apply minimum time step constraint (Eq. 3.63)
            dt_dynamic = torch.maximum(dt_dynamic, torch.full_like(dt_dynamic, self.dt_min))
            
            # Ensure we don't exceed remaining time (Eq. 3.64)
            dt_dynamic = torch.minimum(dt_dynamic, remaining_time)
            
            # Expand dt_dynamic to match spectral dimensions
            dt_expanded = dt_dynamic.unsqueeze(-1).unsqueeze(-1).expand(B, W, H, D, F)
            
            # Semi-implicit update (Eq. 3.65)
            denominator = 1 - self.epsilon * D * dt_expanded
            denominator = torch.clamp(denominator, min=1e-8)  # Avoid division by zero
            
            delta_N = stot * dt_expanded / denominator
            N_new = torch.clamp(N_new + delta_N, min=0.0)  # Ensure non-negative
            
            # Update remaining time
            remaining_time = remaining_time - dt_dynamic
            
            # Optional: Add limiter to prevent unrealistic values
            # This is a simplified version of the parametric limiter
            mask_large = delta_N.abs() > delta_Np
            N_new = torch.where(mask_large, 
                               N_new - delta_N + torch.sign(delta_N) * delta_Np,
                               N_new)
        
        return N_new
