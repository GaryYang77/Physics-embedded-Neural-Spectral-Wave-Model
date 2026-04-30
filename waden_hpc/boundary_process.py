"""Boundary-condition helpers for the public wave propagation path."""
import torch
import torch.nn as nn
import gin
from typing import Optional, Callable, Dict, Any, Union
from abc import ABC, abstractmethod

from . import wavetypings
from . import flux_calculation
from . import tendencies


@gin.register
def update_period_boundary(
    vars: Union[wavetypings.FieldSpectra, wavetypings.Depth]
) -> Union[wavetypings.FieldSpectra, wavetypings.Depth]:
    """Update the period boundary conditions using efficient vectorized operations."""
    b, w, h = vars.shape[:3]
    device = vars.device

    mask_left = torch.zeros((w, h), dtype=torch.bool, device=device); mask_left[0, :] = True
    mask_right = torch.zeros((w, h), dtype=torch.bool, device=device); mask_right[-1, :] = True

    vars_after_left = torch.where(mask_left[None, :, :, None, None], vars[:, -2:-1, ...], vars)
    vars_after_horizontal = torch.where(mask_right[None, :, :, None, None], vars[:, 1:2, ...], vars_after_left)

    mask_top = torch.zeros((w, h), dtype=torch.bool, device=device); mask_top[:, -1] = True
    mask_bottom = torch.zeros((w, h), dtype=torch.bool, device=device); mask_bottom[:, 0] = True

    vars_after_bottom = torch.where(mask_bottom[None, :, :, None, None], vars_after_horizontal[..., -2:-1, :, :], vars_after_horizontal)
    vars_final = torch.where(mask_top[None, :, :, None, None], vars_after_horizontal[..., 1:2, :, :], vars_after_bottom)

    return vars_final

@gin.register
def update_close_boundary(
    vars: Union[wavetypings.FieldSpectra, wavetypings.Depth]
) -> Union[wavetypings.FieldSpectra, wavetypings.Depth]:
    """Update closed boundary conditions by copying adjacent values."""
    b, w, h = vars.shape[:3]
    device = vars.device

    mask_left = torch.zeros((w, h), dtype=torch.bool, device=device); mask_left[0, :] = True
    mask_right = torch.zeros((w, h), dtype=torch.bool, device=device); mask_right[-1, :] = True
    mask_top = torch.zeros((w, h), dtype=torch.bool, device=device); mask_top[:, -1] = True
    mask_bottom = torch.zeros((w, h), dtype=torch.bool, device=device); mask_bottom[:, 0] = True

    vars_after_left = torch.where(mask_left[None, :, :, None, None], vars[:, 1:2, ...], vars)
    vars_after_horizontal = torch.where(mask_right[None, :, :, None, None], vars[:, -2:-1, ...], vars_after_left)

    vars_after_bottom = torch.where(mask_bottom[None, :, :, None, None], vars_after_horizontal[..., 1:2, :, :], vars_after_horizontal)
    vars_final = torch.where(mask_top[None, :, :, None, None], vars_after_horizontal[..., -2:-1, :, :], vars_after_bottom)

    return vars_final

@gin.register
def update_absorb_boundary(
    vars: Union[wavetypings.FieldSpectra, wavetypings.Depth]
) -> Union[wavetypings.FieldSpectra, wavetypings.Depth]:
    """Keep interior values and set boundary cells to zero."""
    mask_interior = torch.zeros_like(vars, dtype=torch.bool)
    mask_interior[:, 1:-1, 1:-1, :, :] = True
    return torch.where(mask_interior, vars, torch.tensor(0.0, device=vars.device, dtype=vars.dtype))


@gin.register
def update_absorb_boundary_global(
    vars: Union[wavetypings.FieldSpectra, wavetypings.Depth]
) -> Union[wavetypings.FieldSpectra, wavetypings.Depth]:
    """Keep interior values and set boundary cells to zero."""
    mask_interior = torch.zeros_like(vars, dtype=torch.bool)
    mask_interior[:, :, 1:-1, :, :] = True
    return torch.where(mask_interior, vars, torch.tensor(0.0, device=vars.device, dtype=vars.dtype))

@gin.register
def update_simple_boundary(
    vars: Union[wavetypings.FieldSpectra, wavetypings.Depth]
) -> Union[wavetypings.FieldSpectra, wavetypings.Depth]:
    """Apply absorbing boundaries in W and periodic boundaries in H."""
    vars_with_absorb_boundary = update_absorb_boundary(vars)
    b, w, h = vars.shape[:3]
    device = vars.device
    
    mask_bottom_2d = torch.zeros((w, h), dtype=torch.bool, device=device)
    mask_bottom_2d[:, 0] = True
    mask_bottom_5d = mask_bottom_2d[None, :, :, None, None]

    mask_top_2d = torch.zeros((w, h), dtype=torch.bool, device=device)
    mask_top_2d[:, -1] = True
    mask_top_5d = mask_top_2d[None, :, :, None, None]

    vars_with_bottom_updated = torch.where(
        mask_bottom_5d,
        vars[:, :, -2:-1, :, :],
        vars_with_absorb_boundary
    )

    vars_final = torch.where(
        mask_top_5d,
        vars[:, :, 1:2, :, :],
        vars_with_bottom_updated
    )

    return vars_final

@gin.register
def apply_boundary_condition(
    spectra: wavetypings.FieldSpectra,
    boundary_condition: wavetypings.BoundaryCondition,
    mask: wavetypings.ValidMask
) -> wavetypings.FieldSpectra:
    """Apply boundary conditions with torch.where."""
    mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand_as(spectra)
    boundary_mask = boundary_condition.sum(dim=(-1, -2)) > 0
    boundary_mask_expanded = boundary_mask.unsqueeze(-1).unsqueeze(-1).expand_as(spectra)
    spectra_masked = torch.where(
        mask_expanded,
        spectra,
        torch.tensor(0.0, device=spectra.device, dtype=spectra.dtype)
    )
    final_spectra = torch.where(
        boundary_mask_expanded,
        boundary_condition,
        spectra_masked
    )
    return final_spectra