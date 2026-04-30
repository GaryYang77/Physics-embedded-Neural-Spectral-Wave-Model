"""Inference entrypoint for source-data reproduction."""

import os
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.compiler
from tqdm import tqdm
import pandas as pd
from typing import List, Optional, Dict
import logging
from datetime import datetime
import gin
import argparse
import json
from utils.norm_utils import load_input_norm

from neuralwave import NeuralWave
from utils.utils import calculate_wave_parameters, calculate_stokes_drift_vector
from infer_dataloader import create_infer_dataloader

DEFAULT_OUTPUT_PARAMS = {'swh', 'mwd', 'mwp'}

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('high')
torch.compiler.cudagraph_mark_step_begin()

logging.getLogger("cfgrib").setLevel(logging.ERROR)
logging.getLogger("eccodes").setLevel(logging.ERROR)
logging.getLogger("ecCodes").setLevel(logging.ERROR)

class _SuppressIndexWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.getMessage())
        if "Ignoring index file" in msg and "incompatible with GRIB file" in msg:
            return False
        return True

logging.getLogger().addFilter(_SuppressIndexWarning())

def setup_directories(
    results_dir: str,
    save_stability_matrix: bool = False,
    save_spectra_samples: bool = False,
):
    """Create the output directories required for the selected artifacts."""
    logger.info(f"Setting up results directory at: {results_dir}")
    os.makedirs(os.path.join(results_dir, "integral_parameters"), exist_ok=True)
    if save_spectra_samples:
        os.makedirs(os.path.join(results_dir, "sample_spectra"), exist_ok=True)
    if save_stability_matrix:
        os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)

def load_dataset_info(data_path: str) -> Dict:
    """Load coordinates and metadata from a dataset file."""
    logger.info(f"Loading dataset info from: {data_path}")
    with xr.open_dataset(data_path) as ds:
        info = {
            'lon': ds['lon'].values,
            'lat': ds['lat'].values,
            'theta': ds['directionNumber'].values,
            'freq': ds['frequencyNumber'].values,
            'num_sequences': ds.dims['batch']
        }
    logger.info(f"  - Total samples found: {info['num_sequences']}")
    return info
def save_results_and_visualize(
    params_list: List[Dict],
    results_dir: str,
    ds_info: Dict,
    lead_time: int,
    model_path: str,
    save_sample_results: bool = False,
    output_params: Optional[set[str]] = None,
    save_stability_matrix: bool = False,
):
    """Save aggregated integral parameters and optional diagnostics."""
    logger.info("Saving integral parameters and calculating metrics...")

    all_indices = []  # Original sample indices aligned with axis 0.
    base_params = ['swh', 'mwd', 'mwp']
    if params_list:
        for key in params_list[0]['predictions'].keys():
            if key not in base_params:
                base_params.append(key)
    all_pred_params = {k: [] for k in base_params}
    all_label_params = {k: [] for k in base_params}
    all_masks = []
    all_actual_times = []
    all_forcings = []
    all_stability = []  # 1 indicates a stable step, 0 indicates NaNs in the rollout.

    for batch_params in params_list:
        all_indices.extend(batch_params['indices'])
        for param in base_params:
            if param in batch_params['predictions']:
                all_pred_params[param].append(batch_params['predictions'][param])
            if param in batch_params['labels']:
                all_label_params[param].append(batch_params['labels'][param])
        all_masks.append(batch_params['mask'])
        all_actual_times.append(batch_params['actual_times'])
        all_forcings.append(batch_params['forcings'])
        if 'stability' in batch_params:
            all_stability.append(batch_params['stability'])

    for param in base_params:
        if all_pred_params.get(param):
            all_pred_params[param] = np.concatenate(all_pred_params[param], axis=0)
        if all_label_params.get(param):
            all_label_params[param] = np.concatenate(all_label_params[param], axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    all_actual_times = np.concatenate(all_actual_times, axis=0)
    all_forcings = np.concatenate(all_forcings, axis=0)
    if len(all_stability) > 0:
        all_stability = np.concatenate(all_stability, axis=0).astype(np.uint8)
    else:
        all_stability = np.ones((len(all_indices), lead_time), dtype=np.uint8)

    total_samples = len(all_indices)
    stable_steps_per_sample = all_stability.sum(axis=1)
    sample_valid_mask = stable_steps_per_sample == lead_time
    num_unstable = int((~sample_valid_mask).sum())
    avg_stable_steps = float(stable_steps_per_sample.mean())

    logger.info(f"Numerical stability summary: {num_unstable}/{total_samples} samples had NaNs; average stable steps = {avg_stable_steps:.2f}")

    if save_stability_matrix:
        try:
            stab_coords = {'sample': all_indices, 'time_step': range(lead_time)}
            stab_ds = xr.Dataset({'stability': (('sample', 'time_step'), all_stability.astype(np.uint8))}, coords=stab_coords)
            stab_ds.attrs['description'] = 'Per-sample, per-step numerical stability (1=stable, 0=NaN)'
            stab_path = os.path.join(results_dir, 'metrics', 'stability_matrix.nc')
            stab_ds.to_netcdf(stab_path)
            logger.info(f"Stability matrix saved to {stab_path}")
        except Exception as e:
            logger.warning(f"Failed to save stability matrix: {e}")

    if num_unstable > 0:
        logger.info(f"Filtering out {num_unstable} unstable samples before output export")
    valid_positions = np.nonzero(sample_valid_mask)[0].tolist()
    filtered_indices = [all_indices[i] for i in valid_positions]

    def _filter_first_axis(arr):
        return arr[sample_valid_mask]

    all_pred_params_filtered = {}
    all_label_params_filtered = {}
    for param in base_params:
        if param in all_pred_params:
            all_pred_params_filtered[param] = _filter_first_axis(all_pred_params[param])
        if param in all_label_params:
            all_label_params_filtered[param] = _filter_first_axis(all_label_params[param])
    all_masks_filtered = _filter_first_axis(all_masks)
    all_actual_times_filtered = _filter_first_axis(all_actual_times)

    # Normalize actual_time to [sample, time_step] datetime64[ns].
    all_actual_times_filtered = np.asarray(all_actual_times_filtered)
    if all_actual_times_filtered.ndim == 0:
        all_actual_times_filtered = all_actual_times_filtered.reshape(1, 1)
    elif all_actual_times_filtered.ndim == 1:
        all_actual_times_filtered = all_actual_times_filtered.reshape(1, -1)
    elif all_actual_times_filtered.ndim > 2:
        all_actual_times_filtered = all_actual_times_filtered.reshape(all_actual_times_filtered.shape[0], -1)
    try:
        all_actual_times_filtered = np.array(
            [pd.to_datetime(row).to_numpy(dtype='datetime64[ns]') for row in all_actual_times_filtered],
            dtype='datetime64[ns]',
        )
    except Exception:
        pass
    all_forcings_filtered = _filter_first_axis(all_forcings)

    coords = {'sample': filtered_indices, 'time_step': range(lead_time), 'lon': ds_info['lon'], 'lat': ds_info['lat']}
    data_vars = {}
    resolved_output = _resolve_output_params(output_params)
    if resolved_output is None:
        params_to_save = list(all_pred_params_filtered.keys())
    else:
        params_to_save = [p for p in resolved_output if p in all_pred_params_filtered]

    for param in params_to_save:
        if param in all_pred_params_filtered:
            data_vars[f'{param}_pred'] = (('sample', 'time_step', 'lon', 'lat'), all_pred_params_filtered[param])
        if param in all_label_params_filtered:
            data_vars[f'{param}_true'] = (('sample', 'time_step', 'lon', 'lat'), all_label_params_filtered[param])
    data_vars['mask'] = (('sample', 'lon', 'lat'), all_masks_filtered.squeeze(-1))
    data_vars['actual_time'] = (('sample', 'time_step'), all_actual_times_filtered)

    if save_sample_results:
        ds = xr.Dataset(data_vars, coords=coords)
        ds.attrs['description'] = 'Wave integral parameters from neural wave model inference'
        ds.attrs['model_path'] = model_path if model_path else 'N/A'
        output_path = os.path.join(results_dir, 'integral_parameters', 'wave_parameters.nc')
        ds.to_netcdf(output_path)
        logger.info(f"Integral parameters saved to {output_path}")

def save_sample_spectra(
    sample_idx: int,
    predictions: np.ndarray,
    spec_label: np.ndarray,
    forcings: np.ndarray,
    bathy: np.ndarray,
    mask: np.ndarray,
    actual_times: np.ndarray,
    ds_info: Dict,
    results_dir: str
):
    """Save one sample's full spectra, forcings, and metadata to NetCDF."""
    actual_times_1d = np.asarray(actual_times)
    actual_times_1d = np.squeeze(actual_times_1d)
    if actual_times_1d.ndim != 1:
        actual_times_1d = actual_times_1d.reshape(-1)

    coords = {
        'time_step': range(len(actual_times_1d)),
        'lon': ds_info['lon'],
        'lat': ds_info['lat'],
        'direction': ds_info['theta'],
        'frequency': ds_info['freq']
    }

    data_vars = {
        'spectrum_pred': (('time_step', 'lon', 'lat', 'direction', 'frequency'), predictions),
        'spectrum_true': (('time_step', 'lon', 'lat', 'direction', 'frequency'), spec_label),
        'forcings': (('time_step', 'lon', 'lat', 'forcing_type'), forcings),
        'bathymetry': (('time_step', 'lon', 'lat'), bathy),
        'mask': (('lon', 'lat'), mask),
        'actual_time': (('time_step',), actual_times_1d)
    }
    
    sample_ds = xr.Dataset(data_vars, coords=coords)
    sample_ds.attrs['sample_index'] = sample_idx
    sample_ds.attrs['description'] = f'Detailed wave spectra for sample {sample_idx}'

    output_path = os.path.join(results_dir, "sample_spectra", f"sample_{sample_idx:06d}.nc")
    sample_ds.to_netcdf(output_path)

    logger.info(f"Sample {sample_idx} spectra saved to {output_path}")

def _denormalize_forcings(forcings, norm_factors):
    """Denormalize forcing fields using the stored statistics."""
    if norm_factors is None:
        return forcings

    forcings_denorm = forcings.copy()
    forcings_denorm[:, :, :, :, 0] = (forcings[:, :, :, :, 0] - norm_factors['spectrum']['norm_mean']) / norm_factors['spectrum']['norm_std'] * norm_factors['bathymetry']['norm_std'] + norm_factors['bathymetry']['norm_mean']
    forcings_denorm[:, :, :, :, 1] = (forcings[:, :, :, :, 1] - norm_factors['spectrum']['norm_mean']) / norm_factors['spectrum']['norm_std'] * norm_factors['u_wind']['norm_std'] + norm_factors['u_wind']['norm_mean']
    forcings_denorm[:, :, :, :, 2] = (forcings[:, :, :, :, 2] - norm_factors['spectrum']['norm_mean']) / norm_factors['spectrum']['norm_std'] * norm_factors['v_wind']['norm.std'] + norm_factors['v_wind']['norm_mean']

    return forcings_denorm

def _rmse_masked(pred: np.ndarray, label: np.ndarray, mask: np.ndarray):
    """
    Compute RMSE with mask applied. Returns (rmse, sum_squared_error, n_samples).
    pred/label: [B, L, W, H], mask: broadcastable to pred shape (bool)
    """
    # Align mask shape and ensure boolean
    m = mask.astype(bool)
    # Remove NaNs
    valid = m & (~np.isnan(pred)) & (~np.isnan(label))
    n = int(valid.sum())
    if n == 0:
        return np.nan, 0.0, 0
    se = (pred[valid] - label[valid]) ** 2
    return float(np.sqrt(se.mean())), float(se.sum()), n

def _circular_rmse_deg_masked(pred_deg: np.ndarray, label_deg: np.ndarray, mask: np.ndarray):
    """
    Circular RMSE in degrees using vector method, with mask. Returns (rmse, sum_squared_error, n_samples).
    """
    m = mask.astype(bool)
    # Convert to radians
    pr = np.deg2rad(pred_deg)
    lr = np.deg2rad(label_deg)
    # Unit vectors
    px, py = np.cos(pr), np.sin(pr)
    lx, ly = np.cos(lr), np.sin(lr)
    cos_d = np.clip(px * lx + py * ly, -1.0, 1.0)
    delta_deg = np.rad2deg(np.arccos(cos_d))
    valid = m & (~np.isnan(delta_deg))
    n = int(valid.sum())
    if n == 0:
        return np.nan, 0.0, 0
    se = (delta_deg[valid]) ** 2
    return float(np.sqrt(se.mean())), float(se.sum()), n

def _extract_checkpoint_state_dict(checkpoint):
    """Support raw state_dict checkpoints and wrapped training checkpoints."""
    if isinstance(checkpoint, dict):
        for key in ('model_state_dict', 'state_dict'):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint

def _normalize_checkpoint_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip known wrapper prefixes so compile-trained checkpoints can load in eager mode."""
    normalized_state = {}
    stripped_prefix = False

    for key, value in state_dict.items():
        normalized_key = key
        for prefix in ('_orig_mod.', 'module.'):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix):]
                stripped_prefix = True
        normalized_state[normalized_key] = value

    if stripped_prefix:
        logger.info('Normalized checkpoint keys for eager model loading.')
    return normalized_state

def load_model(model_path: Optional[str], config_file: str, ds_info: Dict, device: torch.device, compile_model: bool = True) -> nn.Module:
    """Load the model, configure the wave environment, and optionally compile it."""
    logger.info(f"Loading gin config from {config_file}")
    gin.clear_config()
    gin.parse_config_file(config_file)

    logger.info("Initializing model...")
    model = NeuralWave()
    theta_deg = np.asarray(ds_info['theta'])
    theta_rad = ((180.0 - theta_deg) % 360.0) * np.pi / 180.0
    model.set_wave_environment(
        lon=torch.tensor(np.asarray(ds_info['lon']), dtype=torch.float32, device=device),
        lat=torch.tensor(np.asarray(ds_info['lat']), dtype=torch.float32, device=device),
        theta=torch.tensor(theta_rad, dtype=torch.float32, device=device),
        freq=torch.tensor(np.asarray(ds_info['freq']), dtype=torch.float32, device=device),
    )
    model.to(device)
    model.eval()

    if model_path:
        logger.info(f"Loading model weights from {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        state_dict = _extract_checkpoint_state_dict(state_dict)
        state_dict = _normalize_checkpoint_keys(state_dict)
        model.load_state_dict(state_dict)

    if compile_model:
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            logger.info("Model compiled for inference.")
        except Exception as e:
            logger.warning(f"torch.compile failed; using eager mode. Reason: {e}")

    return model

def run_model_inference(
    model: nn.Module,
    batch_data: Dict,
    lead_time: int,
    use_compile: bool = True,
    device: torch.device = torch.device("cpu"),
    input_norm: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Run one autoregressive rollout for a batch."""
    spec_ini = batch_data['spec_ini']
    forcings = batch_data['forcings']
    bathy = batch_data['bathy']
    spec_label = batch_data['spec_label']
    mask_full = batch_data['mask']

    if spec_ini.ndim == 6 and spec_ini.shape[1] == 1:
        spec_next = spec_ini[:, 0]
    else:
        spec_next = spec_ini

    spec_output_rollout = torch.zeros_like(spec_label, device=spec_next.device)

    for step in range(lead_time):
        environments = {
            'depth': bathy[:, step].squeeze(-1).to(device),
            'bathymetry': forcings[:, step, :, :, 0].to(device),
            'wind_u': forcings[:, step, :, :, 1].to(device),
            'wind_v': forcings[:, step, :, :, 2].to(device),
            'input_norm': input_norm,
        }

        boundary_condition = spec_label[:, step].clone()
        boundary_condition[:, 1:-1, 1:-1] = 0
        mask = mask_full.squeeze(-1).squeeze(1).clone()
        mask[boundary_condition.sum(dim=(-2, -1)) > 0] = 0

        if use_compile:
            torch.compiler.cudagraph_mark_step_begin()

        spec_next = model(
            spectra=spec_next.clone().to(device),
            environments=environments,
            mask=mask.to(device),
            boundary_condition=boundary_condition.to(device),
        )
        spec_output_rollout[:, step] = spec_next.cpu()

    return spec_output_rollout

def run_inference(
    data_path: Optional[str] = None,
    model_path: Optional[str] = None,
    config_file: Optional[str] = None,
    results_dir: Optional[str] = None,
    batch_size: int = 1,
    lead_time: int = 48,
    sample_indices: Optional[List[int]] = None,
    save_spectra_samples: Optional[List[int]] = None,
    device_str: str = "cuda:0",
    compile_model: bool = True,
    save_sample_results: bool = False,
    rescale_forcings: bool = True,
    norm_stats_path: str = "./data/data_validate_norm_stats.json",
    spectrum_dir: Optional[str] = None,
    wave_nc: Optional[str] = None,
    wind_nc: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    input_norm_path: Optional[str] = None,
    output_params: Optional[set[str] | str] = None,
    save_stability_matrix: bool = False,
):
    """Public entrypoint for source-data inference."""
    if not (spectrum_dir and wave_nc and wind_nc and start_date and end_date):
        raise ValueError("Please provide spectrum_dir, wave_nc, wind_nc, start_date, end_date for new loader inference.")

    output_params_set = _parse_output_params(output_params)
    return run_inference_with_loader(
        spectrum_dir=spectrum_dir,
        wave_nc=wave_nc,
        wind_nc=wind_nc,
        model_path=model_path,
        config_file=config_file,
        results_dir=results_dir,
        start_date=start_date,
        end_date=end_date,
        batch_size=batch_size,
        lead_time=lead_time,
        sample_indices=sample_indices,
        device_str=device_str,
        compile_model=compile_model,
        save_sample_results=save_sample_results,
        save_spectra_samples=save_spectra_samples,
        rescale_forcings=rescale_forcings,
        norm_stats_path=norm_stats_path,
        input_norm_path=input_norm_path,
        output_params=output_params_set,
        save_stability_matrix=save_stability_matrix,
    )


def run_inference_with_loader(
    spectrum_dir: str,
    wave_nc: str,
    wind_nc: str,
    model_path: str,
    config_file: str,
    results_dir: str,
    start_date: str,
    end_date: str,
    batch_size: int,
    lead_time: int,
    sample_indices: Optional[List[int]] = None,
    device_str: str = "cuda:0",
    compile_model: bool = True,
    save_sample_results: bool = False,
    save_spectra_samples: Optional[List[int]] = None,
    rescale_forcings: bool = True,
    norm_stats_path: str = "./data/data_validate_norm_stats.json",
    input_norm_path: Optional[str] = None,
    output_params: Optional[set[str]] = None,
    save_stability_matrix: bool = False,
):
    """Run inference with the public source-data dataloader."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    setup_directories(
        results_dir,
        save_stability_matrix=save_stability_matrix,
        save_spectra_samples=bool(save_spectra_samples),
    )

    time_index = pd.date_range(start=start_date, end=end_date, freq='H')
    total_samples = len(time_index) - lead_time
    if sample_indices is None:
        sample_indices = list(range(0, max(0, total_samples)))

    infer_loader = create_infer_dataloader(
        spectrum_dir=spectrum_dir,
        wave_nc=wave_nc,
        wind_nc=wind_nc,
        sample_indices=sample_indices,
        time_index=time_index,
        lead_time=lead_time,
        batch_size=batch_size,
        num_workers=0,
        rescale_forcings=rescale_forcings,
        norm_stats_path=norm_stats_path,
        sparse_mask_path=None,
    )

    # Prefer the spectrum coordinates exposed by the dataset when available.
    first_batch = next(iter(infer_loader))
    spec_input, forcings, bathy, spec_label, mask, sparse_mask, batch_time_strs = first_batch
    ds_obj = getattr(infer_loader, 'dataset', None)
    if ds_obj is not None and hasattr(ds_obj, 'spec_lon') and hasattr(ds_obj, 'spec_lat'):
        lon_vals = np.asarray(ds_obj.spec_lon)
        lat_vals = np.asarray(ds_obj.spec_lat)
    else:
        lon_vals = np.arange(mask.shape[2])
        lat_vals = np.arange(mask.shape[3])
    ds_info = {
        'lon': lon_vals,
        'lat': lat_vals,
        'theta': np.arange(7.5, 352.5 + 15, 15),
        'freq': np.full(30, 0.03453) * (1.1 ** np.arange(0, 30)),
        'num_sequences': len(sample_indices),
    }

    model = load_model(model_path, config_file, ds_info, device, compile_model=compile_model)

    input_norm = None
    if input_norm_path:
        if os.path.exists(input_norm_path):
            input_norm = load_input_norm(input_norm_path, device)
            logger.info(f"Loaded input normalization stats from {input_norm_path}")
        else:
            logger.warning(f"Input normalization file not found: {input_norm_path}; proceeding without input_norm")

    integral_params_list = []
    resolved_output = _resolve_output_params(output_params)
    compute_stokes = resolved_output is not None and (('stokes_u' in resolved_output) or ('stokes_v' in resolved_output))

    pbar = tqdm(infer_loader, desc="Inference Batches")
    for b_idx, batch in enumerate(pbar):
        spec_input, forcings, bathy, spec_label, mask, sparse_mask, batch_time_strs = batch

        batch_data = {
            'spec_ini': spec_input,
            'forcings': forcings,
            'bathy': bathy,
            'spec_label': spec_label,
            'mask': mask,
            'time_strs': batch_time_strs,
        }

        with torch.no_grad():
            spec_output_rollout = run_model_inference(
                model=model,
                batch_data=batch_data,
                lead_time=lead_time,
                use_compile=compile_model,
                device=device,
                input_norm=input_norm,
            )

        spec_output_np = spec_output_rollout.detach().cpu().numpy()
        spec_label_np = spec_label.detach().cpu().numpy()

        pred_params = calculate_wave_parameters(spec_output_np, ds_info['freq'], ds_info['theta'])
        label_params = calculate_wave_parameters(spec_label_np, ds_info['freq'], ds_info['theta'])

        if compute_stokes:
            bathy_np = bathy.detach().cpu().numpy()
            if bathy_np.ndim == 5:
                depth_np = bathy_np[..., 0]
            else:
                depth_np = bathy_np
            stokes_pred = calculate_stokes_drift_vector(spec_output_np, ds_info['freq'], ds_info['theta'], depth_np, use_dask=False)
            stokes_label = calculate_stokes_drift_vector(spec_label_np, ds_info['freq'], ds_info['theta'], depth_np, use_dask=False)
            pred_params['stokes_u'] = stokes_pred['u_st']
            pred_params['stokes_v'] = stokes_pred['v_st']
            label_params['stokes_u'] = stokes_label['u_st']
            label_params['stokes_v'] = stokes_label['v_st']

        mask_np = mask.squeeze(1).detach().cpu().numpy()
        if mask_np.ndim == 4:
            mask_2d = np.squeeze(mask_np, axis=-1)
        else:
            mask_2d = mask_np
        mask_exp = np.broadcast_to(mask_2d[:, None, :, :], pred_params['swh'].shape)

        swh_rmse, _, _ = _rmse_masked(pred_params['swh'], label_params['swh'], mask_exp)
        mwp_rmse, _, _ = _rmse_masked(pred_params['mwp'], label_params['mwp'], mask_exp)
        mwd_rmse, _, _ = _circular_rmse_deg_masked(pred_params['mwd'], label_params['mwd'], mask_exp)

        pbar.set_postfix(start_time=str(batch_time_strs[0]), swh_rmse=f"{swh_rmse:.3f}m", mwd_rmse=f"{mwd_rmse:.2f}°", mwp_rmse=f"{mwp_rmse:.3f}s")

        start = b_idx * batch_size
        end = start + spec_input.shape[0]
        batch_indices = sample_indices[start:end]

        bt = np.array(batch_time_strs)
        actual_times_b = np.tile(bt[None, :], (len(batch_indices), 1))

        nan_present = np.isnan(spec_output_rollout.detach().cpu().numpy()).any(axis=tuple(range(2, spec_output_rollout.ndim)))
        stability_flags = (~nan_present).astype(np.uint8)  # [B, L]

        if save_spectra_samples:
            predictions_np = spec_output_rollout.detach().cpu().numpy()
            spec_label_np = spec_label.detach().cpu().numpy()
            forcings_np = forcings.detach().cpu().numpy()
            bathy_np = bathy.detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy()

            for j, sample_idx in enumerate(batch_indices):
                if (sample_idx in save_spectra_samples) and (stability_flags[j].sum() == lead_time):
                    save_sample_spectra(
                        sample_idx=sample_idx,
                        predictions=predictions_np[j],
                        spec_label=spec_label_np[j],
                        forcings=forcings_np[j],
                        bathy=bathy_np[j],
                        mask=mask_np[j,0,:,:,0],
                        actual_times=actual_times_b[j],
                        ds_info=ds_info,
                        results_dir=results_dir,
                    )
                elif sample_idx in save_spectra_samples:
                    logger.info(f"Skip saving spectra for sample {sample_idx} due to instability (NaNs detected)")

        integral_params_list.append({
            'indices': batch_indices,
            'predictions': pred_params,
            'labels': label_params,
            'mask': mask.squeeze(1).detach().cpu().numpy(),
            'actual_times': actual_times_b,
            'forcings': forcings.detach().cpu().numpy(),
            'stability': stability_flags,
        })

    save_results_and_visualize(
        params_list=integral_params_list,
        results_dir=results_dir,
        ds_info=ds_info,
        lead_time=lead_time,
        model_path=model_path,
        save_sample_results=save_sample_results,
        output_params=output_params,
        save_stability_matrix=save_stability_matrix,
    )

    logger.info("Inference (new dataloader) finished successfully!")

def parse_sample_indices(indices_str: str, max_samples: int) -> List[int]:
    """Parse sample indices from string format"""
    if indices_str is None:
        return None

    if ':' in indices_str:
        parts = indices_str.split(':')
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else max_samples
        step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
        return list(range(start, min(end, max_samples), step))
    else:
        return [int(x.strip()) for x in indices_str.split(',')]

def _parse_output_params(output_params: Optional[set[str] | str]) -> Optional[set[str]]:
    if output_params is None:
        return set(DEFAULT_OUTPUT_PARAMS)
    if isinstance(output_params, set):
        return {str(p).strip().lower() for p in output_params}
    if isinstance(output_params, (list, tuple)):
        return {str(p).strip().lower() for p in output_params}
    s = str(output_params).strip()
    if not s or s.lower() in ("all", "*"):
        return None
    try:
        import ast
        if s[0] in "{[()":
            val = ast.literal_eval(s)
            if isinstance(val, (set, list, tuple)):
                return {str(p).strip().lower() for p in val}
            return {str(val).strip().lower()}
    except Exception:
        pass
    parts = [p.strip() for p in s.replace(';', ',').split(',') if p.strip()]
    return {p.lower() for p in parts} if parts else None

def _resolve_output_params(output_params: Optional[set[str]]) -> Optional[set[str]]:
    if output_params is None:
        return None
    resolved = set()
    for p in output_params:
        if p in {'stokes', 'stokes_drift', 'stokes_drift_vector'}:
            resolved.update({'stokes_u', 'stokes_v'})
        elif p in {'u_st', 'stokes_u'}:
            resolved.add('stokes_u')
        elif p in {'v_st', 'stokes_v'}:
            resolved.add('stokes_v')
        else:
            resolved.add(p)
    return resolved

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Wave Inference From Source Data')
    parser.add_argument('--spectrum-dir', type=str, required=True, help='Directory of spectrum NetCDF files')
    parser.add_argument('--wave-nc', type=str, required=True, help='Path to ERA5 wave NetCDF')
    parser.add_argument('--wind-nc', type=str, required=True, help='Path to ERA5 wind NetCDF')
    parser.add_argument('--start-date', type=str, required=True, help='Start datetime, e.g., 2022-01-03 00:00:00')
    parser.add_argument('--end-date', type=str, required=True, help='End datetime, e.g., 2022-01-07 23:00:00')

    parser.add_argument('--model-path', type=str, default=None, help='Path to trained model .pth (optional)')
    parser.add_argument('--config-file', type=str, required=True, help='Path to gin config file')
    parser.add_argument('--results-dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lead-time', type=int, default=48)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile')
    parser.add_argument('--save-sample-nc', action='store_true', help='Save per-sample integral params NetCDF')
    parser.add_argument('--rescale-forcings', action='store_true', help='Use raw normalized forcing values (no denorm)')
    parser.add_argument('--norm-stats-path', type=str, default='./data/data_validate_norm_stats.json')
    parser.add_argument('--input-norm-path', type=str, default=None, help='Path to input normalization stats JSON (mean/std)')

    parser.add_argument('--sample-indices', type=str, default=None, help="Indices format 'start:end:step' or 'i,j,k'")
    parser.add_argument('--save-spectra-samples', type=str, default=None, help='Samples to save full spectra NetCDF')
    parser.add_argument('--output-params', type=str, default=None, help="Params to save, e.g. 'swh,mwd,stokes' or '{""mwp"", ""mwd"", ""stokes""}'")
    parser.add_argument('--save-stability-matrix', action='store_true', help='Save per-sample stability diagnostics')

    args = parser.parse_args()

    sample_indices = parse_sample_indices(args.sample_indices, max_samples=10**9) if args.sample_indices else None
    spectra_samples = parse_sample_indices(args.save_spectra_samples, max_samples=10**9) if args.save_spectra_samples else None

    logger.info("Starting inference via CLI (new loader)")

    run_inference(
        model_path=args.model_path,
        config_file=args.config_file,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
        lead_time=args.lead_time,
        sample_indices=sample_indices,
        save_spectra_samples=spectra_samples,
        device_str=args.device,
        compile_model=args.compile,
        save_sample_results=args.save_sample_nc,
        rescale_forcings=args.rescale_forcings,
        norm_stats_path=args.norm_stats_path,
        input_norm_path=args.input_norm_path,
        output_params=args.output_params,
        save_stability_matrix=args.save_stability_matrix,
        spectrum_dir=args.spectrum_dir,
        wave_nc=args.wave_nc,
        wind_nc=args.wind_nc,
        start_date=args.start_date,
        end_date=args.end_date,
    )
