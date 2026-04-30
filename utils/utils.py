import numpy as np
import dask.array as da
import torch
from typing import Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def get_integral_m_numpy(spectrum_distribution, df, dtheta, frequencies, order: int=0, dim: tuple=(-2,-1)):
        """Compute spectral moments with NumPy."""
        return np.sum(spectrum_distribution * df * dtheta * (frequencies ** order), axis=dim)
    
def get_integral_m_dask(spectrum_distribution, df, dtheta, frequencies, order: int=0, dim: tuple=(-2,-1)):
    """Compute spectral moments with Dask."""
    return da.sum(spectrum_distribution * df * dtheta * (frequencies ** order), axis=dim)

def calculate_wave_parameters(spectra: np.ndarray, freq: np.ndarray, theta: np.ndarray, use_dask: bool = True) -> Dict[str, np.ndarray]:
    """
    Compute integral wave parameters from spectra.
    
    Args:
        spectra: Spectral data [..., D, F].
        freq: Frequency array.
        theta: Direction array in degrees.
        use_dask: Whether to use Dask for large arrays.
        
    Returns:
        Dictionary containing SWH, MWD, and MWP.
    """
    dtheta = np.abs(theta[1] - theta[0]) * np.pi / 180.0
    df = np.diff(freq, prepend=freq[0:1])
    
    freq_array = freq[None, ...]
    theta_rad = theta * np.pi / 180.0
    
    if use_dask and spectra.size > 1e6:
        spectra_da = da.from_array(spectra, chunks='auto')
        df_da = da.from_array(df, chunks='auto')
        dtheta_da = da.from_array(np.array(dtheta), chunks='auto')
        freq_da = da.from_array(freq_array, chunks='auto')
        theta_rad_da = da.from_array(theta_rad, chunks='auto')
        
        m0 = get_integral_m_dask(spectra_da, df_da, dtheta_da, freq_da, order=0)
        swh = 4 * da.sqrt(m0 + 1e-16)
        
        cos_theta_weighted = da.cos(theta_rad_da)[..., None] * spectra_da * df_da * dtheta_da
        sin_theta_weighted = da.sin(theta_rad_da)[..., None] * spectra_da * df_da * dtheta_da
        
        CF = da.sum(cos_theta_weighted, axis=(-2, -1))
        SF = da.sum(sin_theta_weighted, axis=(-2, -1))
        
        mwd = da.rad2deg(da.arctan2(SF + 1e-16, CF + 1e-16) + np.pi) % 360
        
        m_neg1 = get_integral_m_dask(spectra_da, df_da, dtheta_da, freq_da, order=-1)
        mwp = m_neg1 / (m0 + 1e-16)
        
        return {
            'swh': swh.compute(),
            'mwd': mwd.compute(),
            'mwp': mwp.compute()
        }
    else:
        m0 = get_integral_m_numpy(spectra, df, dtheta, freq_array, order=0)
        swh = 4 * np.sqrt(m0 + 1e-16)
        
        cos_theta_weighted = np.cos(theta_rad)[..., None] * spectra * df * dtheta
        sin_theta_weighted = np.sin(theta_rad)[..., None] * spectra * df * dtheta
        
        CF = np.sum(cos_theta_weighted, axis=(-2, -1))
        SF = np.sum(sin_theta_weighted, axis=(-2, -1))
        
        mwd = np.rad2deg(np.arctan2(SF + 1e-16, CF + 1e-16) + np.pi) % 360
        
        m_neg1 = get_integral_m_numpy(spectra, df, dtheta, freq_array, order=-1)
        mwp = m_neg1 / (m0 + 1e-16)
        
        return {
            'swh': swh,
            'mwd': mwd,
            'mwp': mwp
        }

def _wavenumber_from_omega_np(omega: np.ndarray, depth: np.ndarray, g: float = 9.81, iters: int = 20) -> np.ndarray:
    """Compute wavenumber from angular frequency and depth with NumPy."""
    depth_arr = np.asarray(depth)
    omega_arr = np.asarray(omega)
    if depth_arr.ndim == 0:
        depth_exp = depth_arr
        omega_exp = omega_arr
    else:
        depth_exp = depth_arr[..., None]
        omega_exp = omega_arr.reshape((1,) * depth_arr.ndim + omega_arr.shape)

    k = (omega_exp ** 2) / g
    if np.all(depth_exp <= 0):
        return k
    for _ in range(iters):
        tanh_kd = np.tanh(k * depth_exp)
        f = g * k * tanh_kd - omega_exp ** 2
        df = g * tanh_kd + g * k * depth_exp * (1 - tanh_kd ** 2)
        k = k - f / (df + 1e-16)
    return k

def _wavenumber_from_omega_da(omega: da.Array, depth: da.Array, g: float = 9.81, iters: int = 20) -> da.Array:
    """Compute wavenumber from angular frequency and depth with Dask."""
    depth_arr = depth
    omega_arr = omega
    if depth_arr.ndim == 0:
        depth_exp = depth_arr
        omega_exp = omega_arr
    else:
        depth_exp = depth_arr[..., None]
        omega_exp = omega_arr.reshape((1,) * depth_arr.ndim + omega_arr.shape)

    k = (omega_exp ** 2) / g
    for _ in range(iters):
        tanh_kd = da.tanh(k * depth_exp)
        f = g * k * tanh_kd - omega_exp ** 2
        df = g * tanh_kd + g * k * depth_exp * (1 - tanh_kd ** 2)
        k = k - f / (df + 1e-16)
    return k

def calculate_stokes_drift_vector(
    spectra: np.ndarray,
    freq: np.ndarray,
    theta: np.ndarray,
    depth: np.ndarray,
    g: float = 9.81,
    use_dask: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute the surface Stokes drift vector (u, v).
    u = ∫∫ W(f, D) F(f,θ) cosθ df dθ
    v = ∫∫ W(f, D) F(f,θ) sinθ df dθ
    W(f, D) = [2 g k / (ω tanh(2 k D))] k
    """
    dtheta = np.abs(theta[1] - theta[0]) * np.pi / 180.0
    df = np.diff(freq, prepend=freq[0:1])
    theta_rad = theta * np.pi / 180.0
    omega = 2 * np.pi * freq

    if use_dask and spectra.size > 1e6:
        spectra_da = da.from_array(spectra, chunks='auto')
        df_da = da.from_array(df, chunks='auto')
        dtheta_da = da.from_array(np.array(dtheta), chunks='auto')
        theta_da = da.from_array(theta_rad, chunks='auto')
        omega_da = da.from_array(omega, chunks='auto')
        depth_da = da.from_array(np.asarray(depth), chunks='auto')

        k_da = _wavenumber_from_omega_da(omega_da, depth_da, g=g)
        weight_da = (2 * g * k_da / (omega_da * da.tanh(2 * k_da * depth_da[..., None]) + 1e-16)) * k_da
        weight_full = weight_da * df_da.reshape((1,) * (weight_da.ndim - 1) + df_da.shape) * dtheta_da

        cos_term = da.cos(theta_da)[..., None]
        sin_term = da.sin(theta_da)[..., None]
        u = da.sum(spectra_da * weight_full * cos_term, axis=(-2, -1))
        v = da.sum(spectra_da * weight_full * sin_term, axis=(-2, -1))
        return {'u_st': u.compute(), 'v_st': v.compute()}
    else:
        k = _wavenumber_from_omega_np(omega, depth, g=g)
        weight = (2 * g * k / (omega.reshape((1,) * (k.ndim - 1) + omega.shape) * np.tanh(2 * k * np.asarray(depth)[..., None]) + 1e-16)) * k
        df_b = df.reshape((1,) * (weight.ndim - 1) + df.shape)
        weight_full = weight * df_b * dtheta

        weight_full = weight_full[:,:,:,:,None,:]
        cos_term = np.cos(theta_rad)[None, None, None, None, :, None]
        sin_term = np.sin(theta_rad)[None, None, None, None, :, None]
        u = np.sum(spectra * weight_full * cos_term, axis=(-2, -1))
        v = np.sum(spectra * weight_full * sin_term, axis=(-2, -1))
        return {'u_st': u, 'v_st': v}

def calculate_stokes_drift_vector_e2e(
    spectra: np.ndarray,
    freq: np.ndarray,
    theta: np.ndarray,
    depth: np.ndarray,
    g: float = 9.81,
    use_dask: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute the surface Stokes drift vector (u, v).
    u = ∫∫ W(f, D) F(f,θ) cosθ df dθ
    v = ∫∫ W(f, D) F(f,θ) sinθ df dθ
    W(f, D) = [2 g k / (ω tanh(2 k D))] k
    """
    dtheta = np.abs(theta[1] - theta[0]) * np.pi / 180.0
    df = np.diff(freq, prepend=freq[0:1])
    theta_rad = theta * np.pi / 180.0
    omega = 2 * np.pi * freq

    if use_dask and spectra.size > 1e6:
        spectra_da = da.from_array(spectra, chunks='auto')
        df_da = da.from_array(df, chunks='auto')
        dtheta_da = da.from_array(np.array(dtheta), chunks='auto')
        theta_da = da.from_array(theta_rad, chunks='auto')
        omega_da = da.from_array(omega, chunks='auto')
        depth_da = da.from_array(np.asarray(depth), chunks='auto')

        k_da = _wavenumber_from_omega_da(omega_da, depth_da, g=g)
        weight_da = (2 * g * k_da / (omega_da * da.tanh(2 * k_da * depth_da[..., None]) + 1e-16)) * k_da
        weight_full = weight_da * df_da.reshape((1,) * (weight_da.ndim - 1) + df_da.shape) * dtheta_da

        cos_term = da.cos(theta_da)[..., None]
        sin_term = da.sin(theta_da)[..., None]
        u = da.sum(spectra_da * weight_full * cos_term, axis=(-2, -1))
        v = da.sum(spectra_da * weight_full * sin_term, axis=(-2, -1))
        return {'u_st': u.compute(), 'v_st': v.compute()}
    else:
        k = _wavenumber_from_omega_np(omega, depth, g=g)
        weight = (2 * g * k / (omega.reshape((1,) * (k.ndim - 1) + omega.shape) * np.tanh(2 * k * np.asarray(depth)[..., None]) + 1e-16)) * k
        df_b = df.reshape((1,) * (weight.ndim - 1) + df.shape)
        weight_full = weight * df_b * dtheta

        weight_full = weight_full[:,:,:,None,:]
        cos_term = np.cos(theta_rad)[None, None, None, :, None]
        sin_term = np.sin(theta_rad)[None, None, None, :, None]
        u = np.sum(spectra * weight_full * cos_term, axis=(-2, -1))
        v = np.sum(spectra * weight_full * sin_term, axis=(-2, -1))
        return {'u_st': u, 'v_st': v}
    

def _calculate_overall_metrics_from_params(self, pred_params: Dict, label_params: Dict, masks: np.ndarray) -> Dict:
    """
    Compute overall metrics from precomputed integral parameters.
    """
    metrics = {}
    
    for param in ['swh', 'mwd', 'mwp']:
        pred = pred_params[param]
        label = label_params[param]
        
        if masks.ndim == 3:  # [sample, lon, lat]
            mask_expanded = masks[:, None, :, :]  # [sample, 1, lon, lat]
            mask_expanded = np.broadcast_to(mask_expanded, pred.shape)
        else:
            mask_expanded = masks
        
        pred_flat = pred[mask_expanded]
        label_flat = label[mask_expanded]
        
        if len(pred_flat) > 0:
            metrics[f'{param}_mse'] = mean_squared_error(label_flat, pred_flat)
            metrics[f'{param}_rmse'] = np.sqrt(metrics[f'{param}_mse'])
            metrics[f'{param}_mae'] = mean_absolute_error(label_flat, pred_flat)
            
            try:
                metrics[f'{param}_r2'] = r2_score(label_flat, pred_flat)
            except:
                metrics[f'{param}_r2'] = np.nan
        else:
            for metric_name in ['mse', 'rmse', 'mae', 'r2']:
                metrics[f'{param}_{metric_name}'] = np.nan
    
    return metrics

def calculate_batch_metrics(pred_params: Dict, label_params: Dict, mask: torch.Tensor) -> Dict:
    """Compute batch-level metrics."""
    metrics = {}
    
    for param in ['swh', 'mwd', 'mwp']:
        pred = pred_params[param]
        label = label_params[param]
        
        mask_np = mask.squeeze(-1).unsqueeze(1).repeat([1,pred.shape[1],1,1]).cpu().numpy()
        
        pred_flat = pred[mask_np]
        label_flat = label[mask_np]
        
        if len(pred_flat) > 0:
            metrics[f'{param}_mse'] = mean_squared_error(label_flat, pred_flat)
            metrics[f'{param}_rmse'] = np.sqrt(metrics[f'{param}_mse'])
            metrics[f'{param}_mae'] = mean_absolute_error(label_flat, pred_flat)
            
            try:
                metrics[f'{param}_r2'] = r2_score(label_flat, pred_flat)
            except:
                metrics[f'{param}_r2'] = np.nan
        else:
            for metric_name in ['mse', 'rmse', 'mae', 'r2']:
                metrics[f'{param}_{metric_name}'] = np.nan
    
    return metrics


if __name__ == '__main__':
    import argparse
    import pandas as pd
    import sys
    import os
    # Add the project root to the Python path.
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from dataset_source import WaveInferDataset

    parser = argparse.ArgumentParser(description='Test wave integral params & Stokes drift vector with one batch sample')
    parser.add_argument('--spectrum-dir', type=str, help='Directory of spectrum NetCDF/GRIB files',
                        default="../DataCollectionAndPreprocess/data/SouthChinaSea/wave_spectrum_nc_2023")
    parser.add_argument('--wave-nc', type=str,  help='Path to ERA5 wave NetCDF',
                        default="../DataCollectionAndPreprocess/nc_source_data/global/ERA5_2023_wave_global.nc")
    parser.add_argument('--wind-nc', type=str, help='Path to ERA5 wind NetCDF',
                        default="../DataCollectionAndPreprocess/nc_source_data/global/ERA5_2023_wind_global.nc")
    parser.add_argument('--start-date', type=str, help='Start datetime, e.g., 2022-01-03 00:00:00',
                        default="2023-01-03 00:00:00")
    parser.add_argument('--lead-time', type=int, default=2)
    parser.add_argument('--rescale-forcings', action='store_true')
    args = parser.parse_args()

    time_index = pd.date_range(start=args.start_date, periods=args.lead_time + 1, freq='H')
    indices = [0]

    dataset = WaveInferDataset(
        spectrum_root_dir=args.spectrum_dir,
        wave_nc_path=args.wave_nc,
        wind_nc_path=args.wind_nc,
        indices=indices,
        time_index=time_index,
        lead_time=args.lead_time,
        rescale_forcings=args.rescale_forcings,
        norm_stats_path=None,
        sparse_mask_path=None,
    )

    spec_input, forcings, bathy, spec_label, mask, s_mask, time_strs = dataset[0]
    spec_input_np = spec_input.numpy()  # [1, Lon, Lat, D, F]
    depth_np = bathy.numpy()[0] if bathy.ndim == 3 else bathy.numpy()

    spec_input_np[np.isnan(spec_input_np)] = 0.0
    theta = np.arange(7.5, 352.5 + 15, 15)
    freq = np.full(30, 0.03453) * (1.1 ** np.arange(0, 30))

    params = calculate_wave_parameters(spec_input_np, freq, theta, use_dask=False)
    stokes_vec = calculate_stokes_drift_vector(spec_input_np, freq, theta, depth_np, use_dask=False)

    print('SWH/MWD/MWP shapes:', {k: v.shape for k, v in params.items()})
    print('Stokes vector shapes:', {k: v.shape for k, v in stokes_vec.items()})