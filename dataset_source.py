import os
import json
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings("ignore")


class WaveInferDataset(Dataset):
    def __init__(
        self,
        spectrum_root_dir,
        wave_nc_path,
        wind_nc_path,
        indices,
        time_index,
        lead_time=1,
        rescale_forcings=False,
        norm_stats_path=None,
        sparse_mask_path=None,
    ):
        super().__init__()
        self.spectrum_root_dir = spectrum_root_dir
        self.indices = indices
        self.time_index = time_index
        self.lead_time = lead_time
        self.rescale_forcings = rescale_forcings
        self.sparse_mask_path = sparse_mask_path

        # Lazy load ERA5 sources
        self.ds_wave = xr.open_dataset(wave_nc_path)
        self.ds_wind = xr.open_dataset(wind_nc_path)

        # Build spectrum dir map (auto-detect GRIB or NC)
        self.spec_mode, self.grib_dir_map, self.nc_year_dirs = self._detect_and_build_spec_map()

        # Normalization
        self.norm_factors = None
        if self.rescale_forcings and norm_stats_path:
            with open(norm_stats_path, 'r') as f:
                self.norm_factors = json.load(f)

        # Align grids
        self._align_spatial_grids()

        # Sparse mask optional
        self.sparse_mask_ds = None
        if self.sparse_mask_path:
            self.sparse_mask_ds = xr.open_dataset(self.sparse_mask_path)

    def _detect_and_build_spec_map(self):
        root = self.spectrum_root_dir
        base = os.path.basename(os.path.normpath(root))
        # Direct NC year dir
        if base.startswith('wave_spectrum_nc_'):
            return 'nc', {}, [root]
        # Year dir with day folders
        try:
            day_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and len(d) == 8 and d.isdigit()]
            for d in day_dirs:
                p = os.path.join(root, d)
                if any(fn.endswith('.nc') for fn in os.listdir(p)):
                    return 'nc', {}, [root]
        except Exception:
            pass
        # Upper dir containing year dirs
        nc_year_dirs = []
        try:
            for name in os.listdir(root):
                if name.startswith('wave_spectrum_nc_') and os.path.isdir(os.path.join(root, name)):
                    nc_year_dirs.append(os.path.join(root, name))
        except Exception:
            pass
        if nc_year_dirs:
            return 'nc', {}, sorted(nc_year_dirs)
        # Fallback to GRIB monthly dir map (or direct monthly dir with .grib files)
        dir_map = {}
        found_grib_dir = False
        for r, dirs, files in os.walk(root):
            grib_files = [f for f in files if f.endswith('.grib')]
            if grib_files:
                for f in grib_files:
                    parts = f.split('_')
                    if len(parts) >= 5:
                        try:
                            y = int(parts[3]); m = int(parts[4])
                            key = f"{y}-{m:02d}"
                            dir_map[key] = r
                            found_grib_dir = True
                        except Exception:
                            continue
                if found_grib_dir:
                    break
            for d in dirs:
                if d.startswith('wave_spectrum_') and not d.startswith('wave_spectrum_nc_'):
                    try:
                        ps = d.split('_')
                        y = int(ps[-2]); m = int(ps[-1])
                        key = f"{y}-{m:02d}"
                        dir_map[key] = os.path.join(r, d)
                        found_grib_dir = True
                    except Exception:
                        continue
            if found_grib_dir:
                break
        return 'grib', dir_map, []

    def _align_spatial_grids(self):
        raw_lat = self.ds_wave['latitude'].values
        raw_lon = self.ds_wave['longitude'].values

        # Probe one sample to get spectrum lon/lat
        if self.spec_mode == 'grib':
            sample_file = None
            for key in sorted(self.grib_dir_map.keys()):
                files = [f for f in os.listdir(self.grib_dir_map[key]) if f.endswith('.grib')]
                if files:
                    sample_file = os.path.join(self.grib_dir_map[key], files[0])
                    break
            if not sample_file:
                raise FileNotFoundError("No GRIB files found.")
            with xr.open_dataset(sample_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'stepType': 'instant'}}) as ds_spec:
                if 'longitude' in ds_spec.coords:
                    ds_spec = ds_spec.rename({'longitude': 'lon'})
                if 'latitude' in ds_spec.coords:
                    ds_spec = ds_spec.rename({'latitude': 'lat'})
                s_lon = ds_spec['lon'].values
                s_lat = ds_spec['lat'].values
                self.target_spec_shape = (len(s_lon), len(s_lat))
        else:
            sample_file = None
            for ydir in sorted(self.nc_year_dirs):
                day_dirs = [d for d in os.listdir(ydir) if os.path.isdir(os.path.join(ydir, d))]
                day_dirs.sort()
                for d in day_dirs:
                    files = [f for f in os.listdir(os.path.join(ydir, d)) if f.endswith('.nc')]
                    files.sort()
                    if files:
                        sample_file = os.path.join(ydir, d, files[0])
                        break
                if sample_file:
                    break
            if not sample_file:
                raise FileNotFoundError("No NC spectrum files found.")
            with xr.open_dataset(sample_file) as ds_spec:
                if 'longitude' in ds_spec.coords:
                    ds_spec = ds_spec.rename({'longitude': 'lon'})
                if 'latitude' in ds_spec.coords:
                    ds_spec = ds_spec.rename({'latitude': 'lat'})
                s_lon = ds_spec['lon'].values
                s_lat = ds_spec['lat'].values
                self.target_spec_shape = (len(s_lon), len(s_lat))
        # Expose for ds_info
        self.spec_lon = s_lon
        self.spec_lat = s_lat

        # Longitude convention reconciliation (0–360 vs -180–180)
        spec_lon_vals = s_lon
        forc_lon_vals = raw_lon
        target_is_360 = np.nanmax(spec_lon_vals) > 180.0
        forc_is_360 = np.nanmax(forc_lon_vals) > 180.0
        # Following Operator is masked as we found that the order of lon are the same actually.
        # if forc_is_360 != target_is_360:
        #     def convert_and_sort(ds):
        #         if target_is_360:
        #             new_lon = np.mod(ds['longitude'], 360.0)
        #         else:
        #             new_lon = ((ds['longitude'] + 180.0) % 360.0) - 180.0
        #         return ds.assign_coords(longitude=new_lon).sortby('longitude')
        #     self.ds_wave = convert_and_sort(self.ds_wave)
        #     self.ds_wind = convert_and_sort(self.ds_wind)
        #     raw_lon = self.ds_wave['longitude'].values

        # Intersections
        lon_min = max(np.nanmin(raw_lon), np.nanmin(s_lon))
        lon_max = min(np.nanmax(raw_lon), np.nanmax(s_lon))
        lat_min = max(np.nanmin(raw_lat), np.nanmin(s_lat))
        lat_max = min(np.nanmax(raw_lat), np.nanmax(s_lat))
        if not (lon_min < lon_max and lat_min < lat_max):
            raise ValueError("Spatial grids do not overlap!")

        # Slices
        self.s_lon_slice = self._make_slice(s_lon, lon_min, lon_max)
        self.s_lat_slice = self._make_slice(s_lat, lat_min, lat_max)
        self.w_lon_slice = self._make_slice(raw_lon, lon_min, lon_max)
        self.w_lat_slice = self._make_slice(raw_lat, lat_min, lat_max)
        self.u_lon_slice = self._make_slice(self.ds_wind['longitude'].values, lon_min, lon_max)
        self.u_lat_slice = self._make_slice(self.ds_wind['latitude'].values, lat_min, lat_max)

    def _make_slice(self, coords, vmin, vmax):
        if coords.size >= 2 and coords[1] < coords[0]:
            return slice(vmax, vmin)
        return slice(vmin, vmax)

    def _get_spec_path(self, timestamp):
        if self.spec_mode == 'grib':
            key = timestamp.strftime('%Y-%m')
            if key not in self.grib_dir_map:
                return None
            day_str = str(timestamp.day)
            filename = f"era5_wave_spectra_{timestamp.year}_{timestamp.month:02d}_{day_str}.grib"
            return os.path.join(self.grib_dir_map[key], filename)
        else:
            year_dir = self._get_nc_year_root(timestamp.year)
            day_dir = os.path.join(year_dir, timestamp.strftime('%Y%m%d'))
            filename = f"wave_spectrum_{timestamp.strftime('%Y_%m_%d')}_{timestamp.strftime('%H')}.nc"
            path = os.path.join(day_dir, filename)
            return path if os.path.exists(path) else None

    def _get_nc_year_root(self, year):
        if not self.nc_year_dirs:
            return self.spectrum_root_dir
        if len(self.nc_year_dirs) == 1:
            return self.nc_year_dirs[0]
        target = f"wave_spectrum_nc_{year}"
        for ydir in self.nc_year_dirs:
            if os.path.basename(os.path.normpath(ydir)) == target:
                return ydir
        return self.nc_year_dirs[0]

    def _normalize_var(self, data, var_name):
        if not self.norm_factors:
            return data
        stats = self.norm_factors[var_name]
        return (data - stats['norm_mean']) / (stats['norm_std'] + 1e-6)

    def __getitem__(self, idx):
        global_idx = self.indices[idx]

        # Spectrum sequence from GRIB
        spec_frames = []
        for i in range(self.lead_time + 1):
            ts = self.time_index[global_idx + i]
            path = self._get_spec_path(ts)
            frame_val = None
            if path and os.path.exists(path):
                try:
                    if self.spec_mode == 'grib':
                        with xr.open_dataset(path, engine='cfgrib', backend_kwargs={'filter_by_keys': {'stepType': 'instant'}}) as ds:
                            if 'longitude' in ds.coords:
                                ds = ds.rename({'longitude': 'lon'})
                            if 'latitude' in ds.coords:
                                ds = ds.rename({'latitude': 'lat'})
                            ds_crop = ds.sel(lon=self.s_lon_slice, lat=self.s_lat_slice)
                            slice_data = ds_crop.sel(time=ts, method='nearest')
                            frame_val = 10 ** slice_data['d2fd'].values
                    else:
                        with xr.open_dataset(path) as ds:
                            if 'longitude' in ds.coords:
                                ds = ds.rename({'longitude': 'lon'})
                            if 'latitude' in ds.coords:
                                ds = ds.rename({'latitude': 'lat'})
                            ds_crop = ds.sel(lon=self.s_lon_slice, lat=self.s_lat_slice)
                            frame_val = 10 ** ds_crop['d2fd'].values
                except:
                    pass
            if frame_val is None:
                raise RuntimeError(f"Missing spectrum data for {ts}")
            spec_frames.append(frame_val)

        spec_seq = np.stack(spec_frames, axis=0)
        spec_seq = spec_seq.transpose(0, 4, 3, 1, 2)  # [T, Lon, Lat, D, F]
        spec_input = spec_seq[0]
        spec_label = spec_seq[1:]

        # ERA5 forcings
        start_time = self.time_index[global_idx]
        end_time = self.time_index[global_idx + self.lead_time - 1]
        time_var = 'time' if 'time' in self.ds_wave.coords else 'valid_time'
        wave_subset = self.ds_wave['wmb'].sel(
            **{time_var: slice(start_time, end_time)},
            longitude=self.w_lon_slice,
            latitude=self.w_lat_slice,
        )
        bathy_raw = wave_subset.values
        wind_subset_u = self.ds_wind['u10'].sel(
            **{time_var: slice(start_time, end_time)},
            longitude=self.u_lon_slice,
            latitude=self.u_lat_slice,
        )
        wind_subset_v = self.ds_wind['v10'].sel(
            **{time_var: slice(start_time, end_time)},
            longitude=self.u_lon_slice,
            latitude=self.u_lat_slice,
        )
        u_raw = wind_subset_u.values
        v_raw = wind_subset_v.values

        # Downsample wind to 0.5 deg grid and align
        u_down = u_raw[:, ::2, ::2]
        v_down = v_raw[:, ::2, ::2]
        t_dim, lat_dim, lon_dim = bathy_raw.shape
        u_down = u_down[:, :lat_dim, :lon_dim]
        v_down = v_down[:, :lat_dim, :lon_dim]
        bathy_trans = bathy_raw.transpose(0, 2, 1)
        u_trans = u_down.transpose(0, 2, 1)
        v_trans = v_down.transpose(0, 2, 1)

        # Mask invalid and fill
        mask_wmb = np.isnan(bathy_trans)
        mask_wind = np.isnan(u_trans)
        mask_invalid = (mask_wmb | mask_wind)
        bathy_final = bathy_trans.copy()
        u_final = u_trans.copy()
        v_final = v_trans.copy()
        bathy_final[mask_invalid] = 0
        u_final[mask_invalid] = 0
        v_final[mask_invalid] = 0
        mask_out = (~mask_invalid).astype(bool)
        mask_out_static = mask_out[0]

        # Resize to spectrum grid if needed
        if bathy_final.shape[1:] != spec_input.shape[0:2]:
            import scipy.ndimage
            target_shape = spec_input.shape[0:2]
            def resize_t(arr):
                zoom = (1, target_shape[0]/arr.shape[1], target_shape[1]/arr.shape[2])
                return scipy.ndimage.zoom(arr, zoom, order=0)
            bathy_final = resize_t(bathy_final)
            u_final = resize_t(u_final)
            v_final = resize_t(v_final)
            zoom_mask = (target_shape[0]/mask_out_static.shape[0], target_shape[1]/mask_out_static.shape[1])
            mask_out_static = scipy.ndimage.zoom(mask_out_static.astype(float), zoom_mask, order=0).astype(bool)

        # Normalize winds optionally
        if self.rescale_forcings:
            u_final = self._normalize_var(u_final, 'u_wind')
            v_final = self._normalize_var(v_final, 'v_wind')

        forcings = np.stack([bathy_final, u_final, v_final], axis=-1)  # [T, Lon, Lat, 3]

        # Expand dims for inference compatibility
        spec_input = np.expand_dims(spec_input, axis=0)  # [1, Lon, Lat, D, F]
        mask_final = np.expand_dims(mask_out_static, axis=0)
        mask_final = np.expand_dims(mask_final, axis=-1)  # [1, Lon, Lat, 1]

        # Sparse mask placeholder
        if self.sparse_mask_ds:
            s_mask = mask_final.copy()
        else:
            s_mask = mask_final.copy()

        # Time strings for t1..tL
        ts_slice = self.time_index[global_idx + 1 : global_idx + self.lead_time + 1]
        time_strs = [t.strftime('%Y-%m-%d %H:%M:%S') for t in ts_slice]

        return (
            torch.from_numpy(np.nan_to_num(spec_input)).float(),
            torch.from_numpy(forcings[0:self.lead_time]).float(),
            torch.from_numpy(bathy_final).float(),
            torch.from_numpy(np.nan_to_num(spec_label)).float(),
            torch.from_numpy(mask_final).bool(),
            torch.from_numpy(s_mask).bool(),
            time_strs,
        )

    def __len__(self):
        return len(self.indices)
