"""Directory layout and usage notes for convert_grib_to_nc_daily.py.

This script converts monthly GRIB wave spectra into hourly NetCDF files
grouped by day.

Before running it, make sure your files follow one of the supported source
layouts.

Recommended source layout (SouthChinaSea 2023 example)

        data/SouthChinaSea/
            wave_spectrum_grib_2023/
                wave_spectrum_2023_01/
                    *.grib
                wave_spectrum_2023_02/
                    *.grib
                ...
                wave_spectrum_2023_12/
                    *.grib

The script scans source_root for month folders named wave_spectrum_YYYY_MM.
If no such folders are found, it falls back to reading .grib files directly
under source_root.

Important naming and path assumptions

- --source-root should point to the annual GRIB directory, for example
    data/SouthChinaSea/wave_spectrum_grib_2023.
- Monthly folders must be named wave_spectrum_YYYY_MM.
- GRIB filenames should contain _YYYY_. When --month is used, filenames
    should also contain _YYYY_MM_ so they can be filtered correctly.
- --target-root should point to the annual NetCDF output directory, for
    example data/SouthChinaSea/wave_spectrum_nc_2023.

Expected output layout (SouthChinaSea 2023 example)

        data/SouthChinaSea/
            wave_spectrum_nc_2023/
                20230101/
                    wave_spectrum_2023_01_01_00.nc
                    wave_spectrum_2023_01_01_01.nc
                    ...
                20230102/
                    wave_spectrum_2023_01_02_00.nc
                    ...

Example commands

        python convert_grib_to_nc_daily.py \
            --source-root data/SouthChinaSea/wave_spectrum_grib_2023 \
            --year 2023 \
            --target-root data/SouthChinaSea/wave_spectrum_nc_2023 \
            --overwrite

        python convert_grib_to_nc_daily.py \
            --source-root data/SouthChinaSea/wave_spectrum_grib_2023 \
            --year 2023 \
            --month 1 \
            --target-root data/SouthChinaSea/wave_spectrum_nc_2023

Optional cropping bounds (--lon-min, --lon-max, --lat-min, --lat-max) are
applied to each hourly slice before writing the NetCDF file.
"""

import os
import sys
import argparse
from datetime import datetime
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert monthly GRIB wave spectra to daily/hourly NetCDF files"
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help=(
            "Source root that contains monthly GRIB folders, e.g., .../wave_spectrum_grib_2022, "
            "or a directory with month subfolders named wave_spectrum_YYYY_MM, or a directory that directly contains .grib files."
        ),
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year to process (e.g., 2022)",
    )
    parser.add_argument(
        "--month",
        type=int,
        help="Optional month to process (1-12). If omitted, process all months present.",
    )
    parser.add_argument(
        "--target-root",
        required=True,
        help=(
            "Target root to write daily/hourly NetCDF files. "
            "Use a year-specific folder such as .../wave_spectrum_nc_2022. "
            "Output structure: <target-root>/{yyyymmdd}/wave_spectrum_{YYYY_MM_DD}_{HH}.nc"
        ),
    )
    parser.add_argument(
        "--engine",
        default="cfgrib",
        choices=["cfgrib"],
        help="Engine to read GRIB files (cfgrib)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing NetCDF files if present",
    )
    parser.add_argument(
        "--lon-min",
        type=float,
        help="Minimum longitude bound for cropping (supports -180~180 or 0~360)",
    )
    parser.add_argument(
        "--lon-max",
        type=float,
        help="Maximum longitude bound for cropping (supports -180~180 or 0~360)",
    )
    parser.add_argument(
        "--lat-min",
        type=float,
        help="Minimum latitude bound for cropping",
    )
    parser.add_argument(
        "--lat-max",
        type=float,
        help="Maximum latitude bound for cropping",
    )
    return parser.parse_args()


def find_month_dirs(source_root, year, month=None):
    """Locate month directories under source_root.
    Accepts structures like:
      - <source_root>/wave_spectrum_YYYY_MM
      - or <source_root> directly containing .grib files
    Returns: list of directories to scan for .grib files.
    """
    month_dirs = []
    if not os.path.isdir(source_root):
        return month_dirs

    subdirs = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]
    for name in subdirs:
        if not name.startswith("wave_spectrum_"):
            continue
        parts = name.split("_")
        if len(parts) >= 3:
            try:
                y = int(parts[-2])
                m = int(parts[-1])
            except Exception:
                continue
            if y == year and (month is None or m == month):
                month_dirs.append(os.path.join(source_root, name))

    # If no month subfolders found, but source_root has .grib files, use source_root
    if not month_dirs:
        try:
            grib_files = [f for f in os.listdir(source_root) if f.endswith('.grib')]
            if grib_files:
                month_dirs.append(source_root)
        except Exception:
            pass

    month_dirs.sort()
    return month_dirs


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def normalize_variable_names(ds):
    # Ensure lon/lat coordinate names are consistent
    if "longitude" in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    if "latitude" in ds.coords:
        ds = ds.rename({"latitude": "lat"})

    # Try to ensure main data variable is named 'd2fd' to match loader expectations
    data_var = None
    for v in ds.data_vars:
        vl = v.lower()
        if vl in ("d2fd", "2dfd", "2d_wave_spectra_single_msl"):
            data_var = v
            break
    if data_var is None:
        # Fall back to first data var
        data_var = list(ds.data_vars.keys())[0]

    if data_var != "d2fd":
        ds = ds.rename({data_var: "d2fd"})

    return ds


def _coord_name(ds, preferred, fallback):
    if preferred in ds.coords:
        return preferred
    if fallback in ds.coords:
        return fallback
    # also check dims in case coords aren't set explicitly
    if preferred in ds.dims:
        return preferred
    if fallback in ds.dims:
        return fallback
    return preferred  # default


def _normalize_lon_bounds_for_domain(lon_min, lon_max, ds_lon_vals):
    """Map user lon bounds to dataset domain.
    If dataset uses 0..360 and user provides negative bounds, add 360.
    If dataset uses -180..180 and user provides >180, subtract 360.
    Returns (min, max) in dataset domain.
    """
    if lon_min is None or lon_max is None:
        return lon_min, lon_max
    ds_min = float(np.min(ds_lon_vals))
    ds_max = float(np.max(ds_lon_vals))
    domain_0360 = ds_min >= 0 and ds_max > 180

    lm, lx = float(lon_min), float(lon_max)
    if domain_0360:
        if lm < 0:
            lm += 360
        if lx < 0:
            lx += 360
    else:  # assume -180..180
        if lm > 180:
            lm -= 360
        if lx > 180:
            lx -= 360
    return lm, lx


def crop_hour_dataset(ds_hour, lon_min=None, lon_max=None, lat_min=None, lat_max=None):
    """Crop a single-hour dataset on lon/lat. Handles ascending/descending lat,
    and 0..360 vs -180..180 lon domains. Supports cross-dateline by concatenation.
    """
    if all(v is None for v in (lon_min, lon_max, lat_min, lat_max)):
        return ds_hour

    lon_name = _coord_name(ds_hour, "lon", "longitude")
    lat_name = _coord_name(ds_hour, "lat", "latitude")

    out = ds_hour

    # Latitude cropping
    if lat_min is not None and lat_max is not None and lat_name in out.coords:
        lat_vals = out[lat_name].values
        asc = float(lat_vals[0]) < float(lat_vals[-1])
        if asc:
            out = out.sel({lat_name: slice(float(lat_min), float(lat_max))})
        else:
            out = out.sel({lat_name: slice(float(lat_max), float(lat_min))})

    # Longitude cropping
    if lon_min is not None and lon_max is not None and lon_name in out.coords:
        lon_vals = out[lon_name].values
        lon_vals = lon_vals.astype(float)
        lm, lx = _normalize_lon_bounds_for_domain(lon_min, lon_max, lon_vals)
        asc = float(lon_vals[0]) < float(lon_vals[-1])

        # Normal case: bounds don't cross dateline
        if lm <= lx:
            if asc:
                out = out.sel({lon_name: slice(lm, lx)})
            else:
                out = out.sel({lon_name: slice(lx, lm)})
        else:
            # Cross-dateline: union of [lm, max] U [min, lx]
            ds_min = float(np.min(lon_vals))
            ds_max = float(np.max(lon_vals))
            if asc:
                part1 = out.sel({lon_name: slice(lm, ds_max)})
                part2 = out.sel({lon_name: slice(ds_min, lx)})
            else:
                part1 = out.sel({lon_name: slice(ds_min, lm)})
                part2 = out.sel({lon_name: slice(lx, ds_max)})
            try:
                out = xr.concat([part1, part2], dim=lon_name)
            except Exception:
                # Fallback: prefer the larger part if concat fails
                out = part1 if part1.sizes.get(lon_name, 0) >= part2.sizes.get(lon_name, 0) else part2

    # Post-crop diagnostic (tqdm-friendly)
    try:
        lon_span = (float(out[lon_name].min()), float(out[lon_name].max())) if lon_name in out.coords else None
        lat_span = (float(out[lat_name].min()), float(out[lat_name].max())) if lat_name in out.coords else None
        # tqdm.write(f"After crop -> lon: {lon_span}, lat: {lat_span}")
    except Exception:
        pass

    return out


def write_hour_nc(ds_hour, out_path, overwrite=False):
    if (not overwrite) and os.path.exists(out_path):
        return
    # Use simple lossless compression
    encoding = {}
    for v in ds_hour.data_vars:
        encoding[v] = {"zlib": True, "complevel": 4}
    for c in ds_hour.coords:
        if c in ("time", "reftime"):
            encoding[c] = {"zlib": False}
    ds_hour.to_netcdf(out_path, format="NETCDF4", encoding=encoding)


def convert_month_file(grib_file, target_daily_root, overwrite=False,
                       lon_min=None, lon_max=None, lat_min=None, lat_max=None):
    # Open once; cfgrib creates an index file alongside for speed
    ds = xr.open_dataset(
        grib_file,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"stepType": "instant"}},
    )
    ds = normalize_variable_names(ds)

    # Expect a time coordinate with hourly steps
    if "time" not in ds.coords:
        raise ValueError(f"No 'time' coord found in {grib_file}")

    # Convert CF time to pandas Timestamps
    ds_decoded = xr.decode_cf(ds)
    # Robustly get a pandas DatetimeIndex
    idx = ds_decoded.indexes.get("time", None)
    if idx is None:
        # Fall back to coordinate values
        times = pd.to_datetime(ds_decoded["time"].values)
    else:
        # If it's already a DatetimeIndex, use it directly; otherwise convert
        if isinstance(idx, pd.DatetimeIndex):
            times = idx
        else:
            times = pd.to_datetime(idx)

    # Determine date (yyyymmdd) for foldering using each timestamp's date
    for i in tqdm(range(times.size), desc=f"  Hours in {os.path.basename(grib_file)}", unit="hour"):
        ts = times[i]
        day_str = ts.strftime("%Y%m%d")
        hour_str = ts.strftime("%H")
        day_dir = os.path.join(target_daily_root, day_str)
        ensure_dir(day_dir)

        # Select single hour slice, preserve all coords/dims
        # Use the original (non-decoded) ds to avoid unintended dtype conversions
        ds_hour = ds.isel(time=i)

        # Apply lon/lat cropping if requested
        ds_hour = crop_hour_dataset(ds_hour, lon_min=lon_min, lon_max=lon_max,
                        lat_min=lat_min, lat_max=lat_max)

        # Standardize a minimal set of attributes to keep compatibility
        ds_hour = ds_hour.assign_attrs({
            "Conventions": ds.attrs.get("Conventions", "CF-1.6"),
            "source": "Converted from GRIB via xarray/cfgrib",
        })

        out_name = f"wave_spectrum_{ts.strftime('%Y_%m_%d')}_{hour_str}.nc"
        out_path = os.path.join(day_dir, out_name)
        write_hour_nc(ds_hour, out_path, overwrite=overwrite)


def run_conversion(source_root, year, target_root, month=None, overwrite=False,
                   lon_min=None, lon_max=None, lat_min=None, lat_max=None):
    # target_root is expected to be the year folder, e.g., .../wave_spectrum_nc_2022
    target_year_root = target_root
    ensure_dir(target_year_root)

    month_dirs = find_month_dirs(source_root, year, month)
    if not month_dirs:
        print("No monthly GRIB directories found for given year/month.")
        sys.exit(1)

    for mdir in month_dirs:
        print(f"Processing month directory: {mdir}")
        files = [f for f in os.listdir(mdir) if f.endswith('.grib')]
        # If scanning a mixed directory, filter by year/month
        if month is not None:
            files = [f for f in files if f"_{year}_{month:02d}_" in f]
        else:
            files = [f for f in files if f"_{year}_" in f]
        files.sort()
        for fname in tqdm(files, desc="Monthly GRIB files", unit="file"):
            grib_path = os.path.join(mdir, fname)
            convert_month_file(
                grib_path,
                target_year_root,
                overwrite=overwrite,
                lon_min=lon_min,
                lon_max=lon_max,
                lat_min=lat_min,
                lat_max=lat_max,
            )


if __name__ == "__main__":
    if len(sys.argv) > 1 and any(arg.startswith("-") for arg in sys.argv[1:]):
        args = parse_args()
        run_conversion(
            source_root=args.source_root,
            year=args.year,
            target_root=args.target_root,
            month=args.month,
            overwrite=args.overwrite,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
        )
    else:
        SOURCE_ROOT = "../WaveDataProcess/data/SouthChinaSea/wave_spectrum_grib_2023"  # Directory containing monthly wave_spectrum_YYYY_MM directories, or directly containing .grib files
        YEAR = 2023                                      # Year
        MONTH = None                                     # Optional: month (1-12); set to None to process the entire year
        TARGET_ROOT = "../WaveDataProcess/data/SouthChinaSea/wave_spectrum_nc_2023"   # Directly specify the annual NC output directory
        OVERWRITE = True                                 # Whether to overwrite existing nc files

        # Optional cropping range (default is no cropping)
        LON_MIN = None
        LON_MAX = None
        LAT_MIN = None
        LAT_MAX = None

        run_conversion(
            source_root=SOURCE_ROOT,
            year=YEAR,
            target_root=TARGET_ROOT,
            month=MONTH,
            overwrite=OVERWRITE,
            lon_min=LON_MIN,
            lon_max=LON_MAX,
            lat_min=LAT_MIN,
            lat_max=LAT_MAX,
        )
