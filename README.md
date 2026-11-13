<h1 align="center">Physics-embedded Neural Spectral Wave Model</h1>

<p align="center">
  Guangyu Gary Yang<sup>1,†</sup> &nbsp;·&nbsp;
  Haoyu Jiang<sup>2,†</sup> &nbsp;·&nbsp;
  Wenfang Lu<sup>1,*</sup> &nbsp;·&nbsp;
  Dong Wang<sup>3</sup>
</p>

<p align="center">
  <sup>1</sup> Sun Yat-Sen University, China<br>
  <sup>2</sup> Shenzhen University, China<br>
  <sup>3</sup> Xiamen University, China
</p>

<p align="center">
  <sup>†</sup> Contributed equally. <sup>*</sup> Correspondence: Wenfang Lu
  (<a href="mailto:luwf6@sysu.edu.cn">luwf6@sysu.edu.cn</a>)
</p>

<p align="center">
  <a href="https://doi.org/DOI">Paper</a> &nbsp;·&nbsp;
  <a href="https://doi.org/10.5281/zenodo.19920055">Code archive</a> &nbsp;·&nbsp;
  <a href="#released-checkpoints">Checkpoints</a> &nbsp;·&nbsp;
  <a href="#quick-start">Quick Start</a> &nbsp;·&nbsp;
  <a href="https://github.com/GaryYang77/NeuralSWM">JAX implementation <i>(Active!)</i></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C" alt="PyTorch 2.x">
  <img src="https://img.shields.io/badge/Release-Public%20Inference-0A7E8C" alt="Public inference release">
</p>

<p align="center">
  <img src="Figures/Fig.1%20NSWM%20Model%20Framework.jpg" width="1080" alt="NSWM overview">
</p>

---

Official inference package for the paper **"Physics-embedded Neural Spectral Wave Model"**. NSWM couples a differentiable numerical propagation core with a neural source-term parameterization. This public release is focused on practical inference with released checkpoints: give it ERA5 wave and wind forcings together with local spectrum NetCDF inputs, and it will run well-trained checkpoints to produce hourly wave forecasts. Bulk parameters are exported by default, while sample-level spectra and stability diagnostics can be saved when needed.

> **Note on this release:** This is the official code release for the paper. We stripped out some training and dev scripts to keep it lightweight. It's focused purely on running inference with the published checkpoints. For the bleeding-edge (and JAX-optimized) version, check out the link at the bottom.

## What This Release Includes

- Hybrid physics-AI wave modeling with physically consistent propagation and learned source-term inference
- Released checkpoints for both the SCS-Trained model and the multi-region Joint-Trained model
- One-command rollout across South China Sea, East Tropical Pacific, and South Indian Ocean
- NetCDF export for `swh`, `mwd`, and `mwp` by default, with optional Stokes drift and sample spectra outputs

## Released Checkpoints

Both released checkpoints use `model_configs/model_unetParam_c5.gin`.

| Path | Description |
|---|---|
| `saved_models/SCS_Trained/best_model.pth` | Trained on SCS 2022 only; useful for in-domain SCS runs and zero-shot transfer experiments |
| `saved_models/Joint-Trained/best_model.pth` | Jointly trained on SCS, ETP, and SIO; default choice for public inference |

If `MODEL_PATH` is not set, `scripts/run_infer_regions.sh` uses the joint checkpoint by default.

## Environment Setup

The examples below assume a Bash-compatible shell. Python 3.8+ and PyTorch 2.x are required.

```bash
pip install -r requirements.txt
```

## Input Data

By default, `scripts/run_infer_regions.sh` resolves the following paths relative to the repository root:

```text
../WaveDataProcess/data/ERA5_2023_wave_global.nc
../WaveDataProcess/data/ERA5_2023_wind_global.nc
../WaveDataProcess/data/SouthChinaSea/wave_spectrum_nc_2023/
../WaveDataProcess/data/EastPacificSpec/wave_spectrum_nc_2023/
../WaveDataProcess/data/SouthIndianOceanSpec_clipped/wave_spectrum_nc_2023/
```

Internal region IDs used by `REGIONS` are:

| Region | `REGIONS` value |
|---|---|
| South China Sea | `SouthChinaSea` |
| East Tropical Pacific | `EastPacificSpec` |
| South Indian Ocean | `SouthIndianOceanSpec` |

You can override the default inputs with environment variables such as `WAVE_NC`, `WIND_NC`, `REGIONS`, `YEAR`, `START_DATE`, and `END_DATE`.

A minimal runnable demo dataset is also included under `examples/data/`, containing local wave spectra and ERA5 wave/wind forcings for a short Fujian-region example.

### Preparing spectral NetCDF inputs from GRIB

The local spectrum inputs consumed by inference are daily/hourly NetCDF files under `wave_spectrum_nc_<YEAR>`. If your source data are ECMWF wave-spectrum GRIB files, you can convert them with `convert_grib_to_nc_daily.py`.

The converter supports either monthly subfolders named `wave_spectrum_YYYY_MM` or `.grib` files placed directly under the annual source directory. It writes outputs grouped by day:

```text
wave_spectrum_nc_2023/
  20230101/
    wave_spectrum_2023_01_01_00.nc
    wave_spectrum_2023_01_01_01.nc
  20230102/
    ...
```

Example:

```bash
python convert_grib_to_nc_daily.py \
  --source-root ../WaveDataProcess/data/SouthChinaSea/wave_spectrum_grib_2023 \
  --year 2023 \
  --target-root ../WaveDataProcess/data/SouthChinaSea/wave_spectrum_nc_2023 \
  --overwrite
```

Optional `--lon-min`, `--lon-max`, `--lat-min`, and `--lat-max` arguments can be used to crop each hourly slice before writing NetCDF.

## Quick Start

The default public inference setup runs a 240-hour rollout.

### Demo with bundled example data

The fastest way to verify the released checkpoint is to run the bundled example in `examples/data/`. This demo uses the existing inference entrypoint together with:

- `examples/data/wave_spectrum_nc_2022_demo/`
- `examples/data/environ_forcings/ERA5_2022_wave_fujian_demo.nc`
- `examples/data/environ_forcings/ERA5_2022_wind_fujian_demo.nc`

It launches a single 24-hour rollout from the released `.pth` checkpoint and writes outputs to `examples/results/fujian_demo/`.

```bash
bash examples/run_examples_demo.sh
```

If you need a specific Python interpreter or device, override them inline:

```bash
PYTHON_BIN="PATH/TO/envs/Torch/python.exe" \
DEVICE="cuda:0" \
bash examples/run_examples_demo.sh
```

### Recommended: joint inference across all released regions

```bash
bash scripts/run_infer_regions.sh
```

### SCS only with the joint checkpoint

```bash
REGIONS=SouthChinaSea \
MODEL_PATH="saved_models/Joint-Trained/best_model.pth" \
bash scripts/run_infer_regions.sh
```

### SCS only with the SCS-trained checkpoint

```bash
REGIONS=SouthChinaSea \
MODEL_PATH="saved_models/SCS_Trained/best_model.pth" \
bash scripts/run_infer_regions.sh
```

## Performance

Correlation coefficient (CC) for significant wave height (SWH) against ERA5 over the 2023 test period.

| Region | SCS-Trained CC | Joint-Trained CC |
|---|---|---|
| South China Sea (SCS) | **0.991** *(in-domain)* | 0.982 |
| East Tropical Pacific (ETP) | 0.889 *(zero-shot)* | **0.979** |
| South Indian Ocean (SIO) | 0.963 *(zero-shot)* | **0.991** |

Independent satellite altimeter validation for the Joint-Trained model: CC = 0.960 (SCS), 0.942 (ETP), 0.968 (SIO).

### Directional wave spectrum skill across regions

<p align="center">
  <img src="Figures/Fig.6.jpg" width="1080" alt="Joint-trained directional wave spectrum performance">
</p>

Joint-trained NSWM performance for directional wave spectra across SCS, ETP, and SIO. The figure summarizes spectral correlation by frequency, mean predicted versus ERA5 directional spectra, and 2D spectral correlation in frequency-direction space.

### Spatiotemporal examples

<p align="center">
  <img src="Figures/SIO_Sample_2.jpg" width="1080" alt="South Indian Ocean spatiotemporal example">
</p>

South Indian Ocean example comparing Joint-Trained NSWM against ERA5 for SWH, MWD, and MWP.

<p align="center">
  <img src="Figures/SCS_Sample_0.jpg" width="1080" alt="South China Sea spatiotemporal example">
</p>

South China Sea example comparing Joint-Trained NSWM against ERA5 for SWH, MWD, and MWP.

<p align="center">
  <img src="Figures/ETP_Sample_2.jpg" width="1080" alt="East Tropical Pacific spatiotemporal example">
</p>

East Tropical Pacific example comparing Joint-Trained NSWM against ERA5 for SWH, MWD, and MWP.

## Output

Each region writes to:

```text
${RESULTS_ROOT}/${MODEL_CFG_NAME}_${POSTFIX}/${REGION}_${YEAR}/
```

With the current script defaults, that becomes:

```text
infer_results/model_unetParam_c5_allRegions/<region>_2023/
```

The main artifacts are:

- `integral_parameters/wave_parameters.nc` for bulk parameter export
- `sample_spectra/*.nc` when `SAVE_SPECTRA_SAMPLES` is set
- `metrics/stability_matrix.nc` when `SAVE_STABILITY_MATRIX=true`

The bulk-parameter NetCDF contains by default:

| Variable | Description |
|---|---|
| `swh_pred` / `swh_true` | Significant wave height (m) |
| `mwd_pred` / `mwd_true` | Mean wave direction (°) |
| `mwp_pred` / `mwp_true` | Mean wave period (s) |
| `mask` | Ocean/land mask |
| `actual_time` | Timestamps |

If `OUTPUT_PARAMS` includes `stokes`, the export will additionally include `stokes_u` and `stokes_v`.

## Runtime Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `saved_models/Joint-Trained/best_model.pth` | Checkpoint to load |
| `CONFIG_FILE` | `model_configs/model_unetParam_c5.gin` | Gin config file |
| `RESULTS_ROOT` | `infer_results` | Top-level output directory |
| `POSTFIX` | `allRegions` | Suffix used in the output folder name |
| `REGIONS` | `EastPacificSpec,SouthChinaSea,SouthIndianOceanSpec` | Comma-separated region list |
| `YEAR` | `2023` | Year used to build default input paths |
| `WAVE_NC` | derived from `YEAR` | ERA5 integrated wave-parameter NetCDF |
| `WIND_NC` | derived from `YEAR` | ERA5 wind NetCDF |
| `START_DATE` | `${YEAR}-01-01 00:00:00` | Inference start time |
| `END_DATE` | `${YEAR}-12-31 23:00:00` | Inference end time |
| `LEAD_TIME` | `240` | Rollout length in hours |
| `BATCH_SIZE` | `1` | Inference batch size |
| `SAMPLE_INDICES` | `0:8496:120` | Sample selection as a slice or explicit indices |
| `DEVICE` | `cuda:0` | Target device (`cpu` also supported) |
| `COMPILE` | `true` | Enable `torch.compile` during inference |
| `PYTHON_BIN` | `python` | Python executable used by the wrapper script |
| `OUTPUT_PARAMS` | `swh,mwd,mwp` | Bulk variables to export |
| `SAVE_SAMPLE_NC` | `true` | Write `wave_parameters.nc` |
| `SAVE_SPECTRA_SAMPLES` | unset | Save selected full-spectrum samples |
| `SAVE_STABILITY_MATRIX` | `false` | Save per-sample stability diagnostics |
| `RESCALE_FORCINGS` | `false` | Toggle forcing rescaling behavior |
| `NORM_STATS_PATH` | `./data/data_validate_norm_stats.json` | Normalization-statistics JSON |
| `INPUT_NORM_PATH` | unset | Optional alternate input normalization JSON |

## Notes

- The released checkpoints were trained on a `torch.compile`-enabled path, but public inference normalizes checkpoint keys automatically, so `COMPILE=false` works without manual key remapping.
- `scripts/run_infer_regions.sh` changes into the repository root automatically and can be launched from any working directory.
- Full directional spectra are not exported by default; use `SAVE_SPECTRA_SAMPLES` when you want sample-level spectral NetCDF outputs.

## Citation

If you use NSWM in your research, please cite:

```bibtex
@article{yang2026sciadv,
title = {Physics-embedded Neural Spectral Wave Model},
author = {Yang and  G.† and  Jiang and  H.† and  Lu and  W. and  & Wang and  D.},
journal = {Science Advances},
year = {2026},
note = {in press},
}
```

### Continued development

[NeuralSWM](https://github.com/GaryYang77/NeuralSWM) is the maintained
JAX-based continuation of this research framework. It provides computational
optimizations, training and inference workflows, and a pretrained checkpoint
supporting year-long global wave simulations.

For the published experiments and their associated checkpoints, use this
repository or the [archived paper implementation](https://doi.org/10.5281/zenodo.19920055).
For further development with the JAX implementation, visit NeuralSWM.

## Contact

For questions, please contact luwf6@sysu.edu.cn, yanggy25@mail2.sysu.edu.cn.
