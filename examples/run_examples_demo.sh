#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-saved_models/Joint-Trained/best_model.pth}"
CONFIG_FILE="${CONFIG_FILE:-model_configs/model_unetParam_c5.gin}"

SPECTRUM_DIR="${SPECTRUM_DIR:-examples/data/wave_spectrum_nc_2022_demo}"
WAVE_NC="${WAVE_NC:-examples/data/environ_forcings/ERA5_2022_wave_fujian_demo.nc}"
WIND_NC="${WIND_NC:-examples/data/environ_forcings/ERA5_2022_wind_fujian_demo.nc}"
RESULTS_DIR="${RESULTS_DIR:-examples/results/fujian_demo}"

START_DATE="${START_DATE:-2022-01-01 00:00:00}"
END_DATE="${END_DATE:-2022-01-03 23:00:00}"
LEAD_TIME="${LEAD_TIME:-24}"
SAMPLE_INDICES="${SAMPLE_INDICES:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_PARAMS="${OUTPUT_PARAMS:-swh,mwd,mwp}"

COMPILE="${COMPILE:-false}"
SAVE_SAMPLE_NC="${SAVE_SAMPLE_NC:-true}"
SAVE_SPECTRA_SAMPLES="${SAVE_SPECTRA_SAMPLES-}"
SAVE_STABILITY_MATRIX="${SAVE_STABILITY_MATRIX:-false}"
RESCALE_FORCINGS="${RESCALE_FORCINGS:-false}"
NORM_STATS_PATH="${NORM_STATS_PATH-}"
INPUT_NORM_PATH="${INPUT_NORM_PATH-}"
DRY_RUN="${DRY_RUN:-false}"

CMD=(
  "$PYTHON_BIN" inference_from_source.py
  --spectrum-dir "$SPECTRUM_DIR"
  --wave-nc "$WAVE_NC"
  --wind-nc "$WIND_NC"
  --start-date "$START_DATE"
  --end-date "$END_DATE"
  --model-path "$MODEL_PATH"
  --config-file "$CONFIG_FILE"
  --results-dir "$RESULTS_DIR"
  --batch-size "$BATCH_SIZE"
  --lead-time "$LEAD_TIME"
  --device "$DEVICE"
  --sample-indices "$SAMPLE_INDICES"
  --output-params "$OUTPUT_PARAMS"
)

if [[ "$COMPILE" == true ]]; then
  CMD+=(--compile)
fi
if [[ "$SAVE_SAMPLE_NC" == true ]]; then
  CMD+=(--save-sample-nc)
fi
if [[ "$SAVE_STABILITY_MATRIX" == true ]]; then
  CMD+=(--save-stability-matrix)
fi
if [[ "$RESCALE_FORCINGS" == true ]]; then
  CMD+=(--rescale-forcings)
fi
if [[ -n "$SAVE_SPECTRA_SAMPLES" ]]; then
  CMD+=(--save-spectra-samples "$SAVE_SPECTRA_SAMPLES")
fi
if [[ -n "$NORM_STATS_PATH" ]]; then
  CMD+=(--norm-stats-path "$NORM_STATS_PATH")
fi
if [[ -n "$INPUT_NORM_PATH" ]]; then
  CMD+=(--input-norm-path "$INPUT_NORM_PATH")
fi

echo "=========================================="
echo "Example data demo"
echo "Spectrum Dir: $SPECTRUM_DIR"
echo "Wave NC: $WAVE_NC"
echo "Wind NC: $WIND_NC"
echo "Results Dir: $RESULTS_DIR"
echo "Model: $MODEL_PATH"
echo "Config: $CONFIG_FILE"
echo "Time Range: $START_DATE -> $END_DATE"
echo "Lead Time: $LEAD_TIME"
echo "Sample Indices: $SAMPLE_INDICES"
echo "Batch Size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo "Compile: $COMPILE"
echo "=========================================="

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" == true ]]; then
  exit 0
fi

"${CMD[@]}"