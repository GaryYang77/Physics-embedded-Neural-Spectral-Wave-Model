#!/bin/bash

# Simplified wrapper to test a model on multiple regions with minimal inputs.
# Required: MODEL_PATH, CONFIG_FILE
# Optional: POSTFIX (default: default)
# Regions covered: SouthChinaSea, EastPacificSpec, SouthIndianOceanSpec
# Results dir auto pattern: infer_results/<model_config>_<postfix>/<region>_<year>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --- User inputs ---
MODEL_PATH="${MODEL_PATH:-saved_models/Joint-Trained/best_model.pth}"
CONFIG_FILE="${CONFIG_FILE:-model_configs/model_unetParam_c5.gin}"
POSTFIX="${POSTFIX:-allRegions}"
INPUT_NORM_PATH="${INPUT_NORM_PATH-}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_PARAMS="${OUTPUT_PARAMS:-swh,mwd,mwp}"
RESULTS_ROOT="${RESULTS_ROOT:-infer_results}"
REGIONS="${REGIONS:-EastPacificSpec,SouthChinaSea,SouthIndianOceanSpec}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# --- Optional settings ---
YEAR=${YEAR:-2023}
LEAD_TIME=${LEAD_TIME:-240}
BATCH_SIZE=${BATCH_SIZE:-1}
COMPILE=${COMPILE:-true}
RESCALE_FORCINGS=${RESCALE_FORCINGS:-false}
SAVE_SAMPLE_NC=${SAVE_SAMPLE_NC:-true}
SAMPLE_INDICES=${SAMPLE_INDICES:-"0:8496:120"}
SAVE_SPECTRA_SAMPLES=${SAVE_SPECTRA_SAMPLES-}
SAVE_STABILITY_MATRIX=${SAVE_STABILITY_MATRIX:-false}
START_DATE=${START_DATE:-"${YEAR}-01-01 00:00:00"}
END_DATE=${END_DATE:-"${YEAR}-12-31 23:00:00"}
NORM_STATS_PATH=${NORM_STATS_PATH:-"./data/data_validate_norm_stats.json"}

if [[ -z "$MODEL_PATH" || -z "$CONFIG_FILE" ]]; then
  echo "Error: please set MODEL_PATH and CONFIG_FILE (env or inline)."
  echo "Example: MODEL_PATH=path/to/model.pth CONFIG_FILE=model_configs/model.gin bash scripts/run_infer_regions.sh"
  exit 1
fi

# --- Region definitions ---
WAVE_NC="${WAVE_NC:-../WaveDataProcess/data/ERA5_${YEAR}_wave_global.nc}"
WIND_NC="${WIND_NC:-../WaveDataProcess/data/ERA5_${YEAR}_wind_global.nc}"

declare -A REGION_DIRS
REGION_DIRS[SouthChinaSea]="../WaveDataProcess/data/SouthChinaSea/wave_spectrum_nc_${YEAR}"
REGION_DIRS[EastPacificSpec]="../WaveDataProcess/data/EastPacificSpec/wave_spectrum_nc_${YEAR}"
REGION_DIRS[SouthIndianOceanSpec]="../WaveDataProcess/data/SouthIndianOceanSpec_clipped/wave_spectrum_nc_${YEAR}"

# Derive model_config name from CONFIG_FILE basename (strip path and extension)
CFG_BASE=$(basename "$CONFIG_FILE")
MODEL_CFG_NAME="${CFG_BASE%.*}"

run_region() {
  local region=$1
  local spectrum_dir=${REGION_DIRS[$region]}
  if [[ -z "$spectrum_dir" ]]; then
    echo "Skip unknown region $region"; return
  fi
  if [[ ! -d "$spectrum_dir" ]]; then
    echo "Warning: spectrum_dir not found for $region: $spectrum_dir"; return
  fi

  local results_dir="${RESULTS_ROOT}/${MODEL_CFG_NAME}_${POSTFIX}/${region}_${YEAR}"
  mkdir -p "$results_dir"

  CMD="${PYTHON_BIN} inference_from_source.py"
  CMD+=" --spectrum-dir \"$spectrum_dir\""
  CMD+=" --wave-nc \"$WAVE_NC\""
  CMD+=" --wind-nc \"$WIND_NC\""
  CMD+=" --start-date \"$START_DATE\""
  CMD+=" --end-date \"$END_DATE\""
  CMD+=" --config-file \"$CONFIG_FILE\""
  CMD+=" --results-dir \"$results_dir\""
  CMD+=" --batch-size $BATCH_SIZE"
  CMD+=" --lead-time $LEAD_TIME"
  CMD+=" --device $DEVICE"
  CMD+=" --norm-stats-path \"$NORM_STATS_PATH\""
  if [[ -n "$OUTPUT_PARAMS" ]]; then CMD+=" --output-params \"$OUTPUT_PARAMS\""; fi
  if [[ -n "$INPUT_NORM_PATH" ]]; then CMD+=" --input-norm-path \"$INPUT_NORM_PATH\""; fi
  CMD+=" --sample-indices \"$SAMPLE_INDICES\""
  if [[ -n "$SAVE_SPECTRA_SAMPLES" ]]; then CMD+=" --save-spectra-samples \"$SAVE_SPECTRA_SAMPLES\""; fi
  if [[ "$SAVE_STABILITY_MATRIX" == true ]]; then CMD+=" --save-stability-matrix"; fi
  if [[ "$COMPILE" == true ]]; then CMD+=" --compile"; fi
  if [[ "$SAVE_SAMPLE_NC" == true ]]; then CMD+=" --save-sample-nc"; fi
  if [[ "$RESCALE_FORCINGS" == true ]]; then CMD+=" --rescale-forcings"; fi
  if [[ -n "$MODEL_PATH" ]]; then CMD+=" --model-path \"$MODEL_PATH\""; fi

  echo "=========================================="
  echo "Region: $region"
  echo "Spectrum Dir: $spectrum_dir"
  echo "Results Dir: $results_dir"
  echo "Model: $MODEL_PATH"
  echo "Config: $CONFIG_FILE"
  echo "Device: $DEVICE"
  echo "Lead Time: $LEAD_TIME"
  echo "Batch Size: $BATCH_SIZE"
  echo "Compile: $COMPILE"
  echo "Save Sample NC: $SAVE_SAMPLE_NC"
  echo "Rescale Forcings: $RESCALE_FORCINGS"
  echo "Input Norm Path: $INPUT_NORM_PATH"
  echo "Output Params: $OUTPUT_PARAMS"
  echo "Save Stability Matrix: $SAVE_STABILITY_MATRIX"
  echo "Command: $CMD"
  echo "=========================================="

  eval $CMD
}

# Run requested regions (comma-separated)
IFS=',' read -r -a REGION_LIST <<< "$REGIONS"
for region in "${REGION_LIST[@]}"; do
  run_region "$region"
done
