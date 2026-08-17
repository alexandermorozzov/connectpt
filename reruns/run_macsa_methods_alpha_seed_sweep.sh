#!/usr/bin/env bash
# Four MACSA/Mandl scenarios x seven methods x ten seeds x eleven alpha values.
#
#   reruns/run_macsa_methods_alpha_seed_sweep.sh smoke
#   reruns/run_macsa_methods_alpha_seed_sweep.sh full
#   reruns/run_macsa_methods_alpha_seed_sweep.sh smoke --scenarios mandl_8 --seeds 0
#   GPU_IDS="0 1" reruns/run_macsa_methods_alpha_seed_sweep.sh full
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p artifacts/cli_logs

PROFILE="${1:-full}"
EXTRA_ARGS=("${@:2}")
case "$PROFILE" in
    smoke)
        SUITE_ARGS=(--suite-smoke suite_rerun_smoke)
        ;;
    full)
        SUITE_ARGS=(--suite suite_rerun)
        ;;
    *)
        echo "profile must be 'smoke' or 'full'" >&2
        exit 2
        ;;
esac

GPU_ARGS=()
GPU_LABEL="auto"
REQUESTED_GPU_IDS="${GPU_IDS:-${GPU_ID:-}}"
if [[ -n "$REQUESTED_GPU_IDS" ]]; then
    GPU_IDS_NORMALIZED="${REQUESTED_GPU_IDS//,/ }"
    read -r -a GPU_LIST <<< "$GPU_IDS_NORMALIZED"
    for gpu_id in "${GPU_LIST[@]}"; do
        if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
            echo "GPU_IDS must contain physical CUDA indices (for example '0 1')" >&2
            exit 2
        fi
    done
    export CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}"
    GPU_ARGS=(--gpus "${GPU_LIST[@]}")
    GPU_LABEL="$(IFS=_; echo "${GPU_LIST[*]}")"
fi
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

LOG_PATH="artifacts/cli_logs/run_macsa_methods_alpha_seed_sweep_gpus${GPU_LABEL}.log"
COMMAND=(
    python scripts/run_macsa_methods_alpha_seed_sweep.py
    --profile "$PROFILE" "${SUITE_ARGS[@]}"
)
if (( ${#GPU_ARGS[@]} )); then
    COMMAND+=("${GPU_ARGS[@]}")
fi
if (( ${#EXTRA_ARGS[@]} )); then
    COMMAND+=("${EXTRA_ARGS[@]}")
fi
nohup "${COMMAND[@]}" > "$LOG_PATH" 2>&1 < /dev/null &
PID=$!
disown "$PID" 2>/dev/null || true
sleep 3

echo "MACSA seven-method sweep started (PID $PID), profile=$PROFILE, GPUs=$GPU_LABEL"
echo "default grid: 4 scenarios x 7 methods x 10 seeds x 11 alpha = 3080 points"
echo "log:     tail -f $LOG_PATH"
echo "results: artifacts/reruns/macsa_methods_alpha_seed_sweep/"
echo "runs:    artifacts/runs/macsa_methods_alpha_seed_sweep/"
echo "stop:    kill $PID"
