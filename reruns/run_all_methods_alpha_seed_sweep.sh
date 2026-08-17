#!/usr/bin/env bash
# All five EA/NEA methods on Mandl + Mumford0-3, alpha 0..1, seeds 0..9.
#
#   reruns/run_all_methods_alpha_seed_sweep.sh
#   reruns/run_all_methods_alpha_seed_sweep.sh smoke
#   reruns/run_all_methods_alpha_seed_sweep.sh full --cities Mandl Mumford0
#   reruns/run_all_methods_alpha_seed_sweep.sh full --methods EA NEA-Combined
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

# Leave device choice automatic: CUDA is preferred where available, then MPS,
# then CPU. GPU_ID is optional and only restricts CUDA hosts.
if [[ -n "${GPU_ID:-}" ]]; then
    if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
        echo "GPU_ID must be one physical CUDA device index" >&2
        exit 2
    fi
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

LOG_PATH="artifacts/cli_logs/run_all_methods_alpha_seed_sweep.log"
nohup python scripts/run_all_methods_alpha_seed_sweep.py \
    --profile "$PROFILE" "${SUITE_ARGS[@]}" "${EXTRA_ARGS[@]}" \
    > "$LOG_PATH" 2>&1 < /dev/null &
PID=$!
disown "$PID" 2>/dev/null || true
sleep 3

echo "All-method alpha/seed sweep started (PID $PID), profile=$PROFILE"
echo "grid: 5 graphs x 5 methods x 10 seeds x 11 alpha values = 2750 points"
echo "log:     tail -f $LOG_PATH"
echo "results: artifacts/reruns/all_methods_alpha_seed_sweep/"
echo "runs:    artifacts/runs/all_methods_alpha_seed_sweep/"
echo "TB:      tensorboard --logdir artifacts/runs/all_methods_alpha_seed_sweep"
echo "stop:    kill $PID"
