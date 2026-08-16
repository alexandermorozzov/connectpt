#!/usr/bin/env bash
# EKB case study rerun -> artifacts/reruns/. Detached (survives SSH disconnect).
#   reruns/run_ekb.sh nea-edit full
#   reruns/run_ekb.sh nea-combined smoke
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p artifacts/cli_logs

GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "GPU_ID must be one physical CUDA device index (for example 0 or 1)" >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU_ID"

METHOD="${1:-nea-combined}"
PROFILE="${2:-full}"
if [ "$METHOD" != "nea-edit" ] && [ "$METHOD" != "nea-combined" ]; then
    echo "method must be nea-edit or nea-combined" >&2
    exit 2
fi
if [ "$PROFILE" != "smoke" ] && [ "$PROFILE" != "full" ]; then
    echo "profile must be smoke or full" >&2
    exit 2
fi
if [ "$PROFILE" = "smoke" ]; then SUITE_ARGS="--suite-smoke suite_rerun_smoke"; else SUITE_ARGS="--suite suite_rerun"; fi
LOG_PATH="artifacts/cli_logs/rerun_ekb_${METHOD}_gpu${GPU_ID}.log"
setsid nohup python scripts/run_ekb_case_study.py --method "$METHOD" \
    --profile "$PROFILE" $SUITE_ARGS > "$LOG_PATH" 2>&1 < /dev/null &
PID=$!
sleep 3
echo "EKB case study rerun started (PID $PID), method=$METHOD, profile=$PROFILE, physical GPU=$GPU_ID"
echo "log:  tail -f $LOG_PATH"
echo "out:  artifacts/reruns/"
echo "TB:   tensorboard --logdir artifacts/runs/ekb_case_study/${METHOD//-/_}/tb"
echo "stop: kill $PID"
