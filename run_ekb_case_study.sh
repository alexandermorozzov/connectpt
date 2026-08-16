#!/usr/bin/env bash
# Run the EKB case study detached so it survives SSH disconnects (long run).
#   ./run_ekb_case_study.sh nea-edit
#   ./run_ekb_case_study.sh nea-combined
#   ./run_ekb_case_study.sh nea-edit --profile smoke
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p artifacts/cli_logs

METHOD="nea-combined"
if [ "${1:-}" = "nea-edit" ] || [ "${1:-}" = "nea-combined" ]; then
    METHOD="$1"
    shift
fi

# The shell wrapper defaults to a full run. A later explicit `--profile smoke`
# overrides this argparse value when a short check is requested.
EXTRA_ARGS=(--profile full "$@")

LOG_PATH="artifacts/cli_logs/ekb_${METHOD}.log"
setsid nohup python scripts/run_ekb_case_study.py \
    --method "$METHOD" "${EXTRA_ARGS[@]}" \
    > "$LOG_PATH" 2>&1 < /dev/null &
PID=$!
sleep 3
echo "EKB case study started (PID $PID), method=$METHOD, args: ${EXTRA_ARGS[*]}"
echo "log:  tail -f $LOG_PATH"
echo "TB:   tensorboard --logdir artifacts/runs/ekb_case_study/${METHOD//-/_}/tb"
echo "stop: kill $PID"
