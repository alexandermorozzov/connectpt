#!/usr/bin/env bash
# EKB case study (Improved NBCO, alpha sweep) rerun -> artifacts/reruns/. Detached (survives SSH disconnect).
#   reruns/run_ekb.sh            # full paper budget
#   reruns/run_ekb.sh smoke      # fast plumbing check (1 iter, TEMP_ prefix)
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p artifacts/cli_logs
PROFILE="${1:-full}"
if [ "$PROFILE" = "smoke" ]; then SUITE_ARGS="--suite-smoke suite_rerun_smoke"; else SUITE_ARGS="--suite suite_rerun"; fi
setsid nohup python scripts/run_ekb_case_study.py --profile "$PROFILE" $SUITE_ARGS     > artifacts/cli_logs/rerun_ekb.log 2>&1 < /dev/null &
PID=$!
sleep 3
echo "EKB case study (Improved NBCO, alpha sweep) rerun started (PID $PID), profile=$PROFILE"
echo "log:  tail -f artifacts/cli_logs/rerun_ekb.log"
echo "out:  artifacts/reruns/"
echo "TB:   tensorboard --logdir artifacts/runs/ekb_case_study/tb"
echo "stop: kill $PID"
