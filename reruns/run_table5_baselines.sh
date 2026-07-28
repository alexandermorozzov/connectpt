#!/usr/bin/env bash
# Table-5-style SA/GA/HH baselines with LC init on benchmark graphs.
#   reruns/run_table5_baselines.sh
#   reruns/run_table5_baselines.sh smoke
#   reruns/run_table5_baselines.sh full --cities Mumford2 Mumford3
#   reruns/run_table5_baselines.sh full --budget-mode scaled
# Default full budget mode is eval20k: comparable to Table 5 BCO's 200*10*5*2
# candidate evaluations. Other modes: paper40k, scaled.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p artifacts/cli_logs
PROFILE="${1:-full}"
EXTRA_ARGS=("${@:2}")
if [ "$PROFILE" = "smoke" ]; then SUITE_ARGS="--suite-smoke suite_rerun_smoke"; else SUITE_ARGS="--suite suite_rerun"; fi
setsid nohup python scripts/run_table5_baselines.py --profile "$PROFILE" $SUITE_ARGS "${EXTRA_ARGS[@]}"     > artifacts/cli_logs/rerun_table5_baselines.log 2>&1 < /dev/null &
PID=$!
sleep 3
echo "Table 5 baselines (SA/GA/HH, LC init) started (PID $PID), profile=$PROFILE"
echo "log:  tail -f artifacts/cli_logs/rerun_table5_baselines.log"
echo "run:  tail -f artifacts/reruns/table5_baselines_lcinit_run.log"
echo "out:  artifacts/reruns/"
echo "stop: kill $PID"
