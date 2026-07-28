#!/usr/bin/env bash
# Table 5 / Figure 5 (5 operator combos, Mumford1 by default) rerun -> artifacts/reruns/. Detached (survives SSH disconnect).
#   reruns/run_table5.sh            # full paper budget
#   reruns/run_table5.sh smoke      # fast plumbing check (1 iter, TEMP_ prefix)
#   reruns/run_table5.sh full --cities Mandl Mumford0 Mumford1 Mumford2 Mumford3
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p artifacts/cli_logs
PROFILE="${1:-full}"
EXTRA_ARGS=("${@:2}")
if [ "$PROFILE" = "smoke" ]; then SUITE_ARGS="--suite-smoke suite_rerun_smoke"; else SUITE_ARGS="--suite suite_rerun"; fi
setsid nohup python scripts/run_figure5_pareto.py --profile "$PROFILE" $SUITE_ARGS "${EXTRA_ARGS[@]}"     > artifacts/cli_logs/rerun_table5.log 2>&1 < /dev/null &
PID=$!
sleep 3
echo "Table 5 / Figure 5 (5 operator combos) rerun started (PID $PID), profile=$PROFILE"
echo "log:  tail -f artifacts/cli_logs/rerun_table5.log"
echo "out:  artifacts/reruns/"
echo "TB:   tensorboard --logdir artifacts/runs/table5_fig5_5model"
echo "stop: kill $PID"
