#!/usr/bin/env bash
# Table 4 / Figure 4 (adj-target sweep, Mumford0) rerun -> artifacts/reruns/. Detached (survives SSH disconnect).
#   reruns/run_table4.sh            # full paper budget
#   reruns/run_table4.sh smoke      # fast plumbing check (1 iter, TEMP_ prefix)
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p artifacts/cli_logs
PROFILE="${1:-full}"
if [ "$PROFILE" = "smoke" ]; then SUITE_ARGS="--suite-smoke suite_rerun_smoke"; else SUITE_ARGS="--suite suite_rerun"; fi
setsid nohup python scripts/run_table4.py --profile "$PROFILE" $SUITE_ARGS     > artifacts/cli_logs/rerun_table4.log 2>&1 < /dev/null &
PID=$!
sleep 3
echo "Table 4 / Figure 4 (adj-target sweep, Mumford0) rerun started (PID $PID), profile=$PROFILE"
echo "log:  tail -f artifacts/cli_logs/rerun_table4.log"
echo "out:  artifacts/reruns/"
echo "TB:   tensorboard --logdir artifacts/runs/table4_fig4_our_pareto"
echo "stop: kill $PID"
