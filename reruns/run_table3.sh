#!/usr/bin/env bash
# Table 3 (NeuralBCO vs Improved NBCO, all cities) rerun -> artifacts/reruns/. Detached (survives SSH disconnect).
#   reruns/run_table3.sh            # full paper budget
#   reruns/run_table3.sh smoke      # fast plumbing check (1 iter, TEMP_ prefix)
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p artifacts/cli_logs
PROFILE="${1:-full}"
if [ "$PROFILE" = "smoke" ]; then SUITE_ARGS="--suite-smoke suite_rerun_smoke"; else SUITE_ARGS="--suite suite_rerun"; fi
setsid nohup python scripts/run_table3.py --profile "$PROFILE" $SUITE_ARGS     > artifacts/cli_logs/rerun_table3.log 2>&1 < /dev/null &
PID=$!
sleep 3
echo "Table 3 (NeuralBCO vs Improved NBCO, all cities) rerun started (PID $PID), profile=$PROFILE"
echo "log:  tail -f artifacts/cli_logs/rerun_table3.log"
echo "out:  artifacts/reruns/"
echo "TB:   tensorboard --logdir artifacts/runs/table3_nbco_vs_our"
echo "stop: kill $PID"
