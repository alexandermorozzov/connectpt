#!/usr/bin/env bash
# Table 6 (4 BCO configurations, Mumford1 by default) -> artifacts/reruns/.
#   reruns/run_table6.sh
#   reruns/run_table6.sh smoke
#   reruns/run_table6.sh full --cities Mumford0
#   reruns/run_table6.sh full --cities Mandl Mumford0 Mumford1
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p artifacts/cli_logs

PROFILE="${1:-full}"
EXTRA_ARGS=("${@:2}")
if [ "$PROFILE" = "smoke" ]; then
    SUITE_ARGS="--suite-smoke suite_rerun_smoke"
else
    SUITE_ARGS="--suite suite_rerun"
fi

setsid nohup python scripts/run_table6.py \
    --profile "$PROFILE" $SUITE_ARGS "${EXTRA_ARGS[@]}" \
    > artifacts/cli_logs/rerun_table6.log 2>&1 < /dev/null &
PID=$!
sleep 3

echo "Table 6 rerun started (PID $PID), profile=$PROFILE"
echo "log:  tail -f artifacts/cli_logs/rerun_table6.log"
echo "out:  artifacts/reruns/"
echo "TB:   tensorboard --logdir artifacts/runs/table6_bco_comparison"
echo "stop: kill $PID"
