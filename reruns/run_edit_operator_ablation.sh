#!/usr/bin/env bash
# EA vs RSL-EA vs NEA-Edit, alpha 0..1 step 0.1, adjustment off.
# Full mode runs Mandl and Mumford0-3 by default and writes to artifacts/reruns/.
#
#   reruns/run_edit_operator_ablation.sh
#   reruns/run_edit_operator_ablation.sh smoke
#   reruns/run_edit_operator_ablation.sh full --cities Mumford1
#   reruns/run_edit_operator_ablation.sh full --cities Mandl Mumford0
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

setsid nohup python scripts/run_edit_operator_ablation.py \
    --profile "$PROFILE" $SUITE_ARGS "${EXTRA_ARGS[@]}" \
    > artifacts/cli_logs/rerun_edit_operator_ablation.log 2>&1 < /dev/null &
PID=$!
sleep 3

echo "Edit-operator ablation started (PID $PID), profile=$PROFILE"
echo "log:  tail -f artifacts/cli_logs/rerun_edit_operator_ablation.log"
echo "out:  artifacts/reruns/"
echo "TB:   tensorboard --logdir artifacts/runs/edit_operator_ablation"
echo "stop: kill $PID"
