#!/usr/bin/env bash
# Table 6-style random trim/extend ablation -> artifacts/reruns/.
#   reruns/run_table6_random_trim_extend.sh
#   reruns/run_table6_random_trim_extend.sh smoke
#   reruns/run_table6_random_trim_extend.sh full --cities Mandl --n-iterations 200
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

nohup python scripts/run_table6_random_trim_extend.py \
    --profile "$PROFILE" $SUITE_ARGS "${EXTRA_ARGS[@]}" \
    > artifacts/cli_logs/rerun_table6_random_trim_extend.log 2>&1 < /dev/null &
PID=$!
sleep 3

echo "Random trim/extend ablation started (PID $PID), profile=$PROFILE"
echo "log:  tail -f artifacts/cli_logs/rerun_table6_random_trim_extend.log"
echo "out:  artifacts/reruns/"
echo "TB:   tensorboard --logdir artifacts/runs/table6_random_trim_extend"
echo "stop: kill $PID"
