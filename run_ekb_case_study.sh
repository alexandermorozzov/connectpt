#!/usr/bin/env bash
# Run the EKB case study detached so it survives SSH disconnects (long run).
#   ./run_ekb_case_study.sh              # full paper budget
#   ./run_ekb_case_study.sh --profile smoke
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p artifacts/cli_logs
ARGS="${*:---profile full}"
setsid nohup python scripts/run_ekb_case_study.py $ARGS \
    > artifacts/cli_logs/ekb_case_study.log 2>&1 < /dev/null &
PID=$!
sleep 3
echo "EKB case study started (PID $PID), args: $ARGS"
echo "log:  tail -f artifacts/cli_logs/ekb_case_study.log"
echo "TB:   tensorboard --logdir artifacts/runs/ekb_case_study/tb"
echo "stop: kill $PID"
