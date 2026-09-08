#!/usr/bin/env bash
# Launch edit-model training detached so it survives an SSH disconnect.
#
# Every argument is passed straight through to scripts/run_training.py:
#   scripts/run_training.sh
#   scripts/run_training.sh --config-name training/edit_scratch
#   scripts/run_training.sh --dry-run
#   scripts/run_training.sh run.seed=3
#
# Unlike PowerShell, a shell CAN merge stdout+stderr into one file, so the
# console log (INFO + tqdm) is captured here; run_training.py additionally
# writes its own INFO log under artifacts/logs.
set -e
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
mkdir -p artifacts/logs

LOG_PATH="artifacts/logs/train_$(date +%Y%m%d_%H%M%S).log"
setsid nohup python -u scripts/run_training.py "$@" \
    > "$LOG_PATH" 2>&1 < /dev/null &
PID=$!
sleep 2
echo "Training started (PID $PID), args: $*"
echo "  follow: tail -f $LOG_PATH"
echo "  stop:   kill $PID"
echo "  board:  tensorboard --logdir artifacts/runs"
