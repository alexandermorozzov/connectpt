#!/usr/bin/env bash
# Launch an NBCO experiment detached so it survives an SSH disconnect.
#
# Every argument is passed straight through to scripts/run_nbco.py:
#   scripts/run_nbco.sh table3_nbco_vs_our --suite suite_rerun
#   scripts/run_nbco.sh table5_fig5_5model --cities Mumford1
#   scripts/run_nbco.sh ekb_case_study --suite suite_smoke
#   scripts/run_nbco.sh table4_fig4_our_pareto -p n_iterations=200
#
# Unlike PowerShell, a shell CAN merge stdout+stderr into one file, so the
# console log (INFO + tqdm) is captured here; run_nbco.py additionally writes
# its own INFO log next to the results.
set -e
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
mkdir -p artifacts/cli_logs

LOG_PATH="artifacts/cli_logs/nbco_$(date +%Y%m%d_%H%M%S).log"
setsid nohup python -u scripts/run_nbco.py "$@" \
    > "$LOG_PATH" 2>&1 < /dev/null &
PID=$!
sleep 2
echo "NBCO run started (PID $PID), args: $*"
echo "  follow: tail -f $LOG_PATH"
echo "  stop:   kill $PID"
echo "  board:  tensorboard --logdir artifacts/runs"
