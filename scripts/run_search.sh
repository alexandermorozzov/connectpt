#!/usr/bin/env bash
# Launch a config-driven bee-colony search detached.
set -e
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
mkdir -p artifacts/logs

LOG_PATH="artifacts/logs/search_$(date +%Y%m%d_%H%M%S).log"
setsid nohup python -u scripts/run_search.py "$@" \
    > "$LOG_PATH" 2>&1 < /dev/null &
PID=$!
sleep 2
echo "Search started (PID $PID), args: $*"
echo "  follow: tail -f $LOG_PATH"
echo "  stop:   kill $PID"
