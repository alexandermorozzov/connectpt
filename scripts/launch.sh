#!/usr/bin/env bash
# Launch any scripts/*.py in the background with a log; it survives an SSH
# disconnect. Linux twin of launch.ps1.
#
# The first argument is the script; every other argument goes to it unchanged.
# Without --log the launcher adds --log <log-dir>/<script>_<stamp>.log
# (<log-dir> = --log-dir or artifacts/logs). Console output (tqdm, tracebacks)
# goes to <log>.out. --wait blocks until the run ends and returns its exit code.
#
#   scripts/launch.sh run.py --config-name runs/train_construction_overnight trainer=ppo_construction
#   scripts/launch.sh run_search.py --methods ea nea --instances mandl --alpha 0 1
#   scripts/launch.sh --wait run.py --config-name runs/train --dry-run
set -e
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate

wait_for_exit=0
tokens=()
for arg in "$@"; do
    if [ "$arg" = "--wait" ]; then wait_for_exit=1; else tokens+=("$arg"); fi
done
if [ ${#tokens[@]} -eq 0 ] || [[ "${tokens[0]}" != *.py ]]; then
    echo "usage: launch.sh [--wait] <script>.py [arguments]" >&2
    exit 2
fi
script="scripts/${tokens[0]}"
[ -f "$script" ] || { echo "no script $script" >&2; exit 2; }
args=("${tokens[@]:1}")

log=""
log_dir="artifacts/logs"
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[i]}" in
        --log) log="${args[i + 1]}" ;;
        --log-dir) log_dir="${args[i + 1]}" ;;
    esac
done
if [ -z "$log" ]; then
    stem="$(basename "$script" .py)"
    log="$log_dir/${stem}_$(date +%Y%m%d_%H%M%S).log"
    args+=(--log "$log")
fi
mkdir -p "$(dirname "$log")"

setsid nohup python -u "$script" "${args[@]}" > "$log.out" 2>&1 < /dev/null &
pid=$!
echo "Started ${tokens[0]} (PID $pid)"
echo "  log:    $log"
echo "  follow: tail -f $log"
echo "  stop:   kill $pid"
echo "  board:  the existing TensorBoard server (--logdir artifacts)"
if [ $wait_for_exit -eq 1 ]; then
    wait $pid
fi
