#!/usr/bin/env bash
# Start Jupyter Lab detached so it survives SSH disconnects.
#   ./run_notebook.sh
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p artifacts/jupyter_logs
setsid nohup jupyter lab --no-browser --ip 127.0.0.1 --port 8888 \
    > artifacts/jupyter_logs/jupyter.log 2>&1 < /dev/null &
sleep 3
echo "Jupyter started (PID $!). URL with token:"
grep -Eo "http://127.0.0.1:8888/lab\?token=[a-z0-9]+" artifacts/jupyter_logs/jupyter.log | tail -1
echo "Stop: kill $!"
