#!/bin/zsh
# Starts the always-on servers inside one tmux server (`tmux -L srv`):
#   reports  static server for src/neuralssm/reports on :8000
#   watch    monitor.py --watch: starts the live monitor on :8001 whenever an experiment job is running
# Reachable over Tailscale, e.g. http://<tailscale-ip>:8000/ and :8001/.
# Usage: ./serve.sh [start|stop|restart|status]
SRC="${0:A:h}"
PY="$HOME/miniforge3/envs/jaxenv/bin/python"
[[ -x "$PY" ]] || PY=python3
TMUX=${commands[tmux]:-/usr/local/bin/tmux}

up() { $TMUX -L srv has-session -t "$1" 2>/dev/null }
listening() { lsof -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1 }

start() {
    mkdir -p "$SRC/reports"
    if ! up reports; then
        listening 8000 && { echo "port 8000 is already in use by another process; not starting the reports server"; } || \
            $TMUX -L srv new-session -d -s reports -c "$SRC/reports" "$PY -m http.server 8000 --bind 0.0.0.0"
    fi
    up watch || $TMUX -L srv new-session -d -s watch -c "$SRC" "$PY $SRC/monitor.py --watch --port 8001"
}

stop() {
    $TMUX -L srv kill-server 2>/dev/null
    pkill -f "$SRC/monitor.py" 2>/dev/null     # a monitor started by the watcher outlives it
    true
}

status() {
    $TMUX -L srv ls 2>/dev/null || echo "no servers started by serve.sh"
    for p in 8000 8001; do
        listening $p && echo ":$p listening" || echo ":$p not listening$([[ $p == 8001 ]] && echo ' (the monitor only runs while a job is running)')"
    done
}

case "${1:-start}" in
    start) start; status ;;
    stop) stop; status ;;
    restart) stop; sleep 1; start; status ;;
    status) status ;;
    *) echo "usage: $0 [start|stop|restart|status]"; exit 1 ;;
esac
