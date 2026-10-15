#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WATCH="$ROOT/runs/watcher"
SESSION="cavs_perframe_live_v1"
mkdir -p "$WATCH"
if screen -list | grep -q "[.]$SESSION"; then
  echo "watcher already running in screen: $SESSION"
  exit 0
fi
screen -DmS "$SESSION" bash -lc \
  "exec bash '$ROOT/scripts/watch_remote_and_run_live.sh' >> '$WATCH/watcher.log' 2>&1"
echo "started screen: $SESSION"
