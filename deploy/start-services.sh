#!/usr/bin/env bash
# Interim service launcher (until systemd). Fully detaches each service with
# setsid so they survive SSH disconnects. Idempotent: kills old copies first.
set -u
cd /home/tfsm/tfsm
V=/home/tfsm/tfsm/.venv/bin

pkill -f "uvicorn backend" 2>/dev/null || true
pkill -f "betbot.telegram_bot" 2>/dev/null || true
pkill -f "run-daemon" 2>/dev/null || true
sleep 2

setsid "$V/uvicorn" backend.tfsm_api.app:app --host 127.0.0.1 --port 8000 \
  >/tmp/uvicorn.log 2>&1 </dev/null &
setsid "$V/python" -m betbot.telegram_bot >/tmp/bot.log 2>&1 </dev/null &
setsid "$V/tfsm" run-daemon >/tmp/daemon.log 2>&1 </dev/null &

sleep 1
echo "launched: api + bot + daemon (logs in /tmp/{uvicorn,bot,daemon}.log)"
