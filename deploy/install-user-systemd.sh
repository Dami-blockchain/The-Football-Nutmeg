#!/usr/bin/env bash
# Install the 3 tfsm services as **user** systemd units (the current, real
# deployment shape as of Aug 2026 — NOT the old system-level units in
# install-systemd.sh, which are stale; see the header there).
#
# Run as the tfsm user (NOT root):  bash deploy/install-user-systemd.sh
#
# Why user units:
#   - No sudo needed for the services themselves; they run as tfsm.
#   - They load .env via the app`s own dotenv from WorkingDirectory, so there
#     is deliberately NO EnvironmentFile= (a malformed .env line would wedge
#     the unit; the app reads it safely instead).
#   - Ordering api -> daemon -> bot is encoded in the unit files so three cold
#     boots never hit the non-concurrency-safe startup migration at once.
set -euo pipefail

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SRC="$(cd "$(dirname "$0")" && pwd)"

echo "== install user units into $UNIT_DIR =="
mkdir -p "$UNIT_DIR"
cp "$SRC"/tfsm-api.service "$SRC"/tfsm-daemon.service "$SRC"/tfsm-bot.service "$UNIT_DIR"/
systemctl --user daemon-reload

echo
echo "== linger (so the units run without an active login session) =="
echo "   This ONE step needs root; run it once:"
echo "     sudo loginctl enable-linger $USER"
if command -v loginctl >/dev/null 2>&1 && loginctl show-user "$USER" 2>/dev/null | grep -q "Linger=yes"; then
  echo "   Linger is already ENABLED for $USER."
else
  echo "   Linger is NOT enabled yet — enable it before relying on auto-start at boot."
fi

echo
echo "== enable + start (operator decides WHEN — commented out on purpose) =="
echo "   Review the units, then run:"
echo "     systemctl --user enable --now tfsm-api tfsm-daemon tfsm-bot"
echo "   Status / logs:"
echo "     systemctl --user status tfsm-api tfsm-daemon tfsm-bot"
echo "     journalctl --user -u tfsm-daemon -f   # (units also append to /tmp/*.log)"
echo
echo "NOTE: nginx reverse proxy is a SEPARATE, root-level concern — see"
echo "      deploy/nginx-tfsm.conf and the nginx section of the old"
echo "      install-systemd.sh (that part is still valid; only its systemd"
echo "      section is stale)."
echo
echo "DONE (units installed + daemon-reloaded; nothing started)."
