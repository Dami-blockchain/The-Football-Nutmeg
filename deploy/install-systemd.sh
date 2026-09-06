#!/usr/bin/env bash
#
# ============================ STALE / DO NOT USE ============================
# The systemd section below installs the 3 tfsm services as OLD SYSTEM-LEVEL
# units under /etc/systemd/system via sudo. That is NOT how tfsm is deployed
# anymore. As of Aug 2026 the services run as USER units under
# ~/.config/systemd/user (systemctl --user). Use deploy/install-user-systemd.sh
# instead, and take the .service files from this same deploy/ dir (they are now
# exact copies of the installed user units).
#
# Only the NGINX section near the bottom of this script is still current.
# ===========================================================================
# One-shot: install the 3 systemd services + nginx reverse proxy.
# Run as root:  sudo bash /home/tfsm/tfsm/deploy/install-systemd.sh
set -e
D=/home/tfsm/tfsm/deploy

echo "== systemd units =="
cp "$D"/tfsm-api.service "$D"/tfsm-daemon.service "$D"/tfsm-bot.service /etc/systemd/system/
systemctl daemon-reload

echo "== stopping interim setsid copies =="
pkill -f "uvicorn backend" 2>/dev/null || true
pkill -f "betbot.telegram_bot" 2>/dev/null || true
pkill -f "run-daemon" 2>/dev/null || true
sleep 2

echo "== enable + start services =="
systemctl enable --now tfsm-api tfsm-daemon tfsm-bot
sleep 3
systemctl is-active tfsm-api tfsm-daemon tfsm-bot || true

echo "== nginx reverse proxy (HTTP on this host's IP) =="
cp "$D"/nginx-tfsm.conf /etc/nginx/sites-available/tfsm
sed -i 's/SERVER_NAME/_/' /etc/nginx/sites-available/tfsm   # catch-all; replace _ with your domain
ln -sf /etc/nginx/sites-available/tfsm /etc/nginx/sites-enabled/tfsm
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

echo
echo "DONE."
echo "  services:  systemctl status tfsm-api tfsm-daemon tfsm-bot"
echo "  dashboard: http://$(hostname -I | awk '{print $1}')/  (API token required)"
echo "  HTTPS:     point a domain at this IP, edit server_name in"
echo "             /etc/nginx/sites-available/tfsm, then: sudo certbot --nginx -d your.domain"
