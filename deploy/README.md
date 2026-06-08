# Deployment (Phase 9)

The bot runs as three long-lived processes plus a daily cron tick:

| Service | What | Unit |
|---|---|---|
| daemon | scores fixtures + settles bets (daily 08:00 UTC) | `tfsm-daemon.service` |
| api | FastAPI backend + dashboard on 127.0.0.1:8000 | `tfsm-api.service` |
| bot | Telegram bot (deposits + status) | `tfsm-bot.service` |

Until systemd is installed they run under `tmux` (session `tfsm`, windows
`api` / `bot` / `daemon`) — survives SSH disconnects, **not** reboots.

## Install systemd units (needs sudo)

```bash
sudo cp deploy/tfsm-*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tfsm-api tfsm-daemon tfsm-bot
systemctl status tfsm-api tfsm-daemon tfsm-bot
```

Stop the tmux copies first to avoid port/polling conflicts:
`tmux kill-session -t tfsm`.

## Public dashboard (optional, needs a domain)

1. Set a strong `TFSM_API_TOKEN` in `.env` (otherwise the API is open — keep it
   bound to localhost).
2. `sudo cp deploy/nginx-tfsm.conf /etc/nginx/sites-available/tfsm` and edit
   `SERVER_NAME`; symlink into `sites-enabled`, `sudo nginx -t && sudo systemctl reload nginx`.
3. HTTPS: `sudo certbot --nginx -d your.domain`.
4. ufw already allows 80/443.

Without a domain, reach the dashboard via SSH tunnel:
`ssh -L 8000:127.0.0.1:8000 tfsm@<host>` then open http://localhost:8000.

## Backups (optional)

Install litestream, fill `deploy/litestream.yml` with your bucket + creds, and
enable the litestream service to continuously replicate `data/betbot.sqlite`.

## Deploy a new version

```bash
cd ~/tfsm && git pull
source .venv/bin/activate && pip install -e ".[dev,api]"
sudo systemctl restart tfsm-api tfsm-daemon tfsm-bot   # or restart the tmux windows
```

## Egress allowlist (for reference)

football-data.org, gamma-api.polymarket.com, clob.polymarket.com,
api.limitless.exchange, polygon.drpc.org, mainnet.base.org, api.telegram.org.
