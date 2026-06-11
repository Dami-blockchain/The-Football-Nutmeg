# The Football Nutmeg Agent

Multi-exchange football prediction-market bot for Polymarket + Limitless.

CLI: `nutmeg` (compatibility aliases: `tfsm`, `betbot`). Python package: `betbot`.

## Status

Phase 1 — paper-mode skeleton. Reads fixtures from football-data.org, computes
form, logs the model's favourite outcome to SQLite. No exchange calls yet.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# edit .env, at minimum set FOOTBALL_DATA_API_KEY
tfsm init-db
tfsm run-once
tfsm bets list
```

## Layout

```
src/betbot/
  config.py          Settings (pydantic-settings)
  logging.py         structlog setup
  main.py            CLI entrypoint
  data/
    football_data.py football-data.org REST client
    form.py          FormService (last-5 + opponent strength)
    models.py        domain models
  strategy/
    engine.py        StrategyEngine (softmax + edge filter)
    probabilities.py pure math helpers
  storage/
    db.py            SQLAlchemy engine + session
    models.py        ORM models
    repos.py         repository helpers
  exchanges/
    base.py          ExchangeAdapter Protocol (no impls in Phase 1)
  utils/
    cache.py         TTL cache
```
