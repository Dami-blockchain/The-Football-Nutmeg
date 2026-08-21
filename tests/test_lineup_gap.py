"""A lineup the feed cannot supply must reach the operator, not the void.

The live fault: Highlightly stopped serving pre-match starting XIs. The alert
path swallowed that completely — `lineup_fn` returned None, the message fell
back to its "lineup not yet confirmed" caveat, and NOTHING was logged. Not one
warning in a full day of daemon output. The feature had been dead for weeks and
the only way to find out was to ask the bot, which said it had no lineup access.

Pins:
  * the LATE alert (KO-10min) flags a missing XI — that alert exists solely to
    show one, so its absence there is a broken feed;
  * the EARLY alert does NOT flag — it deliberately fires before XIs are posted,
    and an alarm on normal operation is one the operator learns to ignore;
  * every fixture flags separately, so three matches in an evening send three;
  * a repeat of the same alert for the same fixture stays quiet;
  * a raised exception is reported, not just an empty result;
  * the user still gets their prediction — flagging must not eat the alert.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from betbot.daily_jobs import (
    lineup_gap_is_notable,
    report_lineup_gap,
    send_prediction_alert,
)
from betbot.notify import reset_operator_notify_cooldowns

NOW = datetime(2026, 8, 21, 18, 50, tzinfo=timezone.utc)
KO = datetime(2026, 8, 21, 19, 0, tzinfo=timezone.utc)


def _baseline(fixture_id=1, kickoff=KO):
    return SimpleNamespace(
        fixture_id=fixture_id, competition_code="PD",
        home_team="Real Betis", away_team="Real Sociedad",
        kickoff=kickoff, p_home=0.5, p_draw=0.25, p_away=0.25,
        paper_bets=[],
    )


class Sent:
    """Recording sender. Deliberately NOT a list subclass — an empty list is
    falsy, and any `send_fn or default` in the code under test would silently
    discard it and fall through to the real transport."""

    def __init__(self):
        self.calls: list[tuple[int, str]] = []

    async def __call__(self, settings, chat_id, text, *a, **kw):
        self.calls.append((chat_id, text))
        return True

    def __len__(self):
        return len(self.calls)

    def __getitem__(self, i):
        return self.calls[i]


@pytest.fixture(autouse=True)
def _cooldowns():
    reset_operator_notify_cooldowns()
    yield
    reset_operator_notify_cooldowns()


@pytest.fixture
def settings(settings):
    """An operator chat id, or notify_operator has nobody to reach."""
    return settings.model_copy(update={"telegram_allowed_user_id": 4242})


@pytest.fixture(autouse=True)
def _db(tmp_path):
    """The send path writes the reveal ledger, so it needs an engine."""
    from betbot.storage.db import init_engine

    init_engine(tmp_path / "lineup_gap.sqlite")
    yield


# ----------------------------------------------------------------------
# When a gap is worth an alarm
# ----------------------------------------------------------------------
def test_the_late_alert_flags_a_missing_xi():
    assert lineup_gap_is_notable("late", KO, NOW) is True


def test_the_early_alert_does_not_flag():
    """It fires BEFORE XIs are posted; alarming there is alarming on normal ops."""
    early = KO - timedelta(minutes=70)
    assert lineup_gap_is_notable("early", KO, early) is False


def test_an_undated_fixture_is_reported_rather_than_swallowed():
    assert lineup_gap_is_notable("late", None, NOW) is True


def test_a_late_alert_far_from_kickoff_does_not_flag():
    """Guard the guard: a mis-scheduled late job hours out is not a feed fault."""
    assert lineup_gap_is_notable("late", KO, KO - timedelta(hours=5)) is False


def test_a_naive_kickoff_is_treated_as_utc():
    assert lineup_gap_is_notable("late", KO.replace(tzinfo=None), NOW) is True


# ----------------------------------------------------------------------
# The message itself
# ----------------------------------------------------------------------
async def test_the_operator_is_told_which_fixture(settings):
    sent = Sent()
    assert await report_lineup_gap(
        settings, _baseline(), alert_tag="late", now=NOW, send_fn=sent
    )
    body = sent[0][1]
    assert "Real Betis" in body and "Real Sociedad" in body
    assert "Lineup unavailable" in body


async def test_the_error_is_quoted_when_the_fetch_raised(settings):
    sent = Sent()
    await report_lineup_gap(
        settings, _baseline(), alert_tag="late", error="429 rate limited",
        now=NOW, send_fn=sent,
    )
    assert "429 rate limited" in sent[0][1]


async def test_three_fixtures_in_one_evening_send_three_flags(settings):
    """A per-kind cooldown would have swallowed the 2nd and 3rd match."""
    sent = Sent()
    for fid in (1, 2, 3):
        await report_lineup_gap(
            settings, _baseline(fixture_id=fid), alert_tag="late",
            now=NOW, send_fn=sent,
        )
    assert len(sent) == 3


async def test_a_repeat_for_the_same_fixture_stays_quiet(settings):
    sent = Sent()
    for _ in range(3):
        await report_lineup_gap(
            settings, _baseline(), alert_tag="late", now=NOW, send_fn=sent
        )
    assert len(sent) == 1


async def test_the_early_alert_sends_nothing(settings):
    sent = Sent()
    assert await report_lineup_gap(
        settings, _baseline(), alert_tag="early",
        now=KO - timedelta(minutes=70), send_fn=sent,
    ) is False
    assert sent.calls == []


# ----------------------------------------------------------------------
# Wired into the alert path
# ----------------------------------------------------------------------
async def test_a_missing_lineup_flags_the_operator_and_still_sends(settings):
    """The prediction must still reach the user — the flag is an extra, not a swap."""
    operator = Sent()
    users = Sent()

    async def _no_lineup(baseline):
        return None, 0.0, 0.0, None

    sent = await send_prediction_alert(
        settings, 1,
        send_fn=users,
        prediction_fn=lambda fid: _baseline(),
        lineup_fn=_no_lineup,
        rescore_fn=None,
        users_fn=lambda: [SimpleNamespace(telegram_user_id=7, tier="free")],
        entitlement_fn=lambda *a, **kw: SimpleNamespace(
            allowed=True, reason="", reveals=[]
        ),
        alert_tag="late",
        operator_send_fn=operator,
        now=NOW,
    )
    assert len(operator) == 1, "operator was not told the lineup was missing"
    assert sent == 1, "the user lost their prediction"


async def test_a_lineup_that_arrives_flags_nothing(settings):
    operator = Sent()
    users = Sent()

    async def _with_lineup(baseline):
        return (
            {"home": {"formation": "4-3-3", "xi": ["A"] * 11},
             "away": {"formation": "4-4-2", "xi": ["B"] * 11}},
            0.0, 0.0, None,
        )

    await send_prediction_alert(
        settings, 1,
        send_fn=users,
        prediction_fn=lambda fid: _baseline(),
        lineup_fn=_with_lineup,
        rescore_fn=None,
        users_fn=lambda: [SimpleNamespace(telegram_user_id=7, tier="free")],
        entitlement_fn=lambda *a, **kw: SimpleNamespace(
            allowed=True, reason="", reveals=[]
        ),
        alert_tag="late",
        operator_send_fn=operator,
        now=NOW,
    )
    assert operator.calls == []


async def test_a_raising_lineup_fn_flags_the_operator(settings):
    operator = Sent()
    users = Sent()

    async def _boom(baseline):
        raise RuntimeError("highlightly 503")

    await send_prediction_alert(
        settings, 1,
        send_fn=users,
        prediction_fn=lambda fid: _baseline(),
        lineup_fn=_boom,
        rescore_fn=None,
        users_fn=lambda: [SimpleNamespace(telegram_user_id=7, tier="free")],
        entitlement_fn=lambda *a, **kw: SimpleNamespace(
            allowed=True, reason="", reveals=[]
        ),
        alert_tag="late",
        operator_send_fn=operator,
        now=NOW,
    )
    assert len(operator) == 1
    assert "highlightly 503" in operator[0][1]
