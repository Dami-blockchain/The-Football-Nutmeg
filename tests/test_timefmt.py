"""User-facing time rendering is EAT (Africa/Nairobi, UTC+3) everywhere.

Pins the shared helper AND the surfaces routed through it. Storage/scheduling
stay UTC; these tests only assert the DISPLAY string.
"""

from __future__ import annotations

from datetime import datetime, timezone

from betbot.timefmt import EAT, eat_datetime, eat_time, to_eat


# ----------------------------------------------------------------------
# The helper
# ----------------------------------------------------------------------
def test_named_zone_not_hardcoded_offset():
    # Self-documenting IANA zone, not a raw +3 — the spec's explicit ask.
    assert EAT.key == "Africa/Nairobi"


def test_utc_instant_renders_expected_eat():
    u = datetime(2026, 9, 6, 18, 30, tzinfo=timezone.utc)  # 18:30Z == 21:30 EAT
    assert eat_time(u) == "21:30 EAT"
    assert eat_time(u, label=False) == "21:30"
    assert eat_datetime(u) == "2026-09-06 21:30 EAT"


def test_naive_datetime_is_treated_as_utc():
    assert eat_time(datetime(2026, 9, 6, 18, 30)) == "21:30 EAT"


def test_none_renders_empty():
    assert eat_time(None) == ""
    assert eat_datetime(None) == ""
    assert to_eat(None) is None


def test_already_eat_is_not_shifted_again():
    # Regression: a datetime ALREADY in EAT must not be pushed forward a second
    # +3. Converting it through the helper is idempotent on the wall clock.
    u = datetime(2026, 9, 6, 18, 30, tzinfo=timezone.utc)
    once = to_eat(u)
    assert once.strftime("%H:%M") == "21:30"
    assert eat_time(once) == "21:30 EAT"        # not 00:30 the next day
    assert to_eat(once) == once                 # fixed point

    # And an aware datetime built directly in EAT stays put.
    native_eat = datetime(2026, 9, 6, 21, 30, tzinfo=EAT)
    assert eat_time(native_eat) == "21:30 EAT"


def test_midnight_utc_is_three_am_eat():
    # The chat-limit reset boundary: midnight UTC == 03:00 EAT (copy pins this).
    assert eat_time(datetime(2026, 9, 7, 0, 0, tzinfo=timezone.utc)) == "03:00 EAT"


# ----------------------------------------------------------------------
# Surfaces routed through the helper
# ----------------------------------------------------------------------
class _Pred:
    def __init__(self, kickoff):
        self.kickoff = kickoff
        self.home_team = "Arsenal"
        self.away_team = "Spurs"
        self.competition_code = "PL"
        self.p_home = 0.6
        self.p_draw = 0.25
        self.p_away = 0.15
        self.home_xg = None
        self.away_xg = None


def test_tips_kickoff_str_is_eat_labeled():
    from betbot.tips import _kickoff_str, format_prediction

    ko = datetime(2026, 9, 6, 18, 30, tzinfo=timezone.utc)
    assert _kickoff_str(_Pred(ko)) == "21:30 EAT"
    body = format_prediction(_Pred(ko))
    assert "21:30 EAT" in body
    assert "18:30" not in body  # the raw UTC time never leaks


def test_notify_kickoff_eat_bare_then_labeled_by_caller():
    from betbot.notify import _kickoff_eat

    ko = datetime(2026, 9, 6, 18, 30, tzinfo=timezone.utc)
    # Bare (caller appends " EAT"); named-zone conversion, not a hardcoded +3.
    assert _kickoff_eat(_Pred(ko)) == "21:30"


def test_change_announcement_timestamp_is_eat():
    from betbot.notify import format_change_announcement

    when = datetime(2026, 9, 6, 18, 30, tzinfo=timezone.utc)
    msg = format_change_announcement(
        "changing the alert copy", rollback="revert the commit", when=when
    )
    assert "2026-09-06 21:30 EAT" in msg
    assert "UTC" not in msg
