from datetime import datetime

from gushen.trade_calendar import latest_research_trade_date


def test_latest_research_date_before_close_uses_previous_weekday() -> None:
    now = datetime(2026, 5, 26, 9, 23)

    assert latest_research_trade_date(now) == "2026-05-25"


def test_latest_research_date_after_close_uses_today() -> None:
    now = datetime(2026, 5, 26, 16, 0)

    assert latest_research_trade_date(now) == "2026-05-26"


def test_latest_research_date_skips_weekend() -> None:
    now = datetime(2026, 5, 25, 9, 0)

    assert latest_research_trade_date(now) == "2026-05-22"
