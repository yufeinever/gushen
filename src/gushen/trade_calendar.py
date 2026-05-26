from __future__ import annotations

from datetime import date, datetime, time, timedelta


MARKET_CLOSE = time(15, 30)


def latest_research_trade_date(now: datetime | None = None) -> str:
    current = now or datetime.now()
    candidate = current.date()
    if current.time() < MARKET_CLOSE:
        candidate -= timedelta(days=1)
    return previous_weekday(candidate).isoformat()


def previous_weekday(day: date) -> date:
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day
