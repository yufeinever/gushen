from gushen.data import DailyBar
from gushen.technical_timing import analyze_technical_timing


def _bar(index: int, close: float, amount: float = 1_000_000.0) -> DailyBar:
    return DailyBar(
        trade_date=f"2026-03-{index + 1:02d}",
        code="000001.SZ",
        name="Sample",
        open=close - 0.2,
        close=close,
        high=close + 0.4,
        low=close - 0.4,
        volume=100_000,
        amount=amount,
        amplitude=0.03,
        pct_change=0.01,
        turnover=0.02,
    )


def test_technical_timing_marks_insufficient_history() -> None:
    latest = _bar(10, 10.0)
    row = analyze_technical_timing(latest, [_bar(index, 9.0 + index * 0.1) for index in range(10)], latest.trade_date, 3)

    assert row.data_sufficiency == "insufficient"
    assert row.observation_action == "data_insufficient"
    assert "at least 20 daily bars" in row.missing_data


def test_technical_timing_paper_watch_for_ordered_mas_near_ma5() -> None:
    history = [_bar(index, 10 + index * 0.1) for index in range(60)]
    latest = DailyBar(
        trade_date="2026-05-20",
        code="000001.SZ",
        name="Sample",
        open=15.7,
        close=16.0,
        high=16.2,
        low=15.5,
        volume=180_000,
        amount=1_800_000.0,
        amplitude=0.04,
        pct_change=0.02,
        turnover=0.03,
    )

    row = analyze_technical_timing(latest, [*history, latest], latest.trade_date, 1)

    assert row.data_sufficiency == "sufficient_for_paper_observation"
    assert row.trend_state == "strong_uptrend"
    assert row.observation_action == "paper_watch"
    assert row.ma5_gap is not None and row.ma5_gap <= 0.08
    assert row.support_price is not None


def test_technical_timing_avoids_long_upper_shadow_chasing() -> None:
    history = [_bar(index, 10 + index * 0.08) for index in range(60)]
    latest = DailyBar(
        trade_date="2026-05-20",
        code="000001.SZ",
        name="Sample",
        open=14.6,
        close=14.8,
        high=16.3,
        low=14.5,
        volume=220_000,
        amount=2_200_000.0,
        amplitude=0.12,
        pct_change=0.015,
        turnover=0.04,
    )

    row = analyze_technical_timing(latest, [*history, latest], latest.trade_date, 1)

    assert row.kline_state == "long_upper_shadow"
    assert row.observation_action == "observe_pullback"
    assert "upper shadow" in row.risk_note
