from gushen.data import DailyBar
from gushen.strategy_lab import (
    STRATEGIES,
    _strategy_signal,
    build_stock_picks,
)


def _bar(index: int, close: float, amount: float = 1_000_000.0, pct: float = 0.01) -> DailyBar:
    return DailyBar(
        trade_date=f"2026-01-{index + 1:02d}",
        code="000066.SZ",
        name="Sample",
        open=close - 0.05,
        close=close,
        high=close + 0.2,
        low=close - 0.2,
        volume=100_000,
        amount=amount,
        amplitude=0.03,
        pct_change=pct,
        turnover=0.02,
    )


def test_ma20_trend_strategy_finds_ordered_ma_volume_signal() -> None:
    bars = [_bar(index, 10 + index * 0.03) for index in range(80)]
    bars[-1] = _bar(79, 12.4, amount=1_600_000.0, pct=0.02)
    spec = next(item for item in STRATEGIES if item.name == "ma20_trend_hold")

    signal = _strategy_signal(bars, 79, spec)

    assert signal is not None
    assert signal["ma5"] > signal["ma10"] > signal["ma20"]


def test_donchian_strategy_requires_breakout() -> None:
    bars = [_bar(index, 10 + index * 0.01) for index in range(80)]
    bars[-1] = DailyBar("2026-03-21", "000066.SZ", "Sample", 11.0, 12.5, 12.7, 10.9, 1, 1_700_000, 0, 0.04, 0)
    spec = next(item for item in STRATEGIES if item.name == "donchian_20_breakout")

    signal = _strategy_signal(bars, 79, spec)

    assert signal is not None
    assert signal["amount_ratio_5d"] > 1.2


def test_stock_picks_prefers_strategy_only_when_it_beats_buy_hold() -> None:
    from gushen.fixed_universe_backtest import BuyHoldRow, TradeRow

    trades = {
        "test_strategy": [
            TradeRow("000066.SZ", "A", "2026-01-01", "2026-01-02", "2026-01-03", 2, 10, 11, 0.1, 0.09, "time", 10, 9, 8, 7, 1.5, "test"),
            TradeRow("000066.SZ", "A", "2026-01-04", "2026-01-05", "2026-01-06", 2, 10, 11, 0.1, 0.09, "time", 10, 9, 8, 7, 1.5, "test"),
            TradeRow("000066.SZ", "A", "2026-01-07", "2026-01-08", "2026-01-09", 2, 10, 11, 0.1, 0.09, "time", 10, 9, 8, 7, 1.5, "test"),
            TradeRow("000066.SZ", "A", "2026-01-10", "2026-01-11", "2026-01-12", 2, 10, 11, 0.1, 0.09, "time", 10, 9, 8, 7, 1.5, "test"),
            TradeRow("000066.SZ", "A", "2026-01-13", "2026-01-14", "2026-01-15", 2, 10, 11, 0.1, 0.09, "time", 10, 9, 8, 7, 1.5, "test"),
        ]
    }
    buy_hold = [BuyHoldRow("000066.SZ", "A", "2026-01-01", "2026-01-31", 100, 10, 11, 1000, 1100, 100, 0.1)]

    rows = build_stock_picks(trades, buy_hold)
    first = next(row for row in rows if row.code == "000066.SZ")

    assert first.best_strategy == "test_strategy"
    assert first.action == "strategy_watch"
