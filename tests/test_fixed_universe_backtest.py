from gushen.data import DailyBar
from gushen.fixed_universe_backtest import (
    BacktestConfig,
    backtest_stock,
    build_signal,
    simulate_exit,
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


def test_build_signal_requires_ordered_mas_and_volume() -> None:
    bars = [_bar(index, 10 + index * 0.03) for index in range(60)]
    bars.append(_bar(60, 12.0, amount=1_800_000.0, pct=0.025))
    bars[-1] = DailyBar(
        bars[-1].trade_date,
        bars[-1].code,
        bars[-1].name,
        11.6,
        12.0,
        12.1,
        11.5,
        bars[-1].volume,
        bars[-1].amount,
        bars[-1].amplitude,
        bars[-1].pct_change,
        bars[-1].turnover,
    )

    signal = build_signal(bars, 60)

    assert signal is not None
    assert signal["amount_ratio_5d"] > 1.2


def test_simulate_exit_respects_t1_no_same_day_sell() -> None:
    config = BacktestConfig(start_date="2026-01-01", end_date="2026-01-10")
    bars = [
        _bar(0, 10.0),
        DailyBar("2026-01-02", "000066.SZ", "Sample", 10.0, 10.2, 11.5, 9.2, 1, 1, 0, 0, 0),
        DailyBar("2026-01-03", "000066.SZ", "Sample", 10.3, 10.4, 11.1, 10.2, 1, 1, 0, 0, 0),
    ]

    exit_index, exit_price, reason = simulate_exit(bars, entry_index=1, config=config)

    assert exit_index == 2
    assert exit_price == 11.0
    assert reason == "take_profit"


def test_backtest_stock_enters_next_day_after_signal() -> None:
    bars = [_bar(index, 10 + index * 0.03) for index in range(60)]
    bars.append(
        DailyBar("2026-03-02", "000066.SZ", "Sample", 11.6, 12.0, 12.1, 11.5, 1, 1_800_000, 0, 0.025, 0)
    )
    bars.extend(
        [
            DailyBar("2026-03-03", "000066.SZ", "Sample", 12.1, 12.2, 12.3, 12.0, 1, 1, 0, 0, 0),
            DailyBar("2026-03-04", "000066.SZ", "Sample", 12.4, 12.8, 13.5, 12.4, 1, 1, 0, 0, 0),
            DailyBar("2026-03-05", "000066.SZ", "Sample", 12.9, 13.0, 13.1, 12.8, 1, 1, 0, 0, 0),
            DailyBar("2026-03-06", "000066.SZ", "Sample", 12.9, 13.0, 13.1, 12.8, 1, 1, 0, 0, 0),
            DailyBar("2026-03-07", "000066.SZ", "Sample", 12.9, 13.0, 13.1, 12.8, 1, 1, 0, 0, 0),
            DailyBar("2026-03-08", "000066.SZ", "Sample", 12.9, 13.0, 13.1, 12.8, 1, 1, 0, 0, 0),
            DailyBar("2026-03-09", "000066.SZ", "Sample", 12.9, 13.0, 13.1, 12.8, 1, 1, 0, 0, 0),
            DailyBar("2026-03-10", "000066.SZ", "Sample", 12.9, 13.0, 13.1, 12.8, 1, 1, 0, 0, 0),
        ]
    )

    trades, signal_count = backtest_stock(
        "000066.SZ",
        "Sample",
        bars,
        BacktestConfig(start_date="2026-01-01", end_date="2026-03-10"),
    )

    assert signal_count >= 1
    assert trades[0].signal_date == bars[60].trade_date
    assert trades[0].entry_date == "2026-03-03"
    assert trades[0].exit_date > trades[0].entry_date
