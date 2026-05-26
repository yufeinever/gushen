from gushen.data import DailyBar
from gushen.fixed_universe_backtest import (
    BacktestConfig,
    TradeRow,
    backtest_stock,
    build_signal,
    simulate_buy_hold_benchmark,
    simulate_portfolio,
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


def test_portfolio_uses_100k_cash_and_20pct_position() -> None:
    config = BacktestConfig(start_date="2026-01-01", end_date="2026-01-31")
    trades = [
        TradeRow(
            code="000066.SZ",
            name="Sample",
            signal_date="2026-01-01",
            entry_date="2026-01-02",
            exit_date="2026-01-05",
            hold_days=2,
            entry_price=10.0,
            exit_price=11.0,
            gross_return=0.1,
            net_return=0.098,
            exit_reason="take_profit",
            signal_close=9.9,
            ma5=9.7,
            ma10=9.6,
            ma20=9.5,
            amount_ratio_5d=1.5,
            signal_note="test",
        )
    ]

    ledger, summary = simulate_portfolio(trades, config)

    assert summary.initial_cash == 100_000.0
    assert ledger[0].status == "filled"
    assert ledger[0].shares == 1900
    assert 19_000 < ledger[0].cash_invested < 20_000
    assert summary.final_equity > summary.initial_cash


def test_portfolio_skips_when_max_positions_are_full() -> None:
    config = BacktestConfig(start_date="2026-01-01", end_date="2026-01-31", max_positions=1)
    trades = [
        TradeRow("000066.SZ", "A", "2026-01-01", "2026-01-02", "2026-01-10", 5, 10, 10.5, 0.05, 0.048, "time", 10, 9, 8, 7, 1.5, "test"),
        TradeRow("002208.SZ", "B", "2026-01-02", "2026-01-03", "2026-01-06", 3, 10, 10.5, 0.05, 0.048, "time", 10, 9, 8, 7, 1.5, "test"),
    ]

    ledger, summary = simulate_portfolio(trades, config)

    assert ledger[0].status == "filled"
    assert ledger[1].status == "skipped"
    assert ledger[1].reason == "max_positions"
    assert summary.trade_count == 1
    assert summary.skipped_count == 1


def test_buy_hold_benchmark_uses_same_cash_and_target_allocation() -> None:
    config = BacktestConfig(start_date="2026-01-01", end_date="2026-01-31")
    histories = {
        "000066.SZ": [
            DailyBar("2026-01-01", "000066.SZ", "Sample", 10.0, 10.2, 10.5, 9.9, 1, 1, 0, 0, 0),
            DailyBar("2026-01-31", "000066.SZ", "Sample", 12.0, 12.5, 12.8, 11.8, 1, 1, 0, 0, 0),
        ]
    }

    rows, summary = simulate_buy_hold_benchmark(histories, config)

    assert summary.initial_cash == 100_000.0
    assert rows[0].shares == 1900
    assert 19_000 < rows[0].cash_invested < 20_000
    assert summary.final_equity > summary.initial_cash
