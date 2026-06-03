import pandas as pd

from gushen.data import DailyBar
from gushen.guided_factor_backtest import (
    assess_sufficiency,
    bars_to_frame,
    build_factor_frame,
    build_strategy_search_splits,
    calculate_excess_returns,
    normalize_stock_code,
    parse_external_stock_pool,
    parse_guided_stock_pool,
    run_factor_guided_backtest,
    score_factors,
    search_strategy_library,
)


def make_rows(count: int = 520) -> list[DailyBar]:
    dates = pd.date_range("2024-01-01", periods=count, freq="D")
    rows: list[DailyBar] = []
    for index, day in enumerate(dates):
        cycle = index % 20
        close = 10 + index * 0.01 + cycle * 0.03
        open_price = close - 0.02
        high = close + 0.08
        low = close - 0.08
        rows.append(
            DailyBar(
                trade_date=day.strftime("%Y-%m-%d"),
                code="300308.SZ",
                name="Zhongji Innolight",
                open=open_price,
                close=close,
                high=high,
                low=low,
                volume=1000 + index * 10,
                amount=(1000 + index * 10) * close,
                amplitude=(high - low) / close,
                pct_change=0.01 if cycle > 10 else -0.002,
                turnover=0.01 + cycle * 0.001,
            )
        )
    return rows


def test_parse_guided_stock_pool_reads_user_lists() -> None:
    stocks = parse_guided_stock_pool()

    assert len(stocks) == 45
    assert stocks[0].code == "300308"
    assert stocks[0].ts_code == "300308.SZ"
    assert stocks[0].group == "Image 3 - top amount leaders"


def test_parse_external_stock_pool_reads_ranked_csv(tmp_path) -> None:
    path = tmp_path / "top100.csv"
    path.write_text(
        "\ufeff序,代码,名称,金额\n"
        "2,62,深圳华强,46.79亿\n"
        "1,300308,中际旭创,438.81亿\n",
        encoding="utf-8",
    )

    stocks = parse_external_stock_pool(path, limit=2, group="2026-06-03 top100")

    assert [stock.code for stock in stocks] == ["300308", "000062"]
    assert stocks[0].name == "中际旭创"
    assert stocks[0].ts_code == "300308.SZ"
    assert stocks[0].group == "2026-06-03 top100"
    assert normalize_stock_code("sz000725") == "000725"


def test_factor_screening_and_guided_backtest_produces_trades() -> None:
    frame = bars_to_frame(make_rows())

    sufficiency = assess_sufficiency(frame)
    factors = build_factor_frame(frame)
    selected = score_factors(factors, train_end_index=340, max_factors=3)
    trades = run_factor_guided_backtest(factors, selected, train_end_index=340, min_confirmations=1)

    assert sufficiency.status == "pass"
    assert selected
    assert trades
    assert all(trade.entry_date > trade.signal_date for trade in trades)


def test_strategy_library_selects_candidate_on_holdout_window() -> None:
    frame = bars_to_frame(make_rows())
    factors = build_factor_frame(frame)
    train_end, test_start = build_strategy_search_splits(len(factors))
    selected = score_factors(factors, train_end_index=train_end, max_factors=5)

    best, library, trades = search_strategy_library(factors, selected, train_end, test_start)

    assert best is not None
    assert library
    assert trades
    assert all(trade.signal_date >= factors.iloc[test_start]["trade_date"].date().isoformat() for trade in trades)
    assert best.strategy_id == library[0].strategy_id


def test_calculate_excess_returns_keeps_strategy_and_anchor_benchmarks_separate() -> None:
    strategy_excess, strategy_vs_anchor, anchor_excess = calculate_excess_returns(
        strategy_return=12.5,
        anchor_low_return=80.0,
        index_return=10.0,
    )

    assert strategy_excess == 2.5
    assert strategy_vs_anchor == -67.5
    assert anchor_excess == 70.0
