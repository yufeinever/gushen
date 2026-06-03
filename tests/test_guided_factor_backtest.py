import pandas as pd

from gushen.data import DailyBar
from gushen.guided_factor_backtest import (
    assess_sufficiency,
    bars_to_frame,
    build_factor_frame,
    parse_guided_stock_pool,
    run_factor_guided_backtest,
    score_factors,
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
