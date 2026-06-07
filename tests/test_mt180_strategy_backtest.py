from __future__ import annotations

from pathlib import Path

import pandas as pd

from gushen.mt180_strategy_backtest import (
    backtest_signal,
    evaluate_strategy,
    parse_strategy_item,
    select_and_parse_strategies,
)


def make_frame() -> pd.DataFrame:
    rows = []
    close = 15.0
    for index in range(180):
        close += -0.03 if index < 90 else 0.08
        rows.append(
            {
                "trade_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
                "code": "000001.SZ",
                "name": "Ping An Bank",
                "open": close - 0.01,
                "high": close + 0.08,
                "low": close - 0.08,
                "close": close,
                "volume": 1000 + index * 10,
                "amount": (1000 + index * 10) * close,
            }
        )
    return pd.DataFrame(rows)


def test_parse_evaluate_and_backtest_simple_cross_formula() -> None:
    formula = "M5:=MA(CLOSE,5);M20:=MA(CLOSE,20);买入:CROSS(M5,M20);"
    strategy = parse_strategy_item(
        {
            "source_id": "demo-1",
            "name": "均线买入",
            "category": "trend",
            "indicator_type_label": "副图",
            "sales_count": 10,
            "formula_file": "formulas/demo.tdx",
        },
        formula,
    )

    signal = evaluate_strategy(strategy, make_frame())

    assert signal is not None
    assert signal.sum() >= 1
    trades = backtest_signal(
        source_id=strategy.source_id,
        name=strategy.name,
        ts_code="000001.SZ",
        stock_name="Ping An Bank",
        frame=make_frame(),
        signal=signal,
        hold_days=5,
        commission=0.0008,
    )
    assert trades
    assert trades[0].entry_date > trades[0].signal_date


def test_select_and_parse_filters_unsupported_finance_formula(tmp_path: Path) -> None:
    factor_dir = tmp_path / "mt180"
    formulas = factor_dir / "formulas"
    formulas.mkdir(parents=True)
    (formulas / "ok.tdx").write_text("M5:=MA(C,5);信号:C>M5;", encoding="utf-8")
    (formulas / "bad.tdx").write_text("PB:FINANCE(1)*C;", encoding="utf-8")
    items = [
        {
            "source_id": "ok",
            "name": "OK",
            "category": "trend",
            "indicator_type_label": "副图",
            "sales_count": 2,
            "formula_file": "formulas/ok.tdx",
        },
        {
            "source_id": "bad",
            "name": "BAD",
            "category": "other",
            "indicator_type_label": "副图",
            "sales_count": 1,
            "formula_file": "formulas/bad.tdx",
        },
    ]

    parsed, rejected = select_and_parse_strategies(factor_dir, items, limit=10)


    assert [item.source_id for item in parsed] == ["ok"]
    assert rejected[0]["reason"] == "unsupported_data_dependency"
