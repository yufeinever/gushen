from pathlib import Path

import pandas as pd

from gushen.mature_backtest import (
    build_aligned_index_hold_benchmark,
    build_ipo_window_low_hold_benchmark,
    load_ohlcv,
    run_backtesting_py_report,
)


def test_load_ohlcv_normalizes_daily_bar_csv(tmp_path: Path) -> None:
    path = tmp_path / "bars.csv"
    path.write_text(
        "trade_date,open,high,low,close,volume\n"
        "2026-01-02,10,11,9,10.5,1000\n"
        "2026-01-01,9,10,8,9.5,900\n",
        encoding="utf-8",
    )

    frame = load_ohlcv(path)

    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert frame.index.is_monotonic_increasing
    assert frame.iloc[0]["Close"] == 9.5


def test_ipo_window_low_hold_benchmark_buys_window_low() -> None:
    dates = pd.date_range("2026-01-01", periods=8, freq="D")
    close = [12.0, 11.0, 9.0, 10.0, 8.0, 9.5, 9.8, 10.0]
    data = pd.DataFrame(
        {
            "Open": [value + 0.1 for value in close],
            "High": [value + 0.2 for value in close],
            "Low": [value - 0.2 for value in close],
            "Close": close,
            "Volume": [1000.0] * len(close),
        },
        index=dates,
    )

    benchmark = build_ipo_window_low_hold_benchmark(
        data,
        cash=100_000,
        commission=0.0,
        window_start_bar=3,
        window_end_bar=6,
    )

    assert benchmark["status"] == "triggered"
    assert benchmark["entry_date"] == "2026-01-05"
    assert benchmark["entry_price"] == 8.0
    assert benchmark["window_low_bar"] == 5
    assert benchmark["exit_price"] == 10.0
    assert benchmark["return_pct"] == 25.0

def test_aligned_index_hold_benchmark_uses_same_entry_window() -> None:
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    index_data = pd.DataFrame(
        {
            "Open": [100.0, 102.0, 104.0, 106.0, 108.0],
            "High": [101.0, 103.0, 105.0, 107.0, 109.0],
            "Low": [99.0, 101.0, 103.0, 105.0, 107.0],
            "Close": [101.0, 103.0, 105.0, 107.0, 110.0],
            "Volume": [1000.0] * 5,
        },
        index=dates,
    )

    benchmark = build_aligned_index_hold_benchmark(
        index_data,
        entry_date="2026-01-02",
        exit_date="2026-01-05",
        cash=100_000,
        commission=0.0,
        name="SSE Composite",
    )

    assert benchmark["status"] == "triggered"
    assert benchmark["entry_date"] == "2026-01-02"
    assert benchmark["exit_date"] == "2026-01-05"
    assert benchmark["return_pct"] == (110.0 / 102.0 - 1) * 100


def test_run_backtesting_py_report_writes_panel(tmp_path: Path) -> None:
    dates = pd.date_range("2025-01-01", periods=90, freq="D")
    rows = []
    for index, day in enumerate(dates):
        close = 10 + index * 0.05
        rows.append(
            {
                "trade_date": day.strftime("%Y-%m-%d"),
                "open": close - 0.02,
                "high": close + 0.08,
                "low": close - 0.08,
                "close": close,
                "volume": 1000 + index * 20,
            }
        )
    data_path = tmp_path / "bars.csv"
    pd.DataFrame(rows).to_csv(data_path, index=False)

    summary = run_backtesting_py_report(data_path, tmp_path / "out", open_browser=False)

    assert summary["engine"] == "backtesting.py"
    assert Path(summary["panel_path"]).exists()
    assert Path(summary["stats_path"]).exists()
    assert Path(summary["equity_path"]).exists()
    assert "ipo_window_low_hold" in summary["benchmarks"]
    assert "aligned_index_hold" in summary["benchmarks"]
