from pathlib import Path

import pandas as pd

from gushen.mature_backtest import load_ohlcv, run_backtesting_py_report


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
