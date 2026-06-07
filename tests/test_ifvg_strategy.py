import pandas as pd

from gushen.ifvg_strategy import (
    detect_ifvg_signals,
    run_ifvg_backtest,
    run_ifvg_batch,
)


def make_ifvg_rows() -> pd.DataFrame:
    rows = []
    close = 10.0
    for index in range(90):
        close += 0.03
        rows.append(
            {
                "trade_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
                "code": "000001.SZ",
                "name": "Ping An Bank",
                "open": close - 0.02,
                "high": close + 0.05,
                "low": close - 0.05,
                "close": close,
                "volume": 1000 + index,
            }
        )

    pattern = [
        (12.50, 12.55, 12.35, 12.40),
        (12.25, 12.30, 12.05, 12.10),
        (12.00, 12.10, 11.70, 11.80),  # bearish FVG high < candle-1 low
        (12.00, 12.90, 11.95, 12.80),  # inversion through the bearish FVG
        (12.70, 13.05, 12.65, 13.00),
        (12.75, 13.60, 12.20, 13.40),  # retest and bullish displacement
        (13.55, 14.20, 13.50, 14.00),
        (14.05, 14.40, 13.90, 14.30),
    ]
    start = len(rows)
    for offset, (open_price, high, low, close_price) in enumerate(pattern):
        rows.append(
            {
                "trade_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=start + offset),
                "code": "000001.SZ",
                "name": "Ping An Bank",
                "open": open_price,
                "high": high,
                "low": low,
                "close": close_price,
                "volume": 5000 + offset,
            }
        )
    return pd.DataFrame(rows)


def test_detect_ifvg_signals_finds_inverted_bullish_zone() -> None:
    frame = make_ifvg_rows()

    signals = detect_ifvg_signals(
        frame,
        htf_window=20,
        htf_slope_window=3,
        min_gap_pct=0.001,
        confirm_window=3,
    )

    assert signals
    assert signals[-1].direction == "bullish"
    assert signals[-1].lower < signals[-1].upper
    assert signals[-1].confirmation == "3bar_structure_break"


def test_run_ifvg_backtest_produces_trade_with_risk_controls() -> None:
    frame = make_ifvg_rows()

    result, trades, signals = run_ifvg_backtest(
        frame,
        ts_code="000001.SZ",
        name="Ping An Bank",
        htf_window=20,
        htf_slope_window=3,
        min_gap_pct=0.001,
        confirm_window=3,
        risk_reward=1.0,
        max_hold_bars=5,
    )

    assert result.status == "tested"
    assert result.signals == len(signals)
    assert trades
    assert trades[0].entry_date > trades[0].signal_date
    assert trades[0].stop_loss < trades[0].entry_price
    assert trades[0].take_profit > trades[0].entry_price


def test_run_ifvg_batch_writes_outputs(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "out"
    cache_dir.mkdir()
    make_ifvg_rows().to_csv(cache_dir / "000001.SZ_2024-01-01_2024-04-15.csv", index=False)

    result = run_ifvg_batch(
        cache_dir=cache_dir,
        output_dir=output_dir,
        limit=10,
        min_bars=50,
        htf_window=20,
        htf_slope_window=3,
        min_gap_pct=0.001,
        confirm_window=3,
        risk_reward=1.0,
    )

    assert result.stocks_tested == 1
    assert result.total_trades >= 1
    assert (output_dir / "ifvg_stock_summary.csv").exists()
    assert (output_dir / "ifvg_trades.csv").exists()


def test_bullish_only_mode_excludes_bearish_signals() -> None:
    frame = make_ifvg_rows()

    signals = detect_ifvg_signals(
        frame,
        htf_window=20,
        htf_slope_window=3,
        min_gap_pct=0.001,
        confirm_window=3,
        directions=("bearish",),
    )

    assert signals == []


def test_existing_zone_can_signal_without_same_bar_new_fvg() -> None:
    frame = make_ifvg_rows()
    signal_index = 95
    previous_two_index = signal_index - 2

    same_bar_has_bullish_fvg = frame.loc[signal_index, "low"] > frame.loc[previous_two_index, "high"]
    same_bar_has_bearish_fvg = frame.loc[signal_index, "high"] < frame.loc[previous_two_index, "low"]

    signals = detect_ifvg_signals(
        frame,
        htf_window=20,
        htf_slope_window=3,
        min_gap_pct=0.001,
        confirm_window=3,
    )

    assert not same_bar_has_bullish_fvg
    assert not same_bar_has_bearish_fvg
    assert signals
    assert signals[-1].signal_date == frame.loc[signal_index, "trade_date"].date().isoformat()
