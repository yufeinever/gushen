import pandas as pd

from gushen.top20_ma5_pullback_strategy import (
    CandidateRow,
    StrategyConfig,
    build_market_frame,
    build_trades,
    is_excluded_board,
    select_candidates,
)


def _frame(code: str, name: str, amounts: list[float], closes: list[float]) -> pd.DataFrame:
    rows = []
    for index, close in enumerate(closes):
        rows.append(
            {
                "trade_date": f"2026-01-{index + 1:02d}",
                "code": code,
                "name": name,
                "open": close - 0.1,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "amount": amounts[index],
            }
        )
    frame = pd.DataFrame(rows)
    frame["ma5"] = frame["close"].rolling(5).mean()
    return frame


def test_excludes_non_mainboard_codes() -> None:
    assert is_excluded_board("300001.SZ")
    assert is_excluded_board("688001.SH")
    assert is_excluded_board("430001.BJ")
    assert not is_excluded_board("600000.SH")
    assert not is_excluded_board("000001.SZ")


def test_select_candidates_requires_top_amount_and_close_above_ma5() -> None:
    config = StrategyConfig(start_date="2026-01-05", end_date="2026-01-08", top_n=1)
    frames = {
        "600000.SH": _frame(
            "600000.SH",
            "A",
            [10, 10, 10, 10, 100, 100, 100, 100],
            [10, 10, 10, 10, 11, 11.2, 11.3, 11.4],
        ),
        "000001.SZ": _frame(
            "000001.SZ",
            "B",
            [10, 10, 10, 10, 50, 50, 50, 50],
            [10, 10, 10, 10, 12, 12.2, 12.3, 12.4],
        ),
        "300001.SZ": _frame(
            "300001.SZ",
            "C",
            [10, 10, 10, 10, 200, 200, 200, 200],
            [10, 10, 10, 10, 12, 12.2, 12.3, 12.4],
        ),
    }

    market = build_market_frame(frames, config)
    candidates = select_candidates(market, config)

    assert candidates
    assert {candidate.code for candidate in candidates} == {"600000.SH"}


def test_build_trades_waits_for_ma5_pullback_and_exits_next_day() -> None:
    config = StrategyConfig(start_date="2026-01-05", end_date="2026-01-10", wait_days=3)
    frame = pd.DataFrame(
        [
            {"trade_date": "2026-01-01", "code": "600000.SH", "name": "A", "open": 10, "high": 10.2, "low": 9.8, "close": 10.0, "amount": 1},
            {"trade_date": "2026-01-02", "code": "600000.SH", "name": "A", "open": 10, "high": 10.2, "low": 9.8, "close": 10.0, "amount": 1},
            {"trade_date": "2026-01-03", "code": "600000.SH", "name": "A", "open": 10, "high": 10.2, "low": 9.8, "close": 10.0, "amount": 1},
            {"trade_date": "2026-01-04", "code": "600000.SH", "name": "A", "open": 10, "high": 10.2, "low": 9.8, "close": 10.0, "amount": 1},
            {"trade_date": "2026-01-05", "code": "600000.SH", "name": "A", "open": 11.0, "high": 12.2, "low": 10.8, "close": 12.0, "amount": 100},
            {"trade_date": "2026-01-06", "code": "600000.SH", "name": "A", "open": 11.3, "high": 11.8, "low": 10.5, "close": 11.2, "amount": 90},
            {"trade_date": "2026-01-07", "code": "600000.SH", "name": "A", "open": 11.4, "high": 12.0, "low": 11.2, "close": 11.8, "amount": 80},
        ]
    )
    frame["ma5"] = frame["close"].rolling(5).mean()
    candidate = CandidateRow(
        signal_date="2026-01-05",
        code="600000.SH",
        name="A",
        amount_rank=1,
        close=12.0,
        ma5=float(frame.loc[4, "ma5"]),
        amount=100,
    )

    trades = build_trades({"600000.SH": frame}, [candidate], config)

    assert len(trades) == 1
    assert trades[0].entry_date == "2026-01-06"
    assert trades[0].exit_date == "2026-01-07"
    assert trades[0].wait_days == 1
