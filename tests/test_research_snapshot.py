from pathlib import Path

import pytest

from gushen.data import DailyBar
from gushen.agents import StockContext
from gushen.data import MarketFetchResult
from gushen.research import (
    load_or_fetch_a_share_code_names,
    load_or_fetch_daily_snapshot,
    load_or_fetch_top_amount_snapshot,
    write_snapshot,
)


def test_empty_snapshot_cache_is_not_reused(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "data/local/snapshots/a_share_daily_2026-05-22.csv"
    cache.parent.mkdir(parents=True)
    write_snapshot(cache, [])

    monkeypatch.setattr("gushen.research.load_or_fetch_top_amount_snapshot", lambda trade_date: [])
    monkeypatch.setattr("gushen.research.fetch_a_share_code_names", lambda: [])

    with pytest.raises(RuntimeError, match="No valid A-share daily bars"):
        load_or_fetch_daily_snapshot("2026-05-22")

    assert not cache.exists()


def test_code_names_fall_back_to_latest_snapshot(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "data/local/snapshots/a_share_daily_2026-05-20.csv"
    cache.parent.mkdir(parents=True)
    write_snapshot(
        cache,
        [
            DailyBar(
                trade_date="2026-05-20",
                code="000001.SZ",
                name="平安银行",
                open=10,
                close=10,
                high=10,
                low=10,
                volume=1,
                amount=1,
                amplitude=0,
                pct_change=0,
                turnover=0,
            )
        ],
    )

    def fail_code_names():
        raise RuntimeError("source unavailable")

    monkeypatch.setattr("gushen.research.fetch_a_share_code_names", fail_code_names)

    assert load_or_fetch_a_share_code_names() == [("000001", "平安银行")]


def test_top_amount_snapshot_fetches_only_ranked_targets(monkeypatch) -> None:
    calls = []

    def stock(code: str, name: str, rank: int, amount: float) -> StockContext:
        return StockContext(
            date="2026-05-21",
            code=code,
            name=name,
            amount_rank=rank,
            amount=amount,
            pct_change=0.0,
            momentum_5d=0.0,
            volatility_20d=0.0,
        )

    def fake_top_amount(limit: int):
        return MarketFetchResult(
            trade_date="2026-05-21",
            source="test",
            stocks=[
                stock("000001.SZ", "平安银行", 1, 200),
                stock("000002.SZ", "万科A", 2, 100),
            ][:limit],
        )

    def fake_daily_bar(code: str, name: str, trade_date: str):
        calls.append(code)
        return DailyBar(
            trade_date=trade_date,
            code=f"{code}.SZ",
            name=name,
            open=10,
            close=10,
            high=10,
            low=10,
            volume=1,
            amount=300 if code == "000001" else 100,
            amplitude=0,
            pct_change=0,
            turnover=0,
        )

    monkeypatch.setattr("gushen.research.fetch_top_amount_stocks", fake_top_amount)
    monkeypatch.setattr("gushen.research.fetch_daily_bar", fake_daily_bar)

    rows = load_or_fetch_top_amount_snapshot("2026-05-21", limit=2)

    assert calls == ["000001", "000002"] or calls == ["000002", "000001"]
    assert [row.code for row in rows] == ["000001.SZ", "000002.SZ"]
