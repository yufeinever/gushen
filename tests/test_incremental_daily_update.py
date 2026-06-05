from pathlib import Path

from gushen.data import DailyBar
from gushen.incremental_daily_update import (
    latest_cache_file,
    merge_daily_bars,
    update_one_stock,
    write_daily_bars,
)


def bar(day: str, close: float = 10.0) -> DailyBar:
    return DailyBar(day, "000001.SZ", "A", close, close, close, close, 1, 1, 0, 0, 0)


def test_latest_cache_file_picks_latest_end_date(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    path1 = cache_dir / "qfq" / "000001.SZ_1990-01-01_2026-06-03.csv"
    path2 = cache_dir / "qfq" / "000001.SZ_1990-01-01_2026-06-05.csv"
    write_daily_bars(path1, [bar("2026-06-03")])
    write_daily_bars(path2, [bar("2026-06-05")])

    latest = latest_cache_file(cache_dir, "qfq", "000001.SZ")

    assert latest is not None
    assert latest.path == path2


def test_latest_cache_file_prefers_full_history_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    full_history = cache_dir / "qfq" / "000001.SZ_1990-01-01_2026-06-03.csv"
    short_window = cache_dir / "qfq" / "000001.SZ_2024-05-04_2026-06-05.csv"
    write_daily_bars(full_history, [bar("2026-06-03")])
    write_daily_bars(short_window, [bar("2026-06-05")])

    latest = latest_cache_file(cache_dir, "qfq", "000001.SZ")

    assert latest is not None
    assert latest.path == full_history


def test_merge_daily_bars_replaces_overlap_and_sorts() -> None:
    merged = merge_daily_bars(
        [bar("2026-06-03", 10), bar("2026-06-04", 11)],
        [bar("2026-06-04", 12), bar("2026-06-05", 13)],
    )

    assert [item.trade_date for item in merged] == ["2026-06-03", "2026-06-04", "2026-06-05"]
    assert merged[1].close == 12


def test_update_one_stock_fetches_only_gap_window(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    source = cache_dir / "qfq" / "000001.SZ_1990-01-01_2026-06-03.csv"
    write_daily_bars(source, [bar("2026-06-02", 10), bar("2026-06-03", 11)])
    calls = []

    def fake_fetcher(ts_code, name, start_date, end_date, timeout, adjust):
        calls.append((ts_code, start_date, end_date, adjust))
        return [bar("2026-06-03", 11.5), bar("2026-06-04", 12), bar("2026-06-05", 13)]

    event = update_one_stock(
        {"code": "000001", "name": "A", "rank": 1},
        index=1,
        total=1,
        cache_dir=cache_dir,
        adjust="qfq",
        end_date="2026-06-05",
        overlap_days=1,
        timeout=8,
        dry_run=False,
        fetcher=fake_fetcher,
    )

    assert event["status"] == "downloaded"
    assert calls == [("000001.SZ", "2026-06-02", "2026-06-05", "qfq")]
    assert (cache_dir / "qfq" / "000001.SZ_1990-01-01_2026-06-05.csv").exists()
