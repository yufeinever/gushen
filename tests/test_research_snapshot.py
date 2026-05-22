from pathlib import Path

import pytest

from gushen.research import load_or_fetch_daily_snapshot, write_snapshot


def test_empty_snapshot_cache_is_not_reused(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "data/local/snapshots/a_share_daily_2026-05-22.csv"
    cache.parent.mkdir(parents=True)
    write_snapshot(cache, [])

    monkeypatch.setattr("gushen.research.fetch_a_share_code_names", lambda: [])

    with pytest.raises(RuntimeError, match="No valid A-share daily bars"):
        load_or_fetch_daily_snapshot("2026-05-22")

    assert not cache.exists()
