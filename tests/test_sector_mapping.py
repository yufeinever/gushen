from pathlib import Path

from gushen.data import DailyBar
from gushen.sector_mapping import StockSectorMapRow, load_or_build_stock_sector_map


def test_sector_map_cache_roundtrip(tmp_path: Path) -> None:
    cache_dir = tmp_path / "sector_maps"
    cache_dir.mkdir()
    cache_path = cache_dir / "stock_sector_map_2026-05-20.csv"
    cache_path.write_text(
        (
            "trade_date,code,name,industry,concepts,source_status,source,confidence,updated_at,note\n"
            "2026-05-20,603986,\u5146\u6613\u521b\u65b0,\u534a\u5bfc\u4f53,"
            "AI\u82af\u7247;DRAM,ok,Sina industry constituents,0.88,2026-05-22T09:30:00,cached\n"
        ),
        encoding="utf-8",
    )
    top100 = [
        DailyBar(
            trade_date="2026-05-20",
            code="603986.SH",
            name="\u5146\u6613\u521b\u65b0",
            open=100,
            close=110,
            high=112,
            low=98,
            volume=1_000_000,
            amount=10_000_000_000,
            amplitude=0.08,
            pct_change=0.06,
            turnover=0.05,
        )
    ]

    result = load_or_build_stock_sector_map(top100, "2026-05-20", cache_dir)

    assert result["603986"] == StockSectorMapRow(
        trade_date="2026-05-20",
        code="603986",
        name="\u5146\u6613\u521b\u65b0",
        industry="\u534a\u5bfc\u4f53",
        concepts="AI\u82af\u7247;DRAM",
        source_status="ok",
        source="Sina industry constituents",
        confidence=0.88,
        updated_at="2026-05-22T09:30:00",
        note="cached",
    )
