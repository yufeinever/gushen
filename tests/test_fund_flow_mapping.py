from pathlib import Path

from gushen.data import DailyBar
from gushen.fund_flow_mapping import StockFundFlowMapRow, load_or_build_stock_fund_flow_map


def test_fund_flow_map_cache_roundtrip(tmp_path: Path) -> None:
    cache_dir = tmp_path / "fund_flows"
    cache_dir.mkdir()
    cache_path = cache_dir / "stock_fund_flow_map_2026-05-20.csv"
    cache_path.write_text(
        (
            "trade_date,code,name,main_net_inflow,main_net_pct,main_rank_today,"
            "main_net_inflow_5d,main_net_pct_5d,main_rank_5d,source_status,source,"
            "confidence,updated_at,note\n"
            "2026-05-20,603986,\u5146\u6613\u521b\u65b0,1486534000,5.6,1,"
            "900000000,1.8,3,ok,AKShare stock_individual_fund_flow,0.88,"
            "2026-05-22T09:30:00,cached\n"
        ),
        encoding="utf-8",
    )
    top100 = [_bar()]

    result = load_or_build_stock_fund_flow_map(top100, "2026-05-20", cache_dir)

    assert result["603986"] == StockFundFlowMapRow(
        trade_date="2026-05-20",
        code="603986",
        name="\u5146\u6613\u521b\u65b0",
        main_net_inflow=1486534000.0,
        main_net_pct=5.6,
        main_rank_today=1,
        main_net_inflow_5d=900000000.0,
        main_net_pct_5d=1.8,
        main_rank_5d=3,
        source_status="ok",
        source="AKShare stock_individual_fund_flow",
        confidence=0.88,
        updated_at="2026-05-22T09:30:00",
        note="cached",
    )


def test_fund_flow_map_does_not_keep_unusable_failed_cache(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "fund_flows"
    cache_dir.mkdir()
    cache_path = cache_dir / "stock_fund_flow_map_2026-05-20.csv"
    cache_path.write_text(
        (
            "trade_date,code,name,main_net_inflow,main_net_pct,main_rank_today,"
            "main_net_inflow_5d,main_net_pct_5d,main_rank_5d,source_status,source,"
            "confidence,updated_at,note\n"
            "2026-05-20,603986,\u5146\u6613\u521b\u65b0,,,,,,,"
            "failed,,0,2026-05-22T09:30:00,failed\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("gushen.fund_flow_mapping.build_stock_fund_flow_map", lambda top100, trade_date: [])

    result = load_or_build_stock_fund_flow_map([_bar()], "2026-05-20", cache_dir)

    assert result == {}


def _bar() -> DailyBar:
    return DailyBar(
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
