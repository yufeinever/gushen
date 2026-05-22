from gushen.data import DailyBar
from gushen.deep_analysis import DeepFeatureRow
from gushen.tradingagents_dataset import build_fund_flows, build_sector_themes


def _bar() -> DailyBar:
    return DailyBar(
        trade_date="2026-05-20",
        code="603986.SH",
        name="\u82af\u7247\u6d4b\u8bd5",
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


def _feature() -> DeepFeatureRow:
    return DeepFeatureRow(
        trade_date="2026-05-20",
        code="603986.SH",
        name="\u82af\u7247\u6d4b\u8bd5",
        amount_rank=1,
        close=110,
        amount=10_000_000_000,
        pct_1d=0.06,
        ret_5d=0.12,
        ret_10d=0.10,
        ret_20d=0.18,
        ma5_gap=0.03,
        ma10_gap=0.05,
        ma20_gap=0.08,
        volatility_20d=0.04,
        amount_ratio_5d=1.6,
        turnover=0.05,
        ai_readiness="research_only",
        data_note="sample",
    )


def test_sector_theme_fallback_builds_top100_rows(monkeypatch) -> None:
    monkeypatch.setattr("gushen.tradingagents_dataset._fetch_external_sector_theme_map", lambda: {})
    monkeypatch.setattr("gushen.tradingagents_dataset._build_sector_theme_partial", lambda top100, market_technical, trade_date: [])

    rows = build_sector_themes([_bar()], [_feature()], "2026-05-20")

    assert len(rows) == 1
    assert rows[0].source_status == "fallback"
    assert rows[0].theme_heat_score > 50


def test_sector_theme_partial_uses_ths_strength(monkeypatch) -> None:
    monkeypatch.setattr("gushen.tradingagents_dataset._fetch_external_sector_theme_map", lambda: {})
    monkeypatch.setattr(
        "gushen.tradingagents_dataset._fetch_ths_sector_strength",
        lambda: {
            "\u534a\u5bfc\u4f53": {
                "sector_name": "\u534a\u5bfc\u4f53",
                "sector_rank": 3,
                "sector_pct_change": 4.2,
                "sector_main_net_inflow": 12.5,
                "theme_heat_score": 74.0,
            }
        },
    )
    monkeypatch.setattr("gushen.tradingagents_dataset._fetch_ths_concept_events", lambda: [])
    monkeypatch.setattr("gushen.tradingagents_dataset.load_or_build_stock_sector_map", lambda top100, trade_date: {})

    rows = build_sector_themes([_bar()], [_feature()], "2026-05-20")

    assert rows[0].source_status == "partial"
    assert rows[0].sector_name == "\u534a\u5bfc\u4f53"
    assert rows[0].theme_heat_score == 74.0


def test_sector_theme_partial_prefers_stock_sector_map(monkeypatch) -> None:
    from gushen.sector_mapping import StockSectorMapRow

    monkeypatch.setattr("gushen.tradingagents_dataset._fetch_external_sector_theme_map", lambda: {})
    monkeypatch.setattr(
        "gushen.tradingagents_dataset._fetch_ths_sector_strength",
        lambda: {
            "\u5143\u4ef6": {
                "sector_name": "\u5143\u4ef6",
                "sector_rank": 1,
                "sector_pct_change": 7.85,
                "sector_main_net_inflow": 165.32,
                "theme_heat_score": 100.0,
            },
            "\u534a\u5bfc\u4f53": {
                "sector_name": "\u534a\u5bfc\u4f53",
                "sector_rank": 12,
                "sector_pct_change": 2.92,
                "sector_main_net_inflow": 40.0,
                "theme_heat_score": 88.34,
            },
        },
    )
    monkeypatch.setattr("gushen.tradingagents_dataset._fetch_ths_concept_events", lambda: [])
    monkeypatch.setattr(
        "gushen.tradingagents_dataset.load_or_build_stock_sector_map",
        lambda top100, trade_date: {
            "603986": StockSectorMapRow(
                trade_date=trade_date,
                code="603986",
                name="\u82af\u7247\u6d4b\u8bd5",
                industry="\u5143\u4ef6",
                concepts="PCB\u6982\u5ff5;AI\u786c\u4ef6",
                source_status="ok",
                source="Sina industry constituents; Sina concept constituents",
                confidence=0.9,
                updated_at="2026-05-22T09:30:00",
                note="test",
            )
        },
    )

    rows = build_sector_themes([_bar()], [_feature()], "2026-05-20")

    assert rows[0].sector_name == "\u5143\u4ef6"
    assert rows[0].concept_names == "PCB\u6982\u5ff5;AI\u786c\u4ef6"
    assert "Sina industry constituents" in rows[0].source


def test_fund_flow_fallback_builds_top100_rows(monkeypatch) -> None:
    monkeypatch.setattr("gushen.tradingagents_dataset._fetch_external_fund_flow_map", lambda trade_date: {})
    monkeypatch.setattr("gushen.tradingagents_dataset._fetch_lhb_codes", lambda trade_date: {"603986"})

    rows = build_fund_flows([_bar()], [_feature()], "2026-05-20")

    assert len(rows) == 1
    assert rows[0].source_status == "fallback"
    assert rows[0].lhb_signal == "on_lhb"
    assert rows[0].flow_score > 50


def test_fund_flow_partial_uses_market_level_signals(monkeypatch) -> None:
    monkeypatch.setattr(
        "gushen.tradingagents_dataset._fetch_external_fund_flow_map",
        lambda trade_date: {
            "__MARKET__": {
                "northbound_signal": "northbound_net_buy",
                "margin_signal": "margin_loaded",
                "flow_score": 57,
                "partial_only": True,
            }
        },
    )
    monkeypatch.setattr("gushen.tradingagents_dataset._fetch_lhb_codes", lambda trade_date: set())

    rows = build_fund_flows([_bar()], [_feature()], "2026-05-20")

    assert rows[0].source_status == "partial"
    assert rows[0].northbound_signal == "northbound_net_buy"
    assert rows[0].margin_signal == "margin_loaded"


def test_market_flow_map_keeps_market_row_when_lhb_exists() -> None:
    from gushen.tradingagents_dataset import _build_market_flow_only_map

    result = _build_market_flow_only_map("northbound_flat", "margin_loaded", {"000001"})

    assert "__MARKET__" in result
    assert result["000001"]["partial_only"] is True
