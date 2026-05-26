from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from gushen.data import DailyBar
from gushen.deep_analysis import DeepFeatureRow, build_deep_features, load_or_fetch_histories
from gushen.domestic_network import domestic_data_no_proxy
from gushen.fund_flow_mapping import load_or_build_stock_fund_flow_map
from gushen.research import load_or_fetch_daily_snapshot
from gushen.sector_mapping import StockSectorMapRow, load_or_build_stock_sector_map
from gushen.trade_calendar import latest_research_trade_date


def _zh(key: str) -> str:
    values = {
        "code": "\u4ee3\u7801",
        "stock_code": "\u80a1\u7968\u4ee3\u7801",
        "security_code": "\u8bc1\u5238\u4ee3\u7801",
        "name": "\u7b80\u79f0",
        "title": "\u516c\u544a\u6807\u9898",
        "notice_time": "\u516c\u544a\u65f6\u95f4",
        "news_title": "\u65b0\u95fb\u6807\u9898",
        "publish_time": "\u53d1\u5e03\u65f6\u95f4",
        "news_source": "\u6587\u7ae0\u6765\u6e90",
        "pe_dynamic": "\u5e02\u76c8\u7387-\u52a8\u6001",
        "pb": "\u5e02\u51c0\u7387",
        "total_market_cap": "\u603b\u5e02\u503c",
        "circulating_market_cap": "\u6d41\u901a\u5e02\u503c",
    }
    return values[key]


@dataclass(frozen=True)
class RiskTradabilityRow:
    trade_date: str
    code: str
    name: str
    amount_rank: int
    is_st: bool
    is_suspended: bool
    is_limit_up: bool
    is_limit_down: bool
    tradability_note: str


@dataclass(frozen=True)
class FundamentalRow:
    trade_date: str
    code: str
    name: str
    amount_rank: int
    pe_dynamic: float | None
    pb: float | None
    total_market_cap: float | None
    circulating_market_cap: float | None
    source_status: str


@dataclass(frozen=True)
class EventRow:
    trade_date: str
    code: str
    name: str
    amount_rank: int
    event_status: str
    event_summary: str
    source: str = ""
    latest_time: str = ""


@dataclass(frozen=True)
class BacktestRow:
    trade_date: str
    code: str
    name: str
    amount_rank: int
    sample_count: int
    win_rate_1d: float | None
    avg_return_1d: float | None
    win_rate_3d: float | None
    avg_return_3d: float | None
    win_rate_5d: float | None
    avg_return_5d: float | None
    max_drawdown_5d: float | None
    source_status: str
    note: str


@dataclass(frozen=True)
class SectorThemeRow:
    trade_date: str
    code: str
    name: str
    amount_rank: int
    sector_name: str
    sector_rank: int | None
    sector_pct_change: float | None
    sector_main_net_inflow: float | None
    sector_main_net_pct: float | None
    concept_names: str
    theme_heat_score: float
    source_status: str
    source: str
    note: str


@dataclass(frozen=True)
class FundFlowRow:
    trade_date: str
    code: str
    name: str
    amount_rank: int
    main_net_inflow: float | None
    main_net_pct: float | None
    main_rank_today: int | None
    main_net_pct_5d: float | None
    main_rank_5d: int | None
    northbound_signal: str
    margin_signal: str
    lhb_signal: str
    flow_score: float
    source_status: str
    source: str
    note: str


@dataclass(frozen=True)
class TradingAgentsDataset:
    trade_date: str
    market_technical: list[DeepFeatureRow]
    risk_tradability: list[RiskTradabilityRow]
    fundamentals: list[FundamentalRow]
    events: list[EventRow]
    backtests: list[BacktestRow]
    sector_themes: list[SectorThemeRow]
    fund_flows: list[FundFlowRow]
    missing: list[str]


def build_tradingagents_dataset(trade_date: str | None = None) -> TradingAgentsDataset:
    console = Console()
    trade_date = trade_date or latest_research_trade_date()
    raw_date = trade_date.replace("-", "")
    snapshot = load_or_fetch_daily_snapshot(trade_date)
    top100 = sorted(snapshot, key=lambda item: item.amount, reverse=True)[:100]
    histories = load_or_fetch_histories(top100, trade_date)
    market_technical = build_deep_features(top100, histories, trade_date)
    risk_tradability = build_risk_tradability(top100, raw_date)
    fundamentals = build_fundamentals(top100, trade_date)
    events = build_events(top100, trade_date)
    backtests = build_backtests(top100, histories, trade_date)
    sector_themes = build_sector_themes(top100, market_technical, trade_date)
    fund_flows = build_fund_flows(top100, market_technical, trade_date)
    missing = [
        "full exchange announcement body extraction",
        "forum/social sentiment raw feed",
        "portfolio holdings and exposure",
    ]
    if any(row.source_status != "ok" for row in sector_themes):
        missing.append("external sector/theme feed is partial or fallback")
    if any(row.source_status != "ok" for row in fund_flows):
        missing.append("external fund-flow feed is partial or fallback")
    dataset = TradingAgentsDataset(
        trade_date=trade_date,
        market_technical=market_technical,
        risk_tradability=risk_tradability,
        fundamentals=fundamentals,
        events=events,
        backtests=backtests,
        sector_themes=sector_themes,
        fund_flows=fund_flows,
        missing=missing,
    )
    write_dataset(dataset)
    print_dataset_summary(console, dataset)
    return dataset


def build_risk_tradability(top100, raw_date: str) -> list[RiskTradabilityRow]:
    import akshare as ak

    with domestic_data_no_proxy():
        st_codes = _safe_code_set(lambda: ak.stock_zh_a_st_em())
        suspended_codes = _safe_code_set(lambda: ak.stock_tfp_em(date=raw_date))
        limit_up_codes = _safe_code_set(lambda: ak.stock_zt_pool_em(date=raw_date))
        limit_down_codes = _safe_code_set(lambda: ak.stock_zt_pool_dtgc_em(date=raw_date))
    rows = []
    for rank, bar in enumerate(top100, start=1):
        raw_code = bar.code.split(".")[0]
        flags = {
            "is_st": raw_code in st_codes,
            "is_suspended": raw_code in suspended_codes,
            "is_limit_up": raw_code in limit_up_codes,
            "is_limit_down": raw_code in limit_down_codes,
        }
        active_flags = [key for key, value in flags.items() if value]
        rows.append(
            RiskTradabilityRow(
                trade_date=bar.trade_date,
                code=bar.code,
                name=bar.name,
                amount_rank=rank,
                **flags,
                tradability_note=";".join(active_flags) if active_flags else "normal",
            )
        )
    return rows


def build_fundamentals(top100, trade_date: str) -> list[FundamentalRow]:
    spot_fallback = _fetch_spot_fundamentals()
    rows: list[FundamentalRow] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(_fetch_individual_info, bar.code): (rank, bar)
            for rank, bar in enumerate(top100, start=1)
        }
        for future in as_completed(futures):
            rank, bar = futures[future]
            try:
                info = future.result()
                status = "ok"
            except Exception:
                info = {}
                status = "failed"
            fallback = spot_fallback.get(bar.code.split(".")[0], {})
            if not any(_float_or_none(info.get(key)) is not None for key in info):
                valuation = _fetch_industry_valuation(bar.code)
                if valuation:
                    info.update(valuation)
                    status = "valuation_fallback" if status == "failed" else status
            pe_dynamic = _first_float("pe_dynamic", info, fallback)
            pb = _first_float("pb", info, fallback)
            total_market_cap = _first_float("total_market_cap", info, fallback)
            circulating_market_cap = _first_float("circulating_market_cap", info, fallback)
            if circulating_market_cap is None and bar.turnover > 0:
                circulating_market_cap = bar.amount / bar.turnover
            if fallback and status == "failed":
                status = "spot_fallback"
            if status == "ok" and not any(
                value is not None
                for value in [pe_dynamic, pb, total_market_cap, circulating_market_cap]
            ):
                status = "empty"
            rows.append(
                FundamentalRow(
                    trade_date=trade_date,
                    code=bar.code,
                    name=bar.name,
                    amount_rank=rank,
                    pe_dynamic=pe_dynamic,
                    pb=pb,
                    total_market_cap=total_market_cap,
                    circulating_market_cap=circulating_market_cap,
                    source_status=status,
                )
            )
    return sorted(rows, key=lambda item: item.amount_rank)


def build_events(top100, trade_date: str) -> list[EventRow]:
    rows: list[EventRow] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(_fetch_event_summary, bar.code, trade_date): (rank, bar)
            for rank, bar in enumerate(top100, start=1)
        }
        for future in as_completed(futures):
            rank, bar = futures[future]
            try:
                summary = future.result()
            except Exception as exc:
                summary = {
                    "status": "failed",
                    "summary": f"event fetch failed: {type(exc).__name__}",
                    "source": "akshare.stock_news_em",
                    "latest_time": "",
                }
            rows.append(
                EventRow(
                    trade_date=trade_date,
                    code=bar.code,
                    name=bar.name,
                    amount_rank=rank,
                    event_status=summary["status"],
                    event_summary=summary["summary"],
                    source=summary["source"],
                    latest_time=summary["latest_time"],
                )
            )
    return sorted(rows, key=lambda item: item.amount_rank)


def build_backtests(
    top100: list[DailyBar],
    histories: dict[str, list[DailyBar]],
    trade_date: str,
) -> list[BacktestRow]:
    rows: list[BacktestRow] = []
    amount_rank = {bar.code: index + 1 for index, bar in enumerate(top100)}
    for bar in top100:
        rows.append(_backtest_bar(bar, histories.get(bar.code, []), trade_date, amount_rank[bar.code]))
    return sorted(rows, key=lambda item: item.amount_rank)


def build_sector_themes(
    top100: list[DailyBar],
    market_technical: list[DeepFeatureRow],
    trade_date: str,
) -> list[SectorThemeRow]:
    external = _fetch_external_sector_theme_map()
    if external:
        rows = []
        for rank, bar in enumerate(top100, start=1):
            item = external.get(bar.code.split(".")[0], {})
            source_status = str(item.get("source_status") or ("ok" if item else "missing"))
            rows.append(
                SectorThemeRow(
                    trade_date=trade_date,
                    code=bar.code,
                    name=bar.name,
                    amount_rank=rank,
                    sector_name=str(item.get("sector_name") or ""),
                    sector_rank=_int_or_none(item.get("sector_rank")),
                    sector_pct_change=_float_or_none(item.get("sector_pct_change")),
                    sector_main_net_inflow=_float_or_none(item.get("sector_main_net_inflow")),
                    sector_main_net_pct=_float_or_none(item.get("sector_main_net_pct")),
                    concept_names=str(item.get("concept_names") or ""),
                    theme_heat_score=_float_or_none(item.get("theme_heat_score")) or 0.0,
                    source_status=source_status,
                    source=str(item.get("source") or "EastMoney sector/theme via AKShare"),
                    note=str(item.get("note") or "external sector/theme mapping"),
                )
            )
        return sorted(rows, key=lambda item: item.amount_rank)
    partial = _build_sector_theme_partial(top100, market_technical, trade_date)
    if partial:
        return partial
    return _build_sector_theme_fallback(top100, market_technical, trade_date)


def build_fund_flows(
    top100: list[DailyBar],
    market_technical: list[DeepFeatureRow],
    trade_date: str,
) -> list[FundFlowRow]:
    stock_flow = load_or_build_stock_fund_flow_map(top100, trade_date)
    external = _fetch_external_fund_flow_map(trade_date)
    lhb_codes = _fetch_lhb_codes(trade_date)
    if stock_flow:
        market_item = external.get("__MARKET__", {}) if external else {}
        rows = []
        for rank, bar in enumerate(top100, start=1):
            item = stock_flow.get(bar.code.split(".")[0])
            effective = item or None
            market_score = _float_or_none(market_item.get("flow_score")) or 50.0
            flow_score = _fund_flow_score_from_stock(effective, market_score)
            source_status = "ok" if effective and effective.source_status == "ok" else "partial"
            rows.append(
                FundFlowRow(
                    trade_date=trade_date,
                    code=bar.code,
                    name=bar.name,
                    amount_rank=rank,
                    main_net_inflow=effective.main_net_inflow if effective else None,
                    main_net_pct=effective.main_net_pct if effective else None,
                    main_rank_today=effective.main_rank_today if effective else None,
                    main_net_pct_5d=effective.main_net_pct_5d if effective else None,
                    main_rank_5d=effective.main_rank_5d if effective else None,
                    northbound_signal=str(market_item.get("northbound_signal") or "unknown"),
                    margin_signal=str(market_item.get("margin_signal") or "unknown"),
                    lhb_signal="on_lhb" if bar.code.split(".")[0] in lhb_codes else "not_on_lhb",
                    flow_score=round(flow_score, 2),
                    source_status=source_status,
                    source=(
                        "AKShare stock_individual_fund_flow; HSGT / margin / LHB via AKShare"
                        if source_status == "ok"
                        else "market-level HSGT / margin / LHB via AKShare"
                    ),
                    note=(
                        "exact trade-date stock-level fund-flow"
                        if source_status == "ok"
                        else "stock-level fund-flow unavailable for this stock; market-level signals only"
                    ),
                )
            )
        return sorted(rows, key=lambda item: item.amount_rank)
    if external:
        rows = []
        for rank, bar in enumerate(top100, start=1):
            item = external.get(bar.code.split(".")[0], {})
            market_item = external.get("__MARKET__", {})
            effective = item or market_item
            source_status = "ok" if item and not item.get("partial_only") else "partial" if effective else "missing"
            rows.append(
                FundFlowRow(
                    trade_date=trade_date,
                    code=bar.code,
                    name=bar.name,
                    amount_rank=rank,
                    main_net_inflow=_float_or_none(effective.get("main_net_inflow")),
                    main_net_pct=_float_or_none(effective.get("main_net_pct")),
                    main_rank_today=_int_or_none(effective.get("main_rank_today")),
                    main_net_pct_5d=_float_or_none(effective.get("main_net_pct_5d")),
                    main_rank_5d=_int_or_none(effective.get("main_rank_5d")),
                    northbound_signal=str(effective.get("northbound_signal") or "unknown"),
                    margin_signal=str(effective.get("margin_signal") or "unknown"),
                    lhb_signal="on_lhb" if bar.code.split(".")[0] in lhb_codes else "not_on_lhb",
                    flow_score=_float_or_none(effective.get("flow_score")) or 0.0,
                    source_status=source_status,
                    source="EastMoney fund flow / HSGT / margin / LHB via AKShare",
                    note="stock-level main fund-flow" if source_status == "ok" else "market-level fund-flow only; stock-level main flow unavailable",
                )
            )
        return sorted(rows, key=lambda item: item.amount_rank)
    return _build_fund_flow_fallback(top100, market_technical, trade_date, lhb_codes)


def write_dataset(dataset: TradingAgentsDataset) -> None:
    output_dir = Path(f"reports/generated/tradingagents_dataset_{dataset.trade_date}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "market_technical.csv", dataset.market_technical)
    _write_csv(output_dir / "risk_tradability.csv", dataset.risk_tradability)
    _write_csv(output_dir / "fundamentals.csv", dataset.fundamentals)
    _write_csv(output_dir / "events.csv", dataset.events)
    _write_csv(output_dir / "backtests.csv", dataset.backtests)
    _write_csv(output_dir / "sector_themes.csv", dataset.sector_themes)
    _write_csv(output_dir / "fund_flows.csv", dataset.fund_flows)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "trade_date": dataset.trade_date,
                "rows": {
                    "market_technical": len(dataset.market_technical),
                    "risk_tradability": len(dataset.risk_tradability),
                    "fundamentals": len(dataset.fundamentals),
                    "events": len(dataset.events),
                    "backtests": len(dataset.backtests),
                    "sector_themes": len(dataset.sector_themes),
                    "fund_flows": len(dataset.fund_flows),
                },
                "missing": dataset.missing,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def print_dataset_summary(console: Console, dataset: TradingAgentsDataset) -> None:
    table = Table(title=f"{dataset.trade_date} TradingAgents A-share dataset")
    table.add_column("Dataset")
    table.add_column("Rows", justify="right")
    table.add_row("market_technical", str(len(dataset.market_technical)))
    table.add_row("risk_tradability", str(len(dataset.risk_tradability)))
    table.add_row("fundamentals", str(len(dataset.fundamentals)))
    table.add_row("events", str(len(dataset.events)))
    table.add_row("backtests", str(len(dataset.backtests)))
    table.add_row("sector_themes", str(len(dataset.sector_themes)))
    table.add_row("fund_flows", str(len(dataset.fund_flows)))
    console.print(table)
    console.print("Missing: " + "; ".join(dataset.missing))


def _safe_code_set(fetcher) -> set[str]:
    try:
        frame = fetcher()
    except Exception:
        return set()
    if frame is None or frame.empty:
        return set()
    code_col = _find_code_column(frame.columns)
    if not code_col:
        return set()
    return {str(code).zfill(6) for code in frame[code_col].dropna().tolist()}


def _fetch_individual_info(code: str) -> dict[str, Any]:
    import akshare as ak

    raw_code = code.split(".")[0]
    with domestic_data_no_proxy():
        frame = ak.stock_individual_info_em(symbol=raw_code, timeout=12)
    if frame is None or frame.empty:
        return {}
    item_col = frame.columns[0]
    value_col = frame.columns[1]
    raw = {str(row[item_col]): row[value_col] for _, row in frame.iterrows()}
    return {
        "pe_dynamic": raw.get(_zh("pe_dynamic")),
        "pb": raw.get(_zh("pb")),
        "total_market_cap": raw.get(_zh("total_market_cap")),
        "circulating_market_cap": raw.get(_zh("circulating_market_cap")),
    }


def _fetch_event_summary(code: str, trade_date: str) -> dict[str, str]:
    import akshare as ak
    import pandas as pd

    raw_code = code.split(".")[0]
    news_frame = None
    notice_frame = None
    with domestic_data_no_proxy():
        try:
            news_frame = ak.stock_news_em(symbol=raw_code)
        except Exception:
            news_frame = None
        try:
            notice_frame = ak.stock_individual_notice_report(
                security=raw_code,
                begin_date=(date.fromisoformat(trade_date) - timedelta(days=45)).isoformat(),
                end_date=trade_date,
            )
        except Exception:
            notice_frame = None

    if (news_frame is None or news_frame.empty) and (notice_frame is None or notice_frame.empty):
        return {
            "status": "empty",
            "summary": "no recent EastMoney stock news or notices returned",
            "source": "akshare.stock_news_em;akshare.stock_individual_notice_report",
            "latest_time": "",
        }

    cutoff = date.fromisoformat(trade_date) - timedelta(days=30)
    items = []
    if news_frame is not None and not news_frame.empty:
        filtered = news_frame.copy()
        publish_time = _zh("publish_time")
        if publish_time in filtered.columns:
            times = pd.to_datetime(filtered[publish_time], errors="coerce")
            filtered = filtered[times.dt.date >= cutoff]
        for _, row in filtered.head(4).iterrows():
            title = str(row.get(_zh("news_title"), "")).strip()
            time_text = str(row.get(publish_time, "")).strip()
            source = str(row.get(_zh("news_source"), "")).strip()
            if title:
                items.append(("news", time_text, source, title))
    if notice_frame is not None and not notice_frame.empty:
        filtered = notice_frame.copy()
        notice_time = _zh("notice_time")
        if notice_time in filtered.columns:
            times = pd.to_datetime(filtered[notice_time], errors="coerce")
            filtered = filtered[times.dt.date >= cutoff]
        for _, row in filtered.head(4).iterrows():
            title = str(row.get(_zh("title"), "")).strip()
            time_text = str(row.get(notice_time, "")).strip()
            if title:
                items.append(("notice", time_text, "EastMoney notice", title))

    if not items:
        return {
            "status": "empty_recent",
            "summary": "no recent news/notices in 30-day window",
            "source": "akshare.stock_news_em;akshare.stock_individual_notice_report",
            "latest_time": "",
        }

    items = sorted(items, key=lambda item: item[1], reverse=True)
    titles = []
    latest_time = ""
    for kind, time_text, source, title in items[:6]:
        titles.append(f"{kind} {time_text} {source} {title}".strip())
        if not latest_time and time_text:
            latest_time = time_text
    return {
        "status": "loaded",
        "summary": " | ".join(titles),
        "source": "akshare.stock_news_em;akshare.stock_individual_notice_report",
        "latest_time": latest_time,
    }


def _find_code_column(columns) -> str | None:
    for column in columns:
        if str(column) in {_zh("code"), _zh("stock_code"), _zh("security_code")}:
            return column
    return None


def _fetch_spot_fundamentals() -> dict[str, dict[str, Any]]:
    import akshare as ak

    try:
        with domestic_data_no_proxy():
            frame = ak.stock_zh_a_spot_em()
    except Exception:
        return {}
    if frame is None or frame.empty:
        return {}
    mapping = {
        _zh("code"): "code",
        _zh("pe_dynamic"): "pe_dynamic",
        _zh("pb"): "pb",
        _zh("total_market_cap"): "total_market_cap",
        _zh("circulating_market_cap"): "circulating_market_cap",
    }
    result: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        code = str(row.get(_zh("code"), "")).zfill(6)
        if not code:
            continue
        result[code] = {
            target: row.get(source)
            for source, target in mapping.items()
            if target != "code" and source in frame.columns
        }
    return result


def _fetch_external_sector_theme_map() -> dict[str, dict[str, Any]]:
    try:
        import akshare as ak

        with domestic_data_no_proxy():
            industry = ak.stock_board_industry_name_em()
            concept = ak.stock_board_concept_name_em()
    except Exception:
        return {}
    board_frames = [
        ("industry", industry, "stock_board_industry_cons_em"),
        ("concept", concept, "stock_board_concept_cons_em"),
    ]
    if all(frame is None or frame.empty for _, frame, _ in board_frames):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for board_type, frame, cons_func_name in board_frames:
        if frame is None or frame.empty:
            continue
        name_col = _find_first_column(frame.columns, ["板块名称", "名称"])
        code_col = _find_first_column(frame.columns, ["板块代码"])
        rank_col = _find_first_column(frame.columns, ["排名", "序号"])
        pct_col = _find_first_column(frame.columns, ["涨跌幅"])
        if not name_col:
            continue
        strong_boards = sorted(
            [row for _, row in frame.iterrows()],
            key=lambda row: _float_or_none(row.get(pct_col)) or 0.0,
            reverse=True,
        )[:20]
        for board in strong_boards:
            board_name = str(board.get(name_col, "")).strip()
            board_code = str(board.get(code_col, "")).strip() if code_col else ""
            if not board_name and not board_code:
                continue
            members = _fetch_board_members(cons_func_name, board_code or board_name)
            board_pct = _float_or_none(board.get(pct_col)) if pct_col else None
            board_rank = _int_or_none(board.get(rank_col)) if rank_col else None
            for member_code in members:
                current = result.setdefault(
                    member_code,
                    {
                        "sector_name": "",
                        "sector_rank": None,
                        "sector_pct_change": None,
                        "sector_main_net_inflow": None,
                        "sector_main_net_pct": None,
                        "concept_names": "",
                        "theme_heat_score": 0.0,
                    },
                )
                heat = _bounded_score((board_pct or 0) * 8 + (20 - min(board_rank or 20, 20)) * 1.5)
                if heat > (_float_or_none(current.get("theme_heat_score")) or 0):
                    current["theme_heat_score"] = heat
                if board_type == "industry" and not current.get("sector_name"):
                    current["sector_name"] = board_name
                    current["sector_rank"] = board_rank
                    current["sector_pct_change"] = board_pct
                elif board_type == "concept":
                    concepts = [item for item in str(current.get("concept_names") or "").split(";") if item]
                    if board_name not in concepts:
                        concepts.append(board_name)
                    current["concept_names"] = ";".join(concepts[:5])
                    current["source_status"] = "ok"
                    current["source"] = "EastMoney sector/theme via AKShare"
                    current["note"] = "external sector/theme constituent mapping"
    return result


def _fetch_ths_sector_strength() -> dict[str, dict[str, Any]]:
    try:
        import akshare as ak

        with domestic_data_no_proxy():
            summary = ak.stock_board_industry_summary_ths()
    except Exception:
        summary = None
    if summary is None or summary.empty:
        try:
            import akshare as ak

            with domestic_data_no_proxy():
                summary = ak.stock_fund_flow_industry(symbol=_u("\u5373\u65f6"))
        except Exception:
            return {}
    if summary is None or summary.empty:
        return {}
    name_col = _find_first_column(summary.columns, [_u("\u677f\u5757"), _u("\u884c\u4e1a")])
    rank_col = _find_first_column(summary.columns, [_u("\u5e8f\u53f7"), _u("\u6392\u540d")])
    pct_col = _find_first_column(summary.columns, [_u("\u6da8\u8dcc\u5e45"), _u("\u884c\u4e1a-\u6da8\u8dcc\u5e45")])
    net_col = _find_first_column(summary.columns, [_u("\u51c0\u6d41\u5165"), _u("\u51c0\u989d")])
    if not name_col:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for _, row in summary.iterrows():
        name = str(row.get(name_col, "")).strip()
        if not name:
            continue
        rank = _int_or_none(row.get(rank_col)) if rank_col else None
        pct = _float_or_none(row.get(pct_col)) if pct_col else None
        net = _float_or_none(row.get(net_col)) if net_col else None
        result[name] = {
            "sector_name": name,
            "sector_rank": rank,
            "sector_pct_change": pct,
            "sector_main_net_inflow": net,
            "sector_main_net_pct": None,
            "theme_heat_score": _bounded_score((pct or 0) * 6 + (net or 0) * 0.03 + (90 - min(rank or 90, 90)) * 0.25),
        }
    return result


def _fetch_ths_concept_events() -> list[str]:
    try:
        import akshare as ak

        with domestic_data_no_proxy():
            frame = ak.stock_board_concept_summary_ths()
    except Exception:
        return []
    if frame is None or frame.empty:
        return []
    name_col = _find_first_column(frame.columns, [_u("\u6982\u5ff5\u540d\u79f0"), _u("\u6982\u5ff5"), _u("\u884c\u4e1a")])
    if not name_col:
        return []
    return [str(value).strip() for value in frame[name_col].dropna().tolist() if str(value).strip()]


def _build_sector_theme_partial(
    top100: list[DailyBar],
    market_technical: list[DeepFeatureRow],
    trade_date: str,
) -> list[SectorThemeRow]:
    strengths = _fetch_ths_sector_strength()
    if not strengths:
        return []
    concepts = _fetch_ths_concept_events()
    stock_sector_map = load_or_build_stock_sector_map(top100, trade_date)
    feature_map = {row.code: row for row in market_technical}
    rows: list[SectorThemeRow] = []
    for rank, bar in enumerate(top100, start=1):
        feature = feature_map.get(bar.code)
        inferred = _infer_sector_from_name(bar.name)
        mapping = stock_sector_map.get(bar.code.split(".")[0])
        mapped_sector = _mapped_sector(mapping)
        matched_name = _match_sector_name(inferred, bar.name, strengths, mapped_sector)
        item = strengths.get(matched_name or "", {})
        sector_name = str(item.get("sector_name") or mapped_sector or inferred)
        heat = _float_or_none(item.get("theme_heat_score"))
        if heat is None:
            local_heat = 0.0
            if feature:
                local_heat = feature.ret_5d * 90 + feature.ret_20d * 35 + feature.amount_ratio_5d * 5
            heat = _bounded_score(local_heat)
        map_concepts = _mapped_concepts(mapping)
        if not map_concepts:
            map_concepts = _match_concepts(bar.name, concepts)
        source_parts = ["THS sector strength/fund-flow summary via AKShare"]
        if mapping and mapping.source_status == "ok":
            source_parts.append(mapping.source)
        source_status = "partial"
        note = "real THS board strength loaded; stock-sector mapping is cached but still cross-source partial"
        if not mapping or mapping.source_status != "ok":
            note = "real THS board strength loaded, but full stock-to-sector constituents are not wired yet"
        rows.append(
            SectorThemeRow(
                trade_date=trade_date,
                code=bar.code,
                name=bar.name,
                amount_rank=rank,
                sector_name=sector_name,
                sector_rank=_int_or_none(item.get("sector_rank")),
                sector_pct_change=_float_or_none(item.get("sector_pct_change")),
                sector_main_net_inflow=_float_or_none(item.get("sector_main_net_inflow")),
                sector_main_net_pct=_float_or_none(item.get("sector_main_net_pct")),
                concept_names=";".join(map_concepts[:8]),
                theme_heat_score=round(heat, 2),
                source_status=source_status,
                source="; ".join(part for part in source_parts if part),
                note=note,
            )
        )
    return sorted(rows, key=lambda item: item.amount_rank)


def _fetch_board_members(cons_func_name: str, board_symbol: str) -> set[str]:
    try:
        import akshare as ak

        func = getattr(ak, cons_func_name)
        with domestic_data_no_proxy():
            frame = func(symbol=board_symbol)
    except Exception:
        return set()
    if frame is None or frame.empty:
        return set()
    code_col = _find_first_column(frame.columns, ["代码", "证券代码"])
    if not code_col:
        return set()
    return {str(value).zfill(6) for value in frame[code_col].dropna().tolist()}


def _fetch_external_fund_flow_map(trade_date: str) -> dict[str, dict[str, Any]]:
    hsgt_signal = _fetch_northbound_signal()
    margin_signal = _fetch_margin_signal(trade_date)
    lhb_codes = _fetch_lhb_codes(trade_date)
    try:
        import akshare as ak

        with domestic_data_no_proxy():
            frame = ak.stock_main_fund_flow(symbol=_u("全部股票"))
    except Exception:
        return _build_market_flow_only_map(hsgt_signal, margin_signal, lhb_codes)
    if frame is None or frame.empty:
        return _build_market_flow_only_map(hsgt_signal, margin_signal, lhb_codes)
    result: dict[str, dict[str, Any]] = {}
    code_col = _find_first_column(frame.columns, ["代码", "code"])
    net_pct_col = _find_first_column(frame.columns, ["今日排行榜-主力净占比"])
    rank_col = _find_first_column(frame.columns, ["今日排行榜-今日排名"])
    net_pct_5d_col = _find_first_column(frame.columns, ["5日排行榜-主力净占比"])
    rank_5d_col = _find_first_column(frame.columns, ["5日排行榜-5日排名"])
    if not code_col:
        return {}
    for _, row in frame.iterrows():
        code = str(row.get(code_col, "")).zfill(6)
        if not code:
            continue
        net_pct = _float_or_none(row.get(net_pct_col)) if net_pct_col else None
        net_pct_5d = _float_or_none(row.get(net_pct_5d_col)) if net_pct_5d_col else None
        result[code] = {
            "main_net_inflow": None,
            "main_net_pct": net_pct,
            "main_rank_today": row.get(rank_col) if rank_col else None,
            "main_net_pct_5d": net_pct_5d,
            "main_rank_5d": row.get(rank_5d_col) if rank_5d_col else None,
            "northbound_signal": hsgt_signal,
            "margin_signal": margin_signal,
            "flow_score": _bounded_score((net_pct or 0) * 5 + (net_pct_5d or 0) * 3),
        }
    return result


def _build_market_flow_only_map(
    hsgt_signal: str,
    margin_signal: str,
    lhb_codes: set[str],
) -> dict[str, dict[str, Any]]:
    result = {}
    base_score = 50.0
    if hsgt_signal == "northbound_net_buy":
        base_score += 5
    elif hsgt_signal == "northbound_net_sell":
        base_score -= 5
    if margin_signal == "margin_loaded":
        base_score += 2
    for code in lhb_codes:
        result[code] = {
            "main_net_inflow": None,
            "main_net_pct": None,
            "main_rank_today": None,
            "main_net_pct_5d": None,
            "main_rank_5d": None,
            "northbound_signal": hsgt_signal,
            "margin_signal": margin_signal,
            "flow_score": _bounded_score(base_score - 50 + 8),
            "partial_only": True,
        }
    if hsgt_signal != "unknown" or margin_signal != "unknown" or result:
        result["__MARKET__"] = {
            "northbound_signal": hsgt_signal,
            "margin_signal": margin_signal,
            "flow_score": _bounded_score(base_score - 50),
            "partial_only": True,
        }
    return result


def _fund_flow_score_from_stock(item, market_score: float) -> float:
    base = market_score - 50.0
    if not item or item.source_status != "ok":
        return _bounded_score(base)
    net_pct = _float_or_none(item.main_net_pct) or 0.0
    net_pct_5d = _float_or_none(item.main_net_pct_5d) or 0.0
    rank_bonus = 0.0
    if item.main_rank_today:
        rank_bonus += max(0.0, 12.0 - min(item.main_rank_today, 60) * 0.2)
    if item.main_rank_5d:
        rank_bonus += max(0.0, 8.0 - min(item.main_rank_5d, 60) * 0.13)
    return _bounded_score(base + net_pct * 3.2 + net_pct_5d * 2.2 + rank_bonus)


def _fetch_lhb_codes(trade_date: str) -> set[str]:
    try:
        import akshare as ak

        raw_date = trade_date.replace("-", "")
        with domestic_data_no_proxy():
            frame = ak.stock_lhb_detail_em(start_date=raw_date, end_date=raw_date)
    except Exception:
        return set()
    if frame is None or frame.empty:
        return set()
    code_col = _find_first_column(frame.columns, ["代码", "证券代码"])
    if not code_col:
        return set()
    return {str(value).zfill(6) for value in frame[code_col].dropna().tolist()}


def _fetch_northbound_signal() -> str:
    try:
        import akshare as ak

        with domestic_data_no_proxy():
            frame = ak.stock_hsgt_fund_flow_summary_em()
    except Exception:
        return "unknown"
    if frame is None or frame.empty:
        return "unknown"
    direction_col = _find_first_column(frame.columns, ["资金方向"])
    net_col = _find_first_column(frame.columns, ["成交净买额", "资金净流入"])
    if not direction_col or not net_col:
        return "unknown"
    north_rows = frame[frame[direction_col].astype(str).str.contains(_u("北向"), na=False)]
    total = sum(_float_or_none(value) or 0.0 for value in north_rows[net_col].tolist())
    if total > 0:
        return "northbound_net_buy"
    if total < 0:
        return "northbound_net_sell"
    return "northbound_flat"


def _fetch_margin_signal(trade_date: str) -> str:
    raw_date = trade_date.replace("-", "")
    try:
        import akshare as ak

        with domestic_data_no_proxy():
            sse = ak.stock_margin_detail_sse(date=raw_date)
            szse = ak.stock_margin_detail_szse(date=raw_date)
    except Exception:
        return "unknown"
    rows = 0
    for frame in [sse, szse]:
        if frame is not None and not frame.empty:
            rows += len(frame)
    return "margin_loaded" if rows else "unknown"


def _build_sector_theme_fallback(
    top100: list[DailyBar],
    market_technical: list[DeepFeatureRow],
    trade_date: str,
) -> list[SectorThemeRow]:
    feature_map = {row.code: row for row in market_technical}
    ranked = sorted(
        market_technical,
        key=lambda row: row.ret_5d * 60 + row.ret_20d * 25 + row.amount_ratio_5d * 5,
        reverse=True,
    )
    heat_rank = {row.code: index + 1 for index, row in enumerate(ranked)}
    rows = []
    for rank, bar in enumerate(top100, start=1):
        feature = feature_map.get(bar.code)
        heat = 0.0
        if feature:
            heat = _bounded_score(feature.ret_5d * 120 + feature.ret_20d * 60 + feature.amount_ratio_5d * 8)
        rows.append(
            SectorThemeRow(
                trade_date=trade_date,
                code=bar.code,
                name=bar.name,
                amount_rank=rank,
                sector_name=_infer_sector_from_name(bar.name),
                sector_rank=heat_rank.get(bar.code),
                sector_pct_change=feature.ret_5d if feature else None,
                sector_main_net_inflow=None,
                sector_main_net_pct=None,
                concept_names="",
                theme_heat_score=round(heat, 2),
                source_status="fallback",
                source="local Top100 momentum/amount proxy",
                note="external EastMoney sector/theme feed unavailable; using local proxy only",
            )
        )
    return sorted(rows, key=lambda item: item.amount_rank)


def _build_fund_flow_fallback(
    top100: list[DailyBar],
    market_technical: list[DeepFeatureRow],
    trade_date: str,
    lhb_codes: set[str],
) -> list[FundFlowRow]:
    feature_map = {row.code: row for row in market_technical}
    rows = []
    for rank, bar in enumerate(top100, start=1):
        feature = feature_map.get(bar.code)
        proxy = 0.0
        if feature:
            proxy = feature.pct_1d * 140 + (feature.amount_ratio_5d - 1) * 20 + feature.ret_5d * 50
        rows.append(
            FundFlowRow(
                trade_date=trade_date,
                code=bar.code,
                name=bar.name,
                amount_rank=rank,
                main_net_inflow=None,
                main_net_pct=None,
                main_rank_today=None,
                main_net_pct_5d=None,
                main_rank_5d=None,
                northbound_signal="unknown",
                margin_signal="unknown",
                lhb_signal="on_lhb" if bar.code.split(".")[0] in lhb_codes else "not_on_lhb",
                flow_score=round(_bounded_score(proxy), 2),
                source_status="fallback",
                source="local price-volume proxy plus LHB when available",
                note="external EastMoney fund-flow feed unavailable; using local proxy only",
            )
        )
    return sorted(rows, key=lambda item: item.amount_rank)


def _fetch_industry_valuation(code: str) -> dict[str, Any]:
    import requests

    raw_code = code.split(".")[0]
    market = code.split(".")[1] if "." in code else "SH" if raw_code.startswith("6") else "SZ"
    url = "https://datacenter.eastmoney.com/securities/api/data/v1/get"
    params = {
        "reportName": "RPT_PCF10_INDUSTRY_CVALUE",
        "columns": "ALL",
        "quoteColumns": "",
        "filter": f'(SECUCODE="{raw_code}.{market}")',
        "pageNumber": "",
        "pageSize": "",
        "sortTypes": "1",
        "sortColumns": "PAIMING",
        "source": "HSF10",
        "client": "PC",
    }
    try:
        with domestic_data_no_proxy():
            response = requests.get(url, params=params, timeout=12)
        response.raise_for_status()
        rows = response.json().get("result", {}).get("data", [])
    except Exception:
        return {}
    for row in rows:
        if str(row.get("CORRE_SECURITY_CODE", "")).zfill(6) == raw_code:
            return {
                "pe_dynamic": row.get("PE_TTM") or row.get("PE"),
                "pb": row.get("PB_MRQ") or row.get("PB"),
            }
    return {}


def _first_float(key: str, *sources: dict[str, Any]) -> float | None:
    for source in sources:
        value = _float_or_none(source.get(key))
        if value is not None:
            return value
    return None


def _find_first_column(columns, candidates: list[str]) -> str | None:
    lookup = {str(column): column for column in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    for column in columns:
        text = str(column)
        if any(candidate in text for candidate in candidates):
            return column
    return None


def _int_or_none(value) -> int | None:
    try:
        if value in {None, "-", ""}:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _bounded_score(raw: float) -> float:
    return max(0.0, min(100.0, 50.0 + raw))


def _u(text: str) -> str:
    return text


def _mapped_sector(mapping: StockSectorMapRow | None) -> str:
    if mapping and mapping.source_status == "ok" and mapping.industry:
        return mapping.industry
    return ""


def _mapped_concepts(mapping: StockSectorMapRow | None) -> list[str]:
    if not mapping or not mapping.concepts:
        return []
    return [item for item in mapping.concepts.split(";") if item]


def _match_sector_name(
    inferred: str,
    stock_name: str,
    strengths: dict[str, dict[str, Any]],
    mapped_sector: str = "",
) -> str | None:
    if mapped_sector in strengths:
        return mapped_sector
    known = _known_stock_sector().get(stock_name)
    if known and known in strengths:
        return known
    if inferred in strengths:
        return inferred
    candidates = [mapped_sector] if mapped_sector else []
    candidates.extend(_sector_aliases().get(inferred, []))
    ranked: list[tuple[int, str]] = []
    for name in strengths:
        score = 0
        if name in stock_name or stock_name in name:
            score += 5
        for alias in candidates:
            if alias and (alias in name or name in alias or alias in stock_name):
                score += 3
        if score:
            ranked.append((score, name))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], _float_or_none(strengths[item[1]].get("theme_heat_score")) or 0), reverse=True)
    return ranked[0][1]


def _match_concepts(stock_name: str, concepts: list[str]) -> list[str]:
    matches = []
    for concept in concepts:
        if any(token and token in stock_name for token in _concept_tokens(concept)):
            matches.append(concept)
    return matches


def _concept_tokens(concept: str) -> list[str]:
    stopwords = [
        _u("\u6982\u5ff5"),
        _u("\u6307\u6570"),
        _u("\u4e3b\u9898"),
        _u("\u5e74\u62a5"),
        _u("\u5b63\u62a5"),
        "AI",
        "A",
    ]
    text = concept
    for word in stopwords:
        text = text.replace(word, "")
    return [token for token in [text, text[:2], text[-2:]] if len(token) >= 2]


def _sector_aliases() -> dict[str, list[str]]:
    return {
        _u("\u79d1\u6280\u786c\u4ef6"): [
            _u("\u534a\u5bfc\u4f53"),
            _u("\u5143\u4ef6"),
            _u("\u7535\u5b50"),
            _u("\u901a\u4fe1\u8bbe\u5907"),
            _u("\u8f6f\u4ef6"),
            _u("\u8ba1\u7b97\u673a"),
        ],
        _u("\u65b0\u80fd\u6e90"): [
            _u("\u7535\u6c60"),
            _u("\u5149\u4f0f\u8bbe\u5907"),
            _u("\u7535\u529b\u8bbe\u5907"),
            _u("\u80fd\u6e90\u91d1\u5c5e"),
        ],
        _u("\u91d1\u878d\u5730\u4ea7"): [
            _u("\u94f6\u884c"),
            _u("\u8bc1\u5238"),
            _u("\u4fdd\u9669"),
            _u("\u623f\u5730\u4ea7"),
        ],
        _u("\u533b\u836f\u6d88\u8d39"): [
            _u("\u5316\u5b66\u5236\u836f"),
            _u("\u4e2d\u836f"),
            _u("\u533b\u7597\u5668\u68b0"),
            _u("\u767d\u9152"),
            _u("\u98df\u54c1"),
        ],
        _u("\u6709\u8272\u5316\u5de5"): [
            _u("\u6709\u8272\u91d1\u5c5e"),
            _u("\u5c0f\u91d1\u5c5e"),
            _u("\u5316\u5b66\u539f\u6599"),
            _u("\u5316\u5b66\u5236\u54c1"),
            _u("\u5de5\u4e1a\u91d1\u5c5e"),
        ],
        _u("\u6c7d\u8f66\u673a\u68b0"): [
            _u("\u6c7d\u8f66\u96f6\u90e8\u4ef6"),
            _u("\u6c7d\u8f66\u6574\u8f66"),
            _u("\u4e13\u7528\u8bbe\u5907"),
            _u("\u901a\u7528\u8bbe\u5907"),
            _u("\u5de5\u7a0b\u673a\u68b0"),
        ],
    }


def _known_stock_sector() -> dict[str, str]:
    return {
        _u("\u6df1\u79d1\u6280"): _u("\u5143\u4ef6"),
        _u("\u5146\u6613\u521b\u65b0"): _u("\u534a\u5bfc\u4f53"),
        _u("\u5bd2\u6b66\u7eaa"): _u("\u534a\u5bfc\u4f53"),
        _u("\u4e2d\u9645\u65ed\u521b"): _u("\u901a\u4fe1\u8bbe\u5907"),
        _u("\u65b0\u6613\u76db"): _u("\u901a\u4fe1\u8bbe\u5907"),
        _u("\u6f9c\u8d77\u79d1\u6280"): _u("\u534a\u5bfc\u4f53"),
        _u("\u957f\u7535\u79d1\u6280"): _u("\u534a\u5bfc\u4f53"),
        _u("\u6d77\u5149\u4fe1\u606f"): _u("\u534a\u5bfc\u4f53"),
        _u("\u5de5\u4e1a\u5bcc\u8054"): _u("\u6d88\u8d39\u7535\u5b50"),
        _u("\u80dc\u5b8f\u79d1\u6280"): _u("\u5143\u4ef6"),
        _u("\u6caa\u7535\u80a1\u4efd"): _u("\u5143\u4ef6"),
        _u("\u4e1c\u5c71\u7cbe\u5bc6"): _u("\u5143\u4ef6"),
        _u("\u4e2d\u82af\u56fd\u9645"): _u("\u534a\u5bfc\u4f53"),
        _u("\u5317\u65b9\u534e\u521b"): _u("\u534a\u5bfc\u4f53"),
        _u("\u4e2d\u5fae\u516c\u53f8"): _u("\u534a\u5bfc\u4f53"),
        _u("\u7acb\u8baf\u7cbe\u5bc6"): _u("\u6d88\u8d39\u7535\u5b50"),
        _u("\u4eac\u4e1c\u65b9A"): _u("\u5149\u5b66\u5149\u7535"),
        _u("\u6bd4\u4e9a\u8fea"): _u("\u6c7d\u8f66\u6574\u8f66"),
        _u("\u8d5b\u529b\u65af"): _u("\u6c7d\u8f66\u6574\u8f66"),
        _u("\u9633\u5149\u7535\u6e90"): _u("\u5149\u4f0f\u8bbe\u5907"),
        _u("\u5b81\u5fb7\u65f6\u4ee3"): _u("\u7535\u6c60"),
        _u("\u4e94\u7cae\u6db2"): _u("\u767d\u9152"),
        _u("\u8d35\u5dde\u8305\u53f0"): _u("\u767d\u9152"),
    }


def _infer_sector_from_name(name: str) -> str:
    rules = [
        (_u("\u79d1\u6280\u786c\u4ef6"), [_u("\u82af"), _u("\u5fae"), _u("\u7535\u5b50"), _u("\u5149"), _u("\u79d1\u6280"), _u("\u901a\u4fe1"), _u("\u534a\u5bfc\u4f53"), _u("\u4fe1\u606f")]),
        (_u("\u65b0\u80fd\u6e90"), [_u("\u7535\u6c60"), _u("\u80fd\u6e90"), _u("\u5149\u4f0f"), _u("\u9502"), _u("\u9633\u5149"), _u("\u50a8\u80fd")]),
        (_u("\u91d1\u878d\u5730\u4ea7"), [_u("\u94f6\u884c"), _u("\u8bc1\u5238"), _u("\u4fdd\u9669"), _u("\u5730\u4ea7")]),
        (_u("\u533b\u836f\u6d88\u8d39"), [_u("\u836f"), _u("\u533b"), _u("\u751f\u7269"), _u("\u98df\u54c1"), _u("\u9152")]),
        (_u("\u6709\u8272\u5316\u5de5"), [_u("\u94dc"), _u("\u94dd"), _u("\u77ff"), _u("\u5316\u5de5"), _u("\u6750\u6599")]),
        (_u("\u6c7d\u8f66\u673a\u68b0"), [_u("\u8f66"), _u("\u6c7d"), _u("\u673a"), _u("\u91cd\u5de5"), _u("\u88c5\u5907")]),
    ]
    for sector, keywords in rules:
        if any(keyword in name for keyword in keywords):
            return sector
    return _u("\u672a\u5206\u7c7b")


def _backtest_bar(
    bar: DailyBar,
    history: list[DailyBar],
    trade_date: str,
    amount_rank: int,
) -> BacktestRow:
    bars = sorted(
        (item for item in history if item.trade_date <= trade_date),
        key=lambda item: item.trade_date,
    )
    if len(bars) < 40:
        return BacktestRow(
            trade_date=trade_date,
            code=bar.code,
            name=bar.name,
            amount_rank=amount_rank,
            sample_count=max(0, len(bars) - 25),
            win_rate_1d=None,
            avg_return_1d=None,
            win_rate_3d=None,
            avg_return_3d=None,
            win_rate_5d=None,
            avg_return_5d=None,
            max_drawdown_5d=None,
            source_status="insufficient_history",
            note="need at least 40 historical bars for local signal backtest",
        )

    samples: list[dict[str, float]] = []
    closes = [item.close for item in bars]
    amounts = [item.amount for item in bars]
    for index in range(20, len(bars) - 5):
        sample = bars[index]
        previous_close = closes[index - 1]
        ma5 = sum(closes[index - 4 : index + 1]) / 5
        ma20 = sum(closes[index - 19 : index + 1]) / 20
        amount_avg5 = sum(amounts[index - 5 : index]) / 5
        signal = (
            previous_close > 0
            and sample.close / previous_close - 1 > 0
            and sample.close >= ma5 >= ma20
            and amount_avg5 > 0
            and sample.amount / amount_avg5 >= 1.2
            and sample.pct_change < 0.095
        )
        if not signal:
            continue
        returns = {
            "ret_1d": bars[index + 1].close / sample.close - 1,
            "ret_3d": bars[index + 3].close / sample.close - 1,
            "ret_5d": bars[index + 5].close / sample.close - 1,
            "drawdown_5d": min(item.low for item in bars[index + 1 : index + 6]) / sample.close - 1,
        }
        samples.append(returns)

    if len(samples) < 3:
        return BacktestRow(
            trade_date=trade_date,
            code=bar.code,
            name=bar.name,
            amount_rank=amount_rank,
            sample_count=len(samples),
            win_rate_1d=None,
            avg_return_1d=None,
            win_rate_3d=None,
            avg_return_3d=None,
            win_rate_5d=None,
            avg_return_5d=None,
            max_drawdown_5d=None,
            source_status="too_few_signals",
            note="local momentum-volume signal had fewer than 3 historical samples",
        )

    return BacktestRow(
        trade_date=trade_date,
        code=bar.code,
        name=bar.name,
        amount_rank=amount_rank,
        sample_count=len(samples),
        win_rate_1d=_win_rate(samples, "ret_1d"),
        avg_return_1d=_mean_key(samples, "ret_1d"),
        win_rate_3d=_win_rate(samples, "ret_3d"),
        avg_return_3d=_mean_key(samples, "ret_3d"),
        win_rate_5d=_win_rate(samples, "ret_5d"),
        avg_return_5d=_mean_key(samples, "ret_5d"),
        max_drawdown_5d=min(item["drawdown_5d"] for item in samples),
        source_status="ok",
        note="historical local momentum-volume signal; not an execution-grade backtest",
    )


def _write_csv(path: Path, rows: list) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _float_or_none(value) -> float | None:
    try:
        if value in {None, "-", ""}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_key(samples: list[dict[str, float]], key: str) -> float:
    return sum(item[key] for item in samples) / len(samples)


def _win_rate(samples: list[dict[str, float]], key: str) -> float:
    return sum(1 for item in samples if item[key] > 0) / len(samples)


def main() -> None:
    build_tradingagents_dataset()


if __name__ == "__main__":
    main()
