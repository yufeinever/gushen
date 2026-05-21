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
from gushen.research import load_or_fetch_daily_snapshot


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


def build_tradingagents_dataset(trade_date: str = "2026-05-20") -> TradingAgentsDataset:
    console = Console()
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
                    source_status="ok" if item else "missing",
                    source="EastMoney sector/theme via AKShare",
                    note="external sector/theme mapping",
                )
            )
        return sorted(rows, key=lambda item: item.amount_rank)
    return _build_sector_theme_fallback(top100, market_technical, trade_date)


def build_fund_flows(
    top100: list[DailyBar],
    market_technical: list[DeepFeatureRow],
    trade_date: str,
) -> list[FundFlowRow]:
    external = _fetch_external_fund_flow_map(trade_date)
    lhb_codes = _fetch_lhb_codes(trade_date)
    if external:
        rows = []
        for rank, bar in enumerate(top100, start=1):
            item = external.get(bar.code.split(".")[0], {})
            rows.append(
                FundFlowRow(
                    trade_date=trade_date,
                    code=bar.code,
                    name=bar.name,
                    amount_rank=rank,
                    main_net_inflow=_float_or_none(item.get("main_net_inflow")),
                    main_net_pct=_float_or_none(item.get("main_net_pct")),
                    main_rank_today=_int_or_none(item.get("main_rank_today")),
                    main_net_pct_5d=_float_or_none(item.get("main_net_pct_5d")),
                    main_rank_5d=_int_or_none(item.get("main_rank_5d")),
                    northbound_signal=str(item.get("northbound_signal") or "unknown"),
                    margin_signal=str(item.get("margin_signal") or "unknown"),
                    lhb_signal="on_lhb" if bar.code.split(".")[0] in lhb_codes else "not_on_lhb",
                    flow_score=_float_or_none(item.get("flow_score")) or 0.0,
                    source_status="ok" if item else "missing",
                    source="EastMoney fund flow / LHB via AKShare",
                    note="external fund-flow mapping",
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
            frame = ak.stock_main_fund_flow(symbol=_u("全部股票"))
    except Exception:
        return {}
    if frame is None or frame.empty:
        return {}
    result: dict[str, dict[str, Any]] = {}
    code_col = _find_first_column(frame.columns, ["代码", "code"])
    name_col = _find_first_column(frame.columns, ["所属板块", "板块", "行业"])
    rank_col = _find_first_column(frame.columns, ["今日排行榜-今日排名", "排名", "序号"])
    pct_col = _find_first_column(frame.columns, ["今日排行榜-今日涨跌", "涨跌幅"])
    net_pct_col = _find_first_column(frame.columns, ["今日排行榜-主力净占比", "主力净占比"])
    if not code_col:
        return {}
    for _, row in frame.iterrows():
        code = str(row.get(code_col, "")).zfill(6)
        if not code:
            continue
        net_pct = _float_or_none(row.get(net_pct_col)) if net_pct_col else None
        pct_change = _float_or_none(row.get(pct_col)) if pct_col else None
        result[code] = {
            "sector_name": row.get(name_col) if name_col else "",
            "sector_rank": row.get(rank_col) if rank_col else None,
            "sector_pct_change": pct_change,
            "sector_main_net_inflow": None,
            "sector_main_net_pct": net_pct,
            "concept_names": "",
            "theme_heat_score": _bounded_score((net_pct or 0) * 3 + (pct_change or 0) * 2),
        }
    return result


def _fetch_external_fund_flow_map(trade_date: str) -> dict[str, dict[str, Any]]:
    try:
        import akshare as ak

        with domestic_data_no_proxy():
            frame = ak.stock_main_fund_flow(symbol=_u("全部股票"))
    except Exception:
        return {}
    if frame is None or frame.empty:
        return {}
    result: dict[str, dict[str, Any]] = {}
    code_col = _find_first_column(frame.columns, ["代码", "code"])
    net_pct_col = _find_first_column(frame.columns, ["今日排行榜-主力净占比"])
    rank_col = _find_first_column(frame.columns, ["今日排行榜-今日排名"])
    net_pct_5d_col = _find_first_column(frame.columns, ["5日排行榜-主力净占比"])
    rank_5d_col = _find_first_column(frame.columns, ["5日排行榜-5日排名"])
    if not code_col:
        return {}
    hsgt_signal = _fetch_northbound_signal()
    margin_signal = _fetch_margin_signal(trade_date)
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


def _infer_sector_from_name(name: str) -> str:
    rules = [
        ("科技硬件", ["芯", "微", "电子", "光", "科技", "通信", "半导体", "信息"]),
        ("新能源", ["电池", "能源", "光伏", "锂", "阳光", "储能"]),
        ("金融地产", ["银行", "证券", "保险", "地产"]),
        ("医药消费", ["药", "医", "生物", "食品", "酒"]),
        ("有色化工", ["铜", "铝", "矿", "化工", "材料"]),
        ("汽车机械", ["车", "汽", "机", "重工", "装备"]),
    ]
    for sector, keywords in rules:
        if any(keyword in name for keyword in keywords):
            return sector
    return "未分类"


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
