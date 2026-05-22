from __future__ import annotations

import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from gushen.data import DailyBar
from gushen.domestic_network import domestic_data_no_proxy
from gushen.research import load_or_fetch_daily_snapshot


MIN_USABLE_COVERAGE = 0.3


@dataclass(frozen=True)
class StockFundFlowMapRow:
    trade_date: str
    code: str
    name: str
    main_net_inflow: float | None
    main_net_pct: float | None
    main_rank_today: int | None
    main_net_inflow_5d: float | None
    main_net_pct_5d: float | None
    main_rank_5d: int | None
    source_status: str
    source: str
    confidence: float
    updated_at: str
    note: str


def load_or_build_stock_fund_flow_map(
    top100: list[DailyBar],
    trade_date: str,
    cache_dir: Path = Path("data/local/fund_flows"),
) -> dict[str, StockFundFlowMapRow]:
    cache_path = cache_dir / f"stock_fund_flow_map_{trade_date}.csv"
    cached = _read_cache(cache_path)
    target_codes = {bar.code.split(".")[0] for bar in top100}
    cached_target = {code: row for code, row in cached.items() if code in target_codes}
    cached_coverage = _coverage(cached_target.values())
    if target_codes and target_codes.issubset(cached) and cached_coverage >= MIN_USABLE_COVERAGE:
        return {code: row for code, row in cached.items() if code in target_codes}
    rows = build_stock_fund_flow_map(top100, trade_date)
    new_coverage = _coverage(rows)
    if rows and new_coverage >= max(MIN_USABLE_COVERAGE, cached_coverage):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _write_cache(cache_path, rows)
        return {row.code: row for row in rows}
    if cached_coverage >= MIN_USABLE_COVERAGE:
        return cached_target
    return {}


def build_stock_fund_flow_map(top100: list[DailyBar], trade_date: str) -> list[StockFundFlowMapRow]:
    if not top100:
        return []
    updated_at = datetime.now().isoformat(timespec="seconds")
    partial: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_fetch_one_stock_flow, bar, trade_date): bar
            for bar in top100
        }
        for future in as_completed(futures):
            bar = futures[future]
            try:
                item = future.result()
            except Exception as exc:
                item = {
                    "code": bar.code.split(".")[0],
                    "name": bar.name,
                    "main_net_inflow": None,
                    "main_net_pct": None,
                    "main_net_inflow_5d": None,
                    "main_net_pct_5d": None,
                    "source_status": "failed",
                    "note": f"stock individual fund-flow failed: {type(exc).__name__}",
                }
            partial.append(item)
    ok_today = [
        item for item in partial
        if item.get("source_status") == "ok" and item.get("main_net_pct") is not None
    ]
    ok_5d = [
        item for item in partial
        if item.get("source_status") == "ok" and item.get("main_net_pct_5d") is not None
    ]
    today_rank = {
        str(item["code"]): index + 1
        for index, item in enumerate(
            sorted(ok_today, key=lambda row: _none_low(row.get("main_net_pct")), reverse=True)
        )
    }
    rank_5d = {
        str(item["code"]): index + 1
        for index, item in enumerate(
            sorted(ok_5d, key=lambda row: _none_low(row.get("main_net_pct_5d")), reverse=True)
        )
    }
    rows = []
    for item in partial:
        status = str(item.get("source_status") or "missing")
        rows.append(
            StockFundFlowMapRow(
                trade_date=trade_date,
                code=str(item.get("code") or ""),
                name=str(item.get("name") or ""),
                main_net_inflow=_float_or_none(item.get("main_net_inflow")),
                main_net_pct=_float_or_none(item.get("main_net_pct")),
                main_rank_today=today_rank.get(str(item.get("code") or "")),
                main_net_inflow_5d=_float_or_none(item.get("main_net_inflow_5d")),
                main_net_pct_5d=_float_or_none(item.get("main_net_pct_5d")),
                main_rank_5d=rank_5d.get(str(item.get("code") or "")),
                source_status=status,
                source="AKShare stock_individual_fund_flow" if status == "ok" else "",
                confidence=0.88 if status == "ok" else 0.0,
                updated_at=updated_at,
                note=str(item.get("note") or ""),
            )
        )
    return sorted(rows, key=lambda row: row.code)


def _fetch_one_stock_flow(bar: DailyBar, trade_date: str) -> dict[str, object]:
    import akshare as ak
    import pandas as pd

    raw_code = bar.code.split(".")[0]
    market = _market(raw_code, bar.code)
    last_error = ""
    frame = None
    for attempt in range(3):
        try:
            with domestic_data_no_proxy():
                frame = ak.stock_individual_fund_flow(stock=raw_code, market=market)
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.6 * (attempt + 1))
    if frame is None:
        raise RuntimeError(last_error or "stock individual fund-flow request failed")
    if frame is None or frame.empty:
        return _missing(bar, "empty individual fund-flow frame")
    date_col = _find_column(frame.columns, ["\u65e5\u671f"])
    net_col = _find_column(frame.columns, ["\u4e3b\u529b\u51c0\u6d41\u5165-\u51c0\u989d"])
    pct_col = _find_column(frame.columns, ["\u4e3b\u529b\u51c0\u6d41\u5165-\u51c0\u5360\u6bd4"])
    if not date_col or not net_col or not pct_col:
        return _missing(bar, "required individual fund-flow columns are missing")
    rows = frame.copy()
    rows[date_col] = pd.to_datetime(rows[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    rows = rows[rows[date_col].notna()]
    rows = rows[rows[date_col] <= trade_date].sort_values(date_col)
    if rows.empty:
        return _missing(bar, "no fund-flow rows on or before trade date")
    exact = rows[rows[date_col] == trade_date]
    if exact.empty:
        return _missing(bar, "no exact fund-flow row for trade date")
    target = exact.iloc[-1]
    recent = rows.tail(5)
    pct_values = [_float_or_none(value) for value in recent[pct_col].tolist()]
    net_values = [_float_or_none(value) for value in recent[net_col].tolist()]
    pct_values = [value for value in pct_values if value is not None]
    net_values = [value for value in net_values if value is not None]
    return {
        "code": raw_code,
        "name": bar.name,
        "main_net_inflow": _float_or_none(target.get(net_col)),
        "main_net_pct": _float_or_none(target.get(pct_col)),
        "main_net_inflow_5d": sum(net_values) if net_values else None,
        "main_net_pct_5d": sum(pct_values) / len(pct_values) if pct_values else None,
        "source_status": "ok",
        "note": "exact trade-date individual fund-flow row loaded",
    }


def _missing(bar: DailyBar, note: str) -> dict[str, object]:
    return {
        "code": bar.code.split(".")[0],
        "name": bar.name,
        "main_net_inflow": None,
        "main_net_pct": None,
        "main_net_inflow_5d": None,
        "main_net_pct_5d": None,
        "source_status": "missing",
        "note": note,
    }


def _market(raw_code: str, code: str) -> str:
    if "." in code:
        return code.split(".")[1].lower()
    return "sh" if raw_code.startswith(("6", "9")) else "sz"


def _read_cache(path: Path) -> dict[str, StockFundFlowMapRow]:
    if not path.exists():
        return {}
    result = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            for key in [
                "main_net_inflow",
                "main_net_pct",
                "main_net_inflow_5d",
                "main_net_pct_5d",
                "confidence",
            ]:
                row[key] = _float_or_none(row.get(key))
            for key in ["main_rank_today", "main_rank_5d"]:
                row[key] = _int_or_none(row.get(key))
            result[row["code"]] = StockFundFlowMapRow(**row)
    return result


def _write_cache(path: Path, rows: Iterable[StockFundFlowMapRow]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(StockFundFlowMapRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _find_column(columns, candidates: list[str]) -> str | None:
    lookup = {str(column): column for column in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    for column in columns:
        text = str(column)
        if any(candidate in text for candidate in candidates):
            return column
    return None


def _float_or_none(value) -> float | None:
    try:
        if value in {None, "-", ""}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value) -> int | None:
    try:
        if value in {None, "-", ""}:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _none_low(value) -> float:
    parsed = _float_or_none(value)
    return parsed if parsed is not None else -999999.0


def _coverage(rows: Iterable[StockFundFlowMapRow]) -> float:
    rows = list(rows)
    if not rows:
        return 0.0
    return sum(row.source_status == "ok" for row in rows) / len(rows)


def main() -> None:
    from collections import Counter

    from rich.console import Console
    from rich.table import Table

    trade_date = "2026-05-20"
    snapshot = load_or_fetch_daily_snapshot(trade_date)
    top100 = sorted(snapshot, key=lambda item: item.amount, reverse=True)[:100]
    rows = list(load_or_build_stock_fund_flow_map(top100, trade_date).values())
    status = Counter(row.source_status for row in rows)
    table = Table(title=f"{trade_date} Top100 stock fund-flow map")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("rows", str(len(rows)))
    table.add_row("ok", str(status.get("ok", 0)))
    table.add_row("failed", str(status.get("failed", 0)))
    table.add_row("missing", str(status.get("missing", 0)))
    table.add_row("today pct coverage", f"{sum(row.main_net_pct is not None for row in rows)}/{len(rows)}")
    table.add_row("5d pct coverage", f"{sum(row.main_net_pct_5d is not None for row in rows)}/{len(rows)}")
    console = Console()
    console.print(table)
    console.print(f"Cache: data/local/fund_flows/stock_fund_flow_map_{trade_date}.csv")


if __name__ == "__main__":
    main()
