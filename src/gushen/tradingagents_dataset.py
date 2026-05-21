from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from gushen.deep_analysis import DeepFeatureRow, build_deep_features, load_or_fetch_histories
from gushen.research import load_or_fetch_daily_snapshot


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


@dataclass(frozen=True)
class TradingAgentsDataset:
    trade_date: str
    market_technical: list[DeepFeatureRow]
    risk_tradability: list[RiskTradabilityRow]
    fundamentals: list[FundamentalRow]
    events: list[EventRow]
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
    missing = [
        "full exchange announcement body extraction",
        "news/sentiment dataset",
        "fund flow / financing / dragon-tiger dataset",
        "strategy backtest outcomes",
        "portfolio holdings and exposure",
    ]
    dataset = TradingAgentsDataset(
        trade_date=trade_date,
        market_technical=market_technical,
        risk_tradability=risk_tradability,
        fundamentals=fundamentals,
        events=events,
        missing=missing,
    )
    write_dataset(dataset)
    print_dataset_summary(console, dataset)
    return dataset


def build_risk_tradability(top100, raw_date: str) -> list[RiskTradabilityRow]:
    import akshare as ak

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
            rows.append(
                FundamentalRow(
                    trade_date=trade_date,
                    code=bar.code,
                    name=bar.name,
                    amount_rank=rank,
                    pe_dynamic=_float_or_none(info.get("市盈率-动态")),
                    pb=_float_or_none(info.get("市净率")),
                    total_market_cap=_float_or_none(info.get("总市值")),
                    circulating_market_cap=_float_or_none(info.get("流通市值")),
                    source_status=status,
                )
            )
    return sorted(rows, key=lambda item: item.amount_rank)


def build_events(top100, trade_date: str) -> list[EventRow]:
    return [
        EventRow(
            trade_date=trade_date,
            code=bar.code,
            name=bar.name,
            amount_rank=rank,
            event_status="not_loaded",
            event_summary="announcement/news extraction not implemented yet",
        )
        for rank, bar in enumerate(top100, start=1)
    ]


def write_dataset(dataset: TradingAgentsDataset) -> None:
    output_dir = Path(f"reports/generated/tradingagents_dataset_{dataset.trade_date}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "market_technical.csv", dataset.market_technical)
    _write_csv(output_dir / "risk_tradability.csv", dataset.risk_tradability)
    _write_csv(output_dir / "fundamentals.csv", dataset.fundamentals)
    _write_csv(output_dir / "events.csv", dataset.events)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "trade_date": dataset.trade_date,
                "rows": {
                    "market_technical": len(dataset.market_technical),
                    "risk_tradability": len(dataset.risk_tradability),
                    "fundamentals": len(dataset.fundamentals),
                    "events": len(dataset.events),
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
    frame = ak.stock_individual_info_em(symbol=raw_code, timeout=12)
    if frame is None or frame.empty:
        return {}
    item_col = frame.columns[0]
    value_col = frame.columns[1]
    return {str(row[item_col]): row[value_col] for _, row in frame.iterrows()}


def _find_code_column(columns) -> str | None:
    for column in columns:
        if str(column) in {"代码", "股票代码", "证券代码"}:
            return column
    return None


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


def main() -> None:
    build_tradingagents_dataset()


if __name__ == "__main__":
    main()
