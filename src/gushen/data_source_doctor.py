from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable

from rich.console import Console
from rich.table import Table

from gushen.domestic_network import domestic_data_no_proxy


@dataclass(frozen=True)
class DataSourceCheck:
    name: str
    category: str
    provider: str
    endpoint: str
    purpose: str
    domestic_direct: bool
    required_for_gate: bool
    runner: Callable[[], int]


@dataclass(frozen=True)
class DataSourceStatus:
    checked_at: str
    name: str
    category: str
    provider: str
    endpoint: str
    purpose: str
    domestic_direct: bool
    required_for_gate: bool
    status: str
    rows: int
    seconds: float
    error_type: str
    error: str


def run_data_source_doctor() -> list[DataSourceStatus]:
    checks = build_checks()
    checked_at = datetime.now().isoformat(timespec="seconds")
    statuses = [_run_check(check, checked_at) for check in checks]
    write_report(statuses)
    return statuses


def build_checks() -> list[DataSourceCheck]:
    import akshare as ak

    return [
        DataSourceCheck(
            "a_share_code_name",
            "base",
            "AKShare",
            "stock_info_a_code_name",
            "A-share code/name universe",
            True,
            True,
            lambda: len(ak.stock_info_a_code_name()),
        ),
        DataSourceCheck(
            "daily_history",
            "market",
            "AKShare/EastMoney",
            "stock_zh_a_hist",
            "single-stock daily OHLCV history",
            True,
            True,
            lambda: len(
                ak.stock_zh_a_hist(
                    symbol="000001",
                    period="daily",
                    start_date="20260101",
                    end_date="20260521",
                    adjust="qfq",
                )
            ),
        ),
        DataSourceCheck(
            "st_pool",
            "tradability",
            "AKShare/EastMoney",
            "stock_zh_a_st_em",
            "ST/risk-warning pool",
            True,
            True,
            lambda: len(ak.stock_zh_a_st_em()),
        ),
        DataSourceCheck(
            "industry_board",
            "sector_theme",
            "AKShare/EastMoney",
            "stock_board_industry_name_em",
            "industry board strength",
            True,
            False,
            lambda: len(ak.stock_board_industry_name_em()),
        ),
        DataSourceCheck(
            "concept_board",
            "sector_theme",
            "AKShare/EastMoney",
            "stock_board_concept_name_em",
            "concept board strength",
            True,
            False,
            lambda: len(ak.stock_board_concept_name_em()),
        ),
        DataSourceCheck(
            "main_fund_flow",
            "fund_flow",
            "AKShare/EastMoney",
            "stock_main_fund_flow",
            "individual main fund-flow ranking",
            True,
            False,
            lambda: len(ak.stock_main_fund_flow(symbol="全部股票")),
        ),
        DataSourceCheck(
            "hsgt_flow",
            "fund_flow",
            "AKShare/EastMoney",
            "stock_hsgt_fund_flow_summary_em",
            "northbound/southbound market fund flow",
            True,
            False,
            lambda: len(ak.stock_hsgt_fund_flow_summary_em()),
        ),
        DataSourceCheck(
            "dragon_tiger",
            "fund_flow",
            "AKShare/EastMoney",
            "stock_lhb_detail_em",
            "dragon-tiger list detail for target date",
            True,
            False,
            lambda: len(ak.stock_lhb_detail_em(start_date="20260520", end_date="20260520")),
        ),
        DataSourceCheck(
            "margin_sse",
            "fund_flow",
            "AKShare/SSE",
            "stock_margin_detail_sse",
            "SSE margin trading detail",
            True,
            False,
            lambda: len(ak.stock_margin_detail_sse(date="20260520")),
        ),
        DataSourceCheck(
            "margin_szse",
            "fund_flow",
            "AKShare/SZSE",
            "stock_margin_detail_szse",
            "SZSE margin trading detail",
            True,
            False,
            lambda: len(ak.stock_margin_detail_szse(date="20260520")),
        ),
    ]


def write_report(statuses: list[DataSourceStatus]) -> None:
    output_dir = Path("reports/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = [asdict(status) for status in statuses]
    (output_dir / "data_source_doctor_latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "data_source_doctor_latest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(DataSourceStatus.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(payload)


def _run_check(check: DataSourceCheck, checked_at: str) -> DataSourceStatus:
    started = perf_counter()
    rows = 0
    status = "ok"
    error_type = ""
    error = ""
    try:
        if check.domestic_direct:
            with domestic_data_no_proxy():
                rows = check.runner()
        else:
            rows = check.runner()
        if rows <= 0:
            status = "empty"
    except Exception as exc:
        status = "failed"
        error_type = type(exc).__name__
        error = str(exc)[:500]
    seconds = perf_counter() - started
    return DataSourceStatus(
        checked_at=checked_at,
        name=check.name,
        category=check.category,
        provider=check.provider,
        endpoint=check.endpoint,
        purpose=check.purpose,
        domestic_direct=check.domestic_direct,
        required_for_gate=check.required_for_gate,
        status=status,
        rows=rows,
        seconds=round(seconds, 3),
        error_type=error_type,
        error=error,
    )


def print_report(statuses: list[DataSourceStatus]) -> None:
    table = Table(title="A-share data source doctor")
    table.add_column("Name")
    table.add_column("Category")
    table.add_column("Endpoint")
    table.add_column("Status")
    table.add_column("Rows", justify="right")
    table.add_column("Seconds", justify="right")
    for status in statuses:
        label = status.status if not status.error_type else f"{status.status}: {status.error_type}"
        table.add_row(
            status.name,
            status.category,
            status.endpoint,
            label,
            str(status.rows),
            f"{status.seconds:.2f}",
        )
    console = Console()
    console.print(table)
    failed = [status for status in statuses if status.status != "ok"]
    if failed:
        console.print("Failed/empty: " + "; ".join(f"{item.name}={item.error_type or item.status}" for item in failed))


def main() -> None:
    statuses = run_data_source_doctor()
    print_report(statuses)


if __name__ == "__main__":
    main()
