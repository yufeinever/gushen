from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from rich.console import Console
from rich.table import Table


@dataclass(frozen=True)
class Check:
    name: str
    description: str
    runner: Callable[[], int]


def main() -> None:
    console = Console()
    try:
        import akshare as ak
    except ImportError:
        console.print("[red]AKShare is not installed. Run: pip install -e .[data][/]")
        raise SystemExit(2)

    checks = [
        Check(
            "stock_info_a_code_name",
            "A-share code/name list",
            lambda: len(ak.stock_info_a_code_name()),
        ),
        Check(
            "tool_trade_date_hist_sina",
            "trade calendar",
            lambda: len(ak.tool_trade_date_hist_sina()),
        ),
        Check(
            "stock_zh_a_st_em",
            "risk warning/ST pool",
            lambda: len(ak.stock_zh_a_st_em()),
        ),
        Check(
            "stock_zh_a_spot_em",
            "A-share realtime spot table",
            lambda: len(ak.stock_zh_a_spot_em()),
        ),
        Check(
            "stock_zh_a_hist",
            "single-stock daily history",
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
    ]

    table = Table(title="AKShare data interface health check")
    table.add_column("Function")
    table.add_column("Purpose")
    table.add_column("Status")
    table.add_column("Rows", justify="right")
    table.add_column("Seconds", justify="right")

    failed = 0
    for check in checks:
        started = perf_counter()
        try:
            rows = check.runner()
            status = "ok"
        except Exception as exc:
            rows = 0
            status = f"failed: {type(exc).__name__}: {exc}"
            failed += 1
        seconds = perf_counter() - started
        table.add_row(check.name, check.description, status, str(rows), f"{seconds:.2f}")

    console.print(table)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
