from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from rich.console import Console
from rich.table import Table

from gushen.data import _to_ts_code
from gushen.domestic_network import domestic_data_no_proxy


@dataclass(frozen=True)
class IntradayTopAmountRow:
    captured_at: str
    source_trade_date: str
    source_time: str
    amount_rank: int
    code: str
    name: str
    latest_price: float | None
    pct_change: float | None
    price_change: float | None
    volume: float | None
    amount: float | None
    amplitude: float | None
    turnover: float | None
    high: float | None
    low: float | None
    open: float | None
    prev_close: float | None


def fetch_intraday_top_amount(limit: int = 100) -> tuple[list[IntradayTopAmountRow], dict[str, Any]]:
    captured_at = datetime.now().isoformat(timespec="seconds")
    payload = _fetch_eastmoney_intraday_payload(limit)
    rows = payload.get("data", {}).get("diff", []) if payload.get("data") else []
    result = [
        _parse_row(row, index, captured_at)
        for index, row in enumerate(rows[:limit], start=1)
    ]
    meta = {
        "captured_at": captured_at,
        "source": "EastMoney push2 clist intraday top amount",
        "rows": len(result),
        "raw_data_keys": list((payload.get("data") or {}).keys()),
    }
    return result, meta


def save_intraday_top_amount(
    limit: int = 100,
    output_root: Path = Path("data/local/intraday_top_amount"),
) -> tuple[Path | None, Path, list[IntradayTopAmountRow], dict[str, Any]]:
    captured = datetime.now()
    try:
        rows, meta = fetch_intraday_top_amount(limit)
        captured = datetime.fromisoformat(meta["captured_at"])
        error: dict[str, str] | None = None
    except Exception as exc:
        rows = []
        meta = {
            "captured_at": captured.isoformat(timespec="seconds"),
            "source": "EastMoney push2 clist intraday top amount",
            "rows": 0,
            "status": "failed",
        }
        error = {"type": type(exc).__name__, "message": str(exc)}
    output_dir = output_root / captured.strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"top_amount_{captured.strftime('%H%M%S')}"
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    if rows:
        _write_rows(csv_path, rows)
    else:
        csv_path = None
    json_path.write_text(
        json.dumps(
            {"meta": meta, "error": error, "rows": [asdict(row) for row in rows]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return csv_path, json_path, rows, meta


def _fetch_eastmoney_intraday_payload(limit: int) -> dict[str, Any]:
    url = "https://82.push2.eastmoney.com/api/qt/clist/get"
    params: dict[str, Any] = {
        "pn": "1",
        "pz": str(max(limit, 30)),
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f6",
        "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
        "fields": "f12,f14,f2,f3,f4,f5,f6,f7,f8,f15,f16,f17,f18,f124,f297",
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "application/json,text/plain,*/*",
    }
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with domestic_data_no_proxy():
                response = requests.get(url, params=params, headers=headers, timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    if last_error:
        raise last_error
    raise RuntimeError("intraday top amount request failed")


def _parse_row(row: dict[str, Any], rank: int, captured_at: str) -> IntradayTopAmountRow:
    source_ts = _int_or_none(row.get("f124"))
    source_trade_date = str(row.get("f297") or "")
    return IntradayTopAmountRow(
        captured_at=captured_at,
        source_trade_date=source_trade_date,
        source_time=_source_time(source_ts),
        amount_rank=rank,
        code=_to_ts_code(str(row.get("f12") or "")),
        name=str(row.get("f14") or ""),
        latest_price=_float_or_none(row.get("f2")),
        pct_change=_float_or_none(row.get("f3")),
        price_change=_float_or_none(row.get("f4")),
        volume=_float_or_none(row.get("f5")),
        amount=_float_or_none(row.get("f6")),
        amplitude=_float_or_none(row.get("f7")),
        turnover=_float_or_none(row.get("f8")),
        high=_float_or_none(row.get("f15")),
        low=_float_or_none(row.get("f16")),
        open=_float_or_none(row.get("f17")),
        prev_close=_float_or_none(row.get("f18")),
    )


def _source_time(source_ts: int | None) -> str:
    if not source_ts:
        return ""
    return datetime.fromtimestamp(source_ts).isoformat(timespec="seconds")


def _write_rows(path: Path, rows: list[IntradayTopAmountRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(IntradayTopAmountRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _float_or_none(value: Any) -> float | None:
    try:
        if value in {None, "", "-"}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value in {None, "", "-"}:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def main() -> None:
    console = Console()
    csv_path, json_path, rows, meta = save_intraday_top_amount()
    table = Table(title="Intraday top amount snapshot")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("captured_at", str(meta["captured_at"]))
    table.add_row("status", str(meta.get("status", "ok")))
    table.add_row("rows", str(meta["rows"]))
    table.add_row("csv", str(csv_path or ""))
    table.add_row("json", str(json_path))
    if rows:
        top = rows[0]
        table.add_row("top1", f"{top.code} {top.name} amount={top.amount} source_time={top.source_time}")
    console.print(table)


if __name__ == "__main__":
    main()
