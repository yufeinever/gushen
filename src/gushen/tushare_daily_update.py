from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from gushen.domestic_network import domestic_data_no_proxy
from gushen.trade_calendar import latest_research_trade_date

DEFAULT_OUTPUT_ROOT = Path("data/local/tushare_market")
DAILY_COLUMNS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
]
ADJ_FACTOR_COLUMNS = ["ts_code", "trade_date", "adj_factor"]


@dataclass(frozen=True)
class TushareDailyUpdateResult:
    trade_date: str
    daily_rows: int
    adj_factor_rows: int
    daily_path: str
    adj_factor_path: str
    manifest_path: str
    dry_run: bool
    source: str
    started_at: str
    finished_at: str
    note: str


def normalize_trade_date(value: str) -> str:
    raw = value.strip().replace("-", "")
    if len(raw) != 8 or not raw.isdigit():
        raise ValueError(f"trade date must be YYYYMMDD or YYYY-MM-DD: {value}")
    return raw


def display_trade_date(value: str) -> str:
    raw = normalize_trade_date(value)
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"


def fetch_tushare_daily_by_trade_date(pro: Any, trade_date: str) -> pd.DataFrame:
    frame = pro.daily(trade_date=normalize_trade_date(trade_date))
    return normalize_daily_frame(frame, trade_date)


def fetch_tushare_adj_factor_by_trade_date(pro: Any, trade_date: str) -> pd.DataFrame:
    frame = pro.adj_factor(trade_date=normalize_trade_date(trade_date))
    return normalize_adj_factor_frame(frame, trade_date)


def normalize_daily_frame(frame: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    return _normalize_frame(frame, DAILY_COLUMNS, trade_date)


def normalize_adj_factor_frame(frame: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    return _normalize_frame(frame, ADJ_FACTOR_COLUMNS, trade_date)


def _normalize_frame(frame: pd.DataFrame, columns: list[str], trade_date: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Tushare response missing columns: {missing}")
    normalized = frame[columns].copy()
    normalized["trade_date"] = normalized["trade_date"].astype(str).map(normalize_trade_date)
    normalized = normalized[normalized["trade_date"] == normalize_trade_date(trade_date)]
    normalized = normalized.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    normalized = normalized.sort_values("ts_code").reset_index(drop=True)
    return normalized


def get_tushare_pro(token: str | None = None) -> Any:
    try:
        import tushare as ts
    except ImportError as exc:
        raise RuntimeError("Tushare is required. Install with: pip install tushare") from exc
    token = token or os.environ.get("TUSHARE_TOKEN")
    if token:
        ts.set_token(token)
    return ts.pro_api(token) if token else ts.pro_api()


def write_frame(path: Path, frame: pd.DataFrame, fmt: str = "csv") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        frame.to_parquet(path, index=False)
    elif fmt == "csv":
        frame.to_csv(path, index=False)
    else:
        raise ValueError(f"unsupported output format: {fmt}")


def build_output_paths(output_root: Path, trade_date: str, fmt: str = "csv") -> tuple[Path, Path, Path]:
    raw = normalize_trade_date(trade_date)
    suffix = "parquet" if fmt == "parquet" else "csv"
    daily_path = output_root / "daily_by_date" / f"trade_date={raw}.{suffix}"
    adj_path = output_root / "adj_factor_by_date" / f"trade_date={raw}.{suffix}"
    manifest_path = output_root / "manifests" / f"trade_date={raw}.json"
    return daily_path, adj_path, manifest_path


def update_tushare_daily_market(
    trade_date: str | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    fmt: str = "csv",
    token: str | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    pro: Any | None = None,
) -> TushareDailyUpdateResult:
    resolved_trade_date = normalize_trade_date(
        trade_date or latest_research_trade_date().replace("-", "")
    )
    started_at = datetime.now().isoformat(timespec="seconds")
    daily_path, adj_path, manifest_path = build_output_paths(output_root, resolved_trade_date, fmt)
    if not overwrite and daily_path.exists() and adj_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return TushareDailyUpdateResult(
            trade_date=display_trade_date(resolved_trade_date),
            daily_rows=int(manifest.get("daily_rows", 0)),
            adj_factor_rows=int(manifest.get("adj_factor_rows", 0)),
            daily_path=str(daily_path),
            adj_factor_path=str(adj_path),
            manifest_path=str(manifest_path),
            dry_run=dry_run,
            source="tushare.pro",
            started_at=started_at,
            finished_at=datetime.now().isoformat(timespec="seconds"),
            note="cached",
        )
    if dry_run:
        return TushareDailyUpdateResult(
            trade_date=display_trade_date(resolved_trade_date),
            daily_rows=0,
            adj_factor_rows=0,
            daily_path=str(daily_path),
            adj_factor_path=str(adj_path),
            manifest_path=str(manifest_path),
            dry_run=True,
            source="tushare.pro",
            started_at=started_at,
            finished_at=datetime.now().isoformat(timespec="seconds"),
            note="dry run; no network call or file write",
        )
    pro = pro or get_tushare_pro(token)
    with domestic_data_no_proxy():
        daily = fetch_tushare_daily_by_trade_date(pro, resolved_trade_date)
        adj_factor = fetch_tushare_adj_factor_by_trade_date(pro, resolved_trade_date)
    if daily.empty:
        raise RuntimeError(f"Tushare daily returned no rows for {resolved_trade_date}")
    if adj_factor.empty:
        raise RuntimeError(f"Tushare adj_factor returned no rows for {resolved_trade_date}")
    write_frame(daily_path, daily, fmt)
    write_frame(adj_path, adj_factor, fmt)
    finished_at = datetime.now().isoformat(timespec="seconds")
    result = TushareDailyUpdateResult(
        trade_date=display_trade_date(resolved_trade_date),
        daily_rows=len(daily),
        adj_factor_rows=len(adj_factor),
        daily_path=str(daily_path),
        adj_factor_path=str(adj_path),
        manifest_path=str(manifest_path),
        dry_run=False,
        source="tushare.pro.daily+adj_factor",
        started_at=started_at,
        finished_at=finished_at,
        note="market daily update by trade_date; two provider calls",
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Update full-market A-share daily data by Tushare trade_date."
    )
    parser.add_argument(
        "--trade-date",
        default=None,
        help="YYYYMMDD or YYYY-MM-DD; defaults to latest research trade date.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--token", default=None, help="Tushare token; defaults to TUSHARE_TOKEN.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = update_tushare_daily_market(
        trade_date=args.trade_date,
        output_root=Path(args.output_root),
        fmt=args.format,
        token=args.token,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
