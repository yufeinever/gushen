from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from gushen.bulk_daily_download import DEFAULT_POOL, load_pool
from gushen.data_update_status import DEFAULT_STATUS_PATH, update_job_status
from gushen.domestic_network import domestic_data_no_proxy
from gushen.trade_calendar import latest_research_trade_date

DEFAULT_OUTPUT_ROOT = Path("data/local/akshare_market")
TENCENT_QUOTE_URL = "https://qt.gtimg.cn/q="
SPOT_COLUMNS = [
    "trade_date",
    "code",
    "ts_code",
    "name",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "pct_change",
    "volume",
    "amount",
    "turnover",
    "source_rank",
    "fetched_at",
]


@dataclass(frozen=True)
class AkshareSpotDailyUpdateResult:
    trade_date: str
    rows: int
    valid_rows: int
    output_path: str
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


def to_ts_code(code: str) -> str:
    raw = "".join(character for character in str(code) if character.isdigit()).zfill(6)
    if raw.startswith(("6", "9")):
        return f"{raw}.SH"
    if raw.startswith(("0", "3")):
        return f"{raw}.SZ"
    if raw.startswith(("4", "8")):
        return f"{raw}.BJ"
    return raw


def fetch_akshare_spot_frame() -> pd.DataFrame:
    import akshare as ak

    with domestic_data_no_proxy():
        return ak.stock_zh_a_spot_em()


def tencent_symbol(code: str) -> str:
    raw = "".join(character for character in str(code) if character.isdigit()).zfill(6)
    if raw.startswith(("6", "9")):
        return f"sh{raw}"
    return f"sz{raw}"


def float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def fetch_tencent_spot_frame(pool_file: Path = DEFAULT_POOL, batch_size: int = 120) -> pd.DataFrame:
    stocks = load_pool(pool_file)
    rows: list[dict[str, Any]] = []
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"}
    symbols = [tencent_symbol(stock["code"]) for stock in stocks]
    with requests.Session() as session:
        session.trust_env = False
        for offset in range(0, len(symbols), batch_size):
            batch = symbols[offset : offset + batch_size]
            response = session.get(TENCENT_QUOTE_URL + ",".join(batch), headers=headers, timeout=12, proxies={})
            response.raise_for_status()
            for line in response.text.split(";"):
                if "~" not in line:
                    continue
                try:
                    body = line.split('="', 1)[1].rsplit('"', 1)[0]
                except IndexError:
                    continue
                parts = body.split("~")
                if len(parts) < 39:
                    continue
                code = "".join(character for character in parts[2] if character.isdigit()).zfill(6)
                close = float_or_none(parts[3])
                pre_close = float_or_none(parts[4])
                open_price = float_or_none(parts[5])
                high = float_or_none(parts[33])
                low = float_or_none(parts[34])
                amount_wan = float_or_none(parts[37])
                if close is None or open_price is None or high is None or low is None or amount_wan is None:
                    continue
                rows.append(
                    {
                        "代码": code,
                        "名称": parts[1],
                        "今开": open_price,
                        "最高": high,
                        "最低": low,
                        "最新价": close,
                        "昨收": pre_close,
                        "涨跌幅": float_or_none(parts[32]),
                        "成交量": float_or_none(parts[36]),
                        "成交额": amount_wan * 10000,
                        "换手率": float_or_none(parts[38]),
                        "序号": None,
                    }
                )
            time.sleep(0.05)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("成交额", ascending=False).reset_index(drop=True)
        frame["序号"] = frame.index + 1
    return frame


def fetch_spot_frame_with_fallback() -> tuple[pd.DataFrame, str, str | None]:
    try:
        return fetch_akshare_spot_frame(), "akshare.stock_zh_a_spot_em", None
    except Exception as exc:  # noqa: BLE001 - use Tencent when EastMoney/AKShare is unavailable.
        frame = fetch_tencent_spot_frame()
        return frame, "tencent.qt.gtimg.cn", f"AKShare failed: {type(exc).__name__}: {exc}"


def normalize_spot_frame(frame: pd.DataFrame, trade_date: str, fetched_at: str) -> pd.DataFrame:
    column_map = {
        "代码": "code",
        "名称": "name",
        "今开": "open",
        "最高": "high",
        "最低": "low",
        "最新价": "close",
        "昨收": "pre_close",
        "涨跌幅": "pct_change",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover",
        "序号": "source_rank",
    }
    missing = [column for column in column_map if column not in frame.columns]
    if missing:
        raise RuntimeError(f"AKShare spot frame missing columns: {missing}")
    normalized = frame.rename(columns=column_map)[list(column_map.values())].copy()
    normalized["code"] = normalized["code"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    normalized["ts_code"] = normalized["code"].map(to_ts_code)
    normalized["trade_date"] = display_trade_date(trade_date)
    normalized["fetched_at"] = fetched_at
    for column in [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "pct_change",
        "volume",
        "amount",
        "turnover",
        "source_rank",
    ]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized[SPOT_COLUMNS]
    normalized = normalized.dropna(subset=["code", "ts_code", "name"])
    normalized = normalized.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    normalized = normalized.sort_values("ts_code").reset_index(drop=True)
    return normalized


def build_output_paths(output_root: Path, trade_date: str) -> tuple[Path, Path]:
    raw = normalize_trade_date(trade_date)
    output_path = output_root / "raw_daily_by_date" / f"trade_date={raw}.csv"
    manifest_path = output_root / "manifests" / f"trade_date={raw}.json"
    return output_path, manifest_path


def assess_valid_rows(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    required = frame[["open", "high", "low", "close", "amount"]].notna().all(axis=1)
    positive_price = (frame["open"] > 0) & (frame["high"] > 0) & (frame["low"] > 0) & (frame["close"] > 0)
    valid_ohlc = (frame["high"] >= frame[["open", "close", "low"]].max(axis=1)) & (
        frame["low"] <= frame[["open", "close", "high"]].min(axis=1)
    )
    return int((required & positive_price & valid_ohlc).sum())


def update_akshare_spot_daily(
    trade_date: str | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    dry_run: bool = False,
    overwrite: bool = False,
    frame: pd.DataFrame | None = None,
    min_rows: int = 4000,
    min_valid_rows: int = 3500,
    status_path: Path = DEFAULT_STATUS_PATH,
) -> AkshareSpotDailyUpdateResult:
    resolved_trade_date = normalize_trade_date(
        trade_date or latest_research_trade_date().replace("-", "")
    )
    started_at = datetime.now().isoformat(timespec="seconds")
    output_path, manifest_path = build_output_paths(output_root, resolved_trade_date)
    update_job_status(
        "daily_spot",
        {
            "job_id": "daily_spot",
            "name": "AKShare full-market raw daily update",
            "status": "running",
            "trade_date": display_trade_date(resolved_trade_date),
            "started_at": started_at,
            "output_path": str(output_path),
            "manifest_path": str(manifest_path),
            "dry_run": dry_run,
        },
        status_path,
    )
    if not overwrite and output_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = AkshareSpotDailyUpdateResult(**manifest)
        update_job_status(
            "daily_spot",
            {"status": "cached", "rows": result.rows, "valid_rows": result.valid_rows, "finished_at": result.finished_at},
            status_path,
        )
        return result
    if dry_run:
        result = AkshareSpotDailyUpdateResult(
            trade_date=display_trade_date(resolved_trade_date),
            rows=0,
            valid_rows=0,
            output_path=str(output_path),
            manifest_path=str(manifest_path),
            dry_run=True,
            source="akshare.stock_zh_a_spot_em",
            started_at=started_at,
            finished_at=datetime.now().isoformat(timespec="seconds"),
            note="dry run; no network call or file write",
        )
        update_job_status("daily_spot", {**asdict(result), "status": "dry_run"}, status_path)
        return result
    source = "injected_frame"
    fallback_note = None
    if frame is None:
        frame, source, fallback_note = fetch_spot_frame_with_fallback()
    normalized = normalize_spot_frame(frame, resolved_trade_date, datetime.now().isoformat(timespec="seconds"))
    valid_rows = assess_valid_rows(normalized)
    if len(normalized) < min_rows:
        raise RuntimeError(f"AKShare spot frame has too few rows: {len(normalized)}")
    if valid_rows < min_valid_rows:
        raise RuntimeError(f"AKShare spot frame has too few valid OHLC rows: {valid_rows}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_path, index=False)
    result = AkshareSpotDailyUpdateResult(
        trade_date=display_trade_date(resolved_trade_date),
        rows=len(normalized),
        valid_rows=valid_rows,
        output_path=str(output_path),
        manifest_path=str(manifest_path),
        dry_run=False,
        source=source,
        started_at=started_at,
        finished_at=datetime.now().isoformat(timespec="seconds"),
        note="full-market raw daily snapshot after close" if fallback_note is None else fallback_note,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    update_job_status("daily_spot", {**asdict(result), "status": "success"}, status_path)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Update full-market raw daily bars from AKShare spot snapshot.")
    parser.add_argument("--trade-date", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--status-path", default=str(DEFAULT_STATUS_PATH))
    parser.add_argument("--min-rows", type=int, default=4000)
    parser.add_argument("--min-valid-rows", type=int, default=3500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = update_akshare_spot_daily(
        trade_date=args.trade_date,
        output_root=Path(args.output_root),
        status_path=Path(args.status_path),
        min_rows=args.min_rows,
        min_valid_rows=args.min_valid_rows,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
