from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from gushen.top20_ma5_pullback_strategy import DEFAULT_CACHE_DIR, StrategyConfig, build_market_frame, load_daily_frames, round_float
from gushen.top20_ma5_realistic_backtest import (
    OBSERVATION_DAYS,
    PRICE_CAP,
    PER_POSITION,
    build_observation_pool,
    dynamic_boundary_for_date,
    next_trade_date,
    next_weekday,
    recommend_lot,
    select_signal_candidates,
)
from gushen.top20_ma5_live_monitor import TENCENT_MINUTE_URL, tencent_symbol

DEFAULT_OUTPUT_DIR = Path("data/local/intraday/tencent_1m")
DEFAULT_REPORT_DIR = Path("reports/generated/top20_ma5_realistic_20260601_20260701")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Tencent 1-minute bars needed by Top20 MA5 realistic backtest.")
    parser.add_argument("--daily-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--start-date", default="2026-06-01")
    parser.add_argument("--end-date", default="2026-07-01")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--per-position", type=float, default=PER_POSITION)
    parser.add_argument("--price-cap", type=float, default=PRICE_CAP)
    parser.add_argument("--limit", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = download_required_minutes(
        daily_cache_dir=args.daily_cache_dir,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=args.top_n,
        per_position=args.per_position,
        price_cap=args.price_cap,
        limit=args.limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def download_required_minutes(
    daily_cache_dir: Path,
    output_dir: Path,
    report_dir: Path,
    start_date: str,
    end_date: str,
    top_n: int,
    per_position: float,
    price_cap: float,
    limit: int,
) -> dict[str, Any]:
    required = build_required_pairs(daily_cache_dir, start_date, end_date, top_n, per_position, price_cap)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    existing = existing_pairs(output_dir)
    by_code: dict[str, dict[str, Any]] = {}
    for item in required:
        by_code.setdefault(item["code"], {"name": item["name"], "dates": set()})["dates"].add(item["date"])

    downloaded_pairs = 0
    unavailable_pairs = 0
    skipped_existing = 0
    source_code_dates: dict[str, list[str]] = {}
    missing_rows: list[dict[str, Any]] = []
    for code, payload in sorted(by_code.items()):
        name = str(payload["name"])
        needed_dates = sorted(payload["dates"])
        frame = fetch_tencent_1m(code, limit)
        available_dates = sorted(frame["time"].str[:10].unique().tolist()) if not frame.empty else []
        source_code_dates[code] = available_dates
        for trade_date in needed_dates:
            if (code, trade_date) in existing:
                skipped_existing += 1
                continue
            day = frame[frame["time"].str.startswith(trade_date)].copy() if not frame.empty else pd.DataFrame()
            if day.empty or not has_complete_morning_window(day):
                unavailable_pairs += 1
                missing_rows.append({"code": code, "name": name, "date": trade_date, "available_dates": ",".join(available_dates)})
                continue
            path = cache_path(output_dir, code, name, trade_date)
            path.parent.mkdir(parents=True, exist_ok=True)
            day.sort_values("time").to_csv(path, index=False)
            downloaded_pairs += 1
            append_manifest(manifest_path, code, name, trade_date, path, day, "tencent_1m")
    required_df = pd.DataFrame(required).sort_values(["date", "code", "purpose"])
    missing_df = pd.DataFrame(missing_rows)
    required_code_date_pairs = int(required_df[["code", "date"]].drop_duplicates().shape[0]) if not required_df.empty else 0
    report_dir.mkdir(parents=True, exist_ok=True)
    required_path = report_dir / "required_1m_pairs.csv"
    missing_path = report_dir / "missing_1m_pairs.csv"
    required_df.to_csv(required_path, index=False)
    missing_df.to_csv(missing_path, index=False)
    summary = {
        "start_date": start_date,
        "end_date": end_date,
        "source": "tencent_1m",
        "cache_dir": str(output_dir),
        "required_purpose_pairs": int(len(required_df)),
        "required_code_date_pairs": required_code_date_pairs,
        "required_codes": int(required_df["code"].nunique()) if not required_df.empty else 0,
        "downloaded_pairs": downloaded_pairs,
        "skipped_existing_pairs": skipped_existing,
        "unavailable_pairs": unavailable_pairs,
        "available_dates_by_code_sample": {code: dates for code, dates in list(source_code_dates.items())[:10]},
        "required_pairs_path": str(required_path),
        "missing_pairs_path": str(missing_path),
        "note": "Tencent m1 only exposes recent minute bars. Older June dates are expected to be unavailable from this free endpoint.",
    }
    summary_path = report_dir / "download_1m_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_required_pairs(
    daily_cache_dir: Path,
    start_date: str,
    end_date: str,
    top_n: int,
    per_position: float,
    price_cap: float,
) -> list[dict[str, Any]]:
    config = StrategyConfig(
        start_date=start_date,
        end_date=end_date,
        top_n=top_n,
        wait_days=OBSERVATION_DAYS,
        max_positions=999999,
        position_pct=1.0,
        commission_rate=0.00025,
        slippage_rate=0.0,
    )
    # load_daily_frames already adds warmup before config.start_date.
    frames = load_daily_frames(daily_cache_dir, config)
    market = build_market_frame(frames, config)
    candidates = select_signal_candidates(market, top_n)
    trade_dates = sorted({str(item) for item in market["trade_date"].unique()})
    execution_dates = [item for item in trade_dates if start_date <= item <= end_date]
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for execution_date in execution_dates:
        pool = build_observation_pool(candidates, frames, trade_dates, execution_date, OBSERVATION_DAYS)
        for item in pool:
            frame = frames.get(item["code"])
            if frame is None:
                continue
            boundary = dynamic_boundary_for_date(frame, execution_date)
            if boundary is None or boundary > price_cap or recommend_lot(boundary, per_position) <= 0:
                continue
            exit_date = next_trade_date(trade_dates, execution_date) or next_weekday(execution_date)
            add_required(rows, item, execution_date, "entry", boundary)
            add_required(rows, item, exit_date, "exit", boundary)
    return list(rows.values())


def add_required(rows: dict[tuple[str, str, str], dict[str, Any]], item: dict[str, Any], trade_date: str, purpose: str, boundary: float) -> None:
    key = (item["code"], trade_date, purpose)
    rows[key] = {
        "code": item["code"],
        "name": item["name"],
        "date": trade_date,
        "purpose": purpose,
        "best_rank": item.get("best_rank"),
        "source": item.get("source"),
        "source_ranks": item.get("source_ranks"),
        "boundary_price": round_float(boundary, 4),
    }


def fetch_tencent_1m(code: str, limit: int) -> pd.DataFrame:
    symbol = tencent_symbol(code)
    try:
        response = requests.get(
            TENCENT_MINUTE_URL,
            params={"param": f"{symbol},m1,,{limit}"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=12,
        )
        response.raise_for_status()
        rows = ((response.json().get("data") or {}).get(symbol) or {}).get("m1", [])
    except Exception:
        return empty_frame()
    records = []
    for row in rows:
        if len(row) < 6:
            continue
        stamp = str(row[0])
        if len(stamp) < 12:
            continue
        records.append(
            {
                "time": f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[8:10]}:{stamp[10:12]}:00",
                "open": to_float(row[1]),
                "close": to_float(row[2]),
                "high": to_float(row[3]),
                "low": to_float(row[4]),
                "volume": to_float(row[5]),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        return empty_frame()
    return frame.dropna(subset=["open", "close", "high", "low"]).sort_values("time").reset_index(drop=True)


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])


def has_complete_morning_window(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    minutes = set(frame["time"].str[11:16].tolist())
    return "09:30" in minutes and "10:00" in minutes


def cache_path(output_dir: Path, code: str, name: str, trade_date: str) -> Path:
    return output_dir / f"{code}-{safe_name(name)}" / f"{trade_date}.csv"


def existing_pairs(output_dir: Path) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for path in output_dir.glob("*/*.csv"):
        folder = path.parent.name
        code = folder.split("-", 1)[0]
        pairs.add((code, path.stem))
    return pairs


def append_manifest(manifest_path: Path, code: str, name: str, trade_date: str, path: Path, frame: pd.DataFrame, source: str) -> None:
    payload = {
        "source": source,
        "code": code,
        "name": name,
        "trade_date": trade_date,
        "path": str(path),
        "rows": int(len(frame)),
        "first_time": str(frame["time"].iloc[0]) if not frame.empty else None,
        "last_time": str(frame["time"].iloc[-1]) if not frame.empty else None,
    }
    with manifest_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def safe_name(value: str) -> str:
    name = str(value).strip() or "UNKNOWN"
    return re.sub(r'[\\/:*?"<>|\s]+', "_", name)


if __name__ == "__main__":
    main()
