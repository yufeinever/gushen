from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from gushen.bulk_daily_download import DEFAULT_CACHE_DIR, DEFAULT_FULL_HISTORY_START_DATE, DEFAULT_POOL, load_pool
from gushen.data import DailyBar, fetch_daily_bars
from gushen.data_update_status import DEFAULT_STATUS_PATH, append_job_log, update_job_status
from gushen.guided_factor_backtest import to_ts_code
from gushen.trade_calendar import latest_research_trade_date

CACHE_NAME_RE = re.compile(r"^(?P<ts_code>.+)_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})\.csv$")


@dataclass(frozen=True)
class CacheFile:
    path: Path
    start_date: str
    end_date: str


@dataclass(frozen=True)
class IncrementalDailyUpdateResult:
    trade_date: str
    requested: int
    processed: int
    downloaded: int
    skipped_current: int
    failed: int
    empty: int
    cache_dir: str
    state_path: str
    dry_run: bool
    started_at: str
    finished_at: str


def latest_cache_file(cache_dir: Path, adjust: str, ts_code: str) -> CacheFile | None:
    candidates: list[CacheFile] = []
    for path in (cache_dir / adjust).glob(f"{ts_code}_*.csv"):
        match = CACHE_NAME_RE.match(path.name)
        if not match:
            continue
        candidates.append(CacheFile(path=path, start_date=match["start"], end_date=match["end"]))
    if not candidates:
        return None
    full_history = [item for item in candidates if item.start_date == DEFAULT_FULL_HISTORY_START_DATE]
    if full_history:
        return max(full_history, key=lambda item: (item.end_date, item.path.stat().st_mtime))
    return max(candidates, key=lambda item: (item.end_date, item.start_date, item.path.stat().st_mtime))


def read_daily_bars(path: Path) -> list[DailyBar]:
    with path.open(newline="", encoding="utf-8") as file:
        return [DailyBar(**coerce_daily_bar(row)) for row in csv.DictReader(file)]


def coerce_daily_bar(row: dict[str, str]) -> dict[str, Any]:
    values: dict[str, Any] = {"trade_date": row["trade_date"], "code": row["code"], "name": row["name"]}
    for key in ["open", "close", "high", "low", "volume", "amount", "amplitude", "pct_change", "turnover"]:
        values[key] = float(row[key])
    return values


def write_daily_bars(path: Path, rows: list[DailyBar]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def merge_daily_bars(existing: list[DailyBar], fetched: list[DailyBar]) -> list[DailyBar]:
    by_date = {row.trade_date: row for row in existing}
    by_date.update({row.trade_date: row for row in fetched})
    return [by_date[trade_date] for trade_date in sorted(by_date)]


def backfill_start_date(last_date: str, overlap_days: int) -> str:
    return (date.fromisoformat(last_date) - timedelta(days=max(0, overlap_days))).isoformat()


def update_one_stock(
    stock: dict[str, Any],
    index: int,
    total: int,
    cache_dir: Path,
    adjust: str,
    end_date: str,
    overlap_days: int,
    timeout: float,
    dry_run: bool,
    fetcher: Callable[..., list[DailyBar]] = fetch_daily_bars,
) -> dict[str, Any]:
    started = time.monotonic()
    code = stock["code"]
    name = stock["name"]
    ts_code = to_ts_code(code)
    cache = latest_cache_file(cache_dir, adjust, ts_code)
    event: dict[str, Any] = {
        "event": "stock",
        "index": index,
        "total": total,
        "rank": stock.get("rank"),
        "code": code,
        "ts_code": ts_code,
        "name": name,
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        if cache is None:
            start_date = DEFAULT_FULL_HISTORY_START_DATE
            existing: list[DailyBar] = []
            source_path = None
        else:
            source_path = cache.path
            existing = read_daily_bars(cache.path)
            if not existing:
                start_date = cache.start_date
            else:
                start_date = cache.start_date
                actual_last = existing[-1].trade_date
                event["current_last_date"] = actual_last
                if actual_last >= end_date:
                    desired_path = cache_dir / adjust / f"{ts_code}_{start_date}_{end_date}.csv"
                    if desired_path != cache.path and not dry_run:
                        write_daily_bars(desired_path, existing)
                    event |= {
                        "status": "skipped_current",
                        "source_path": str(cache.path),
                        "cache_path": str(desired_path),
                        "rows": len(existing),
                    }
                    return event
        fetch_start = start_date if not existing else backfill_start_date(existing[-1].trade_date, overlap_days)
        output_path = cache_dir / adjust / f"{ts_code}_{start_date}_{end_date}.csv"
        event |= {
            "source_path": None if source_path is None else str(source_path),
            "cache_path": str(output_path),
            "fetch_start": fetch_start,
            "fetch_end": end_date,
        }
        if dry_run:
            event |= {"status": "dry_run_missing" if not existing else "dry_run_update"}
            return event
        fetched = fetcher(ts_code, name, fetch_start, end_date, timeout=timeout, adjust=adjust)
        if not fetched:
            event |= {"status": "empty", "existing_rows": len(existing)}
            return event
        merged = merge_daily_bars(existing, fetched)
        write_daily_bars(output_path, merged)
        event |= {
            "status": "downloaded",
            "fetched_rows": len(fetched),
            "rows": len(merged),
            "first_date": merged[0].trade_date,
            "last_date": merged[-1].trade_date,
            "bytes": output_path.stat().st_size,
        }
    except Exception as exc:  # noqa: BLE001 - keep daily repair resilient.
        event |= {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
    finally:
        event["duration_seconds"] = round(time.monotonic() - started, 3)
    return event


def update_incremental_daily_bars(
    trade_date: str | None = None,
    pool_file: Path = DEFAULT_POOL,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    state_dir: Path = Path("data/local/incremental_daily_updates"),
    status_path: Path = DEFAULT_STATUS_PATH,
    adjust: str = "qfq",
    workers: int = 3,
    timeout: float = 8.0,
    overlap_days: int = 7,
    sleep_min: float = 0.8,
    sleep_max: float = 1.5,
    limit: int | None = None,
    dry_run: bool = False,
) -> IncrementalDailyUpdateResult:
    end_date = trade_date or latest_research_trade_date()
    stocks = load_pool(pool_file)
    if limit is not None:
        stocks = stocks[:limit]
    state_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat(timespec="seconds")
    state_path = state_dir / f"incremental_daily_update_{end_date}_{adjust}.jsonl"
    stats: dict[str, Any] = {
        "job_id": "daily_gap_fill",
        "name": "Daily missing-range qfq update",
        "status": "running",
        "trade_date": end_date,
        "requested": len(stocks),
        "processed": 0,
        "downloaded": 0,
        "skipped_current": 0,
        "failed": 0,
        "empty": 0,
        "workers": workers,
        "overlap_days": overlap_days,
        "sleep_min": sleep_min,
        "sleep_max": sleep_max,
        "cache_dir": str(cache_dir),
        "state_path": str(state_path),
        "dry_run": dry_run,
        "started_at": started_at,
    }
    update_job_status("daily_gap_fill", stats, status_path)
    append_job_log("daily_gap_fill", f"starting incremental update to {end_date}, requested={len(stocks)}", status_path)
    indexed = list(enumerate(stocks, start=1))
    errors = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                update_one_stock,
                stock,
                index,
                len(stocks),
                cache_dir,
                adjust,
                end_date,
                overlap_days,
                timeout,
                dry_run,
            ): index
            for index, stock in indexed
        }
        for future in as_completed(futures):
            event = future.result()
            status = event.get("status")
            stats["processed"] = int(stats["processed"]) + 1
            if status in {"downloaded", "skipped_current", "failed", "empty"}:
                stats[status] = int(stats[status]) + 1
            if status == "failed":
                errors += 1
            stats["last_stock"] = {
                "index": event.get("index"),
                "ts_code": event.get("ts_code"),
                "name": event.get("name"),
                "status": status,
            }
            stats["progress_pct"] = round(int(stats["processed"]) / max(len(stocks), 1) * 100, 2)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            with state_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(event, ensure_ascii=False) + "\n")
            if status in {"failed", "downloaded", "empty"}:
                append_job_log("daily_gap_fill", json.dumps(event, ensure_ascii=False), status_path)
            update_job_status("daily_gap_fill", stats, status_path)
            time.sleep(random.uniform(sleep_min, sleep_max))
    finished_at = datetime.now().isoformat(timespec="seconds")
    stats["status"] = "failed" if errors else "success"
    stats["finished_at"] = finished_at
    update_job_status("daily_gap_fill", stats, status_path)
    return IncrementalDailyUpdateResult(
        trade_date=end_date,
        requested=len(stocks),
        processed=int(stats["processed"]),
        downloaded=int(stats["downloaded"]),
        skipped_current=int(stats["skipped_current"]),
        failed=int(stats["failed"]),
        empty=int(stats["empty"]),
        cache_dir=str(cache_dir),
        state_path=str(state_path),
        dry_run=dry_run,
        started_at=started_at,
        finished_at=finished_at,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Incrementally fill missing A-share daily bars by stock.")
    parser.add_argument("--trade-date", default=None)
    parser.add_argument("--pool-file", default=str(DEFAULT_POOL))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--state-dir", default="data/local/incremental_daily_updates")
    parser.add_argument("--status-path", default=str(DEFAULT_STATUS_PATH))
    parser.add_argument("--adjust", default="qfq")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--overlap-days", type=int, default=7)
    parser.add_argument("--sleep-min", type=float, default=0.8)
    parser.add_argument("--sleep-max", type=float, default=1.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = update_incremental_daily_bars(
        trade_date=args.trade_date,
        pool_file=Path(args.pool_file),
        cache_dir=Path(args.cache_dir),
        state_dir=Path(args.state_dir),
        status_path=Path(args.status_path),
        adjust=args.adjust,
        workers=args.workers,
        timeout=args.timeout,
        overlap_days=args.overlap_days,
        sleep_min=args.sleep_min,
        sleep_max=args.sleep_max,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
