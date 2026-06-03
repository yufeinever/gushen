from __future__ import annotations

import argparse
import csv
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from gushen.data import DailyBar, fetch_daily_bars
from gushen.guided_factor_backtest import normalize_stock_code, to_ts_code

DEFAULT_POOL = Path('data/local/Table_4860_2026-06-03.xlsx')
DEFAULT_CACHE_DIR = Path('data/local/guided_factor_backtests/daily_bars')
DEFAULT_STATE_DIR = Path('data/local/bulk_daily_downloads')
DEFAULT_FULL_HISTORY_START_DATE = '1990-01-01'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slowly bulk-download A-share daily bars with resume support.')
    parser.add_argument('--pool-file', default=str(DEFAULT_POOL))
    parser.add_argument('--start-date', default=DEFAULT_FULL_HISTORY_START_DATE)
    parser.add_argument('--end-date', default='2026-06-03')
    parser.add_argument('--years', type=int, default=None, help='Optional rolling window override; omitted downloads full history.')
    parser.add_argument('--adjust', default='qfq')
    parser.add_argument('--cache-dir', default=str(DEFAULT_CACHE_DIR))
    parser.add_argument('--state-dir', default=str(DEFAULT_STATE_DIR))
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--sleep-min', type=float, default=2.0)
    parser.add_argument('--sleep-max', type=float, default=3.0)
    parser.add_argument('--batch-sleep-min', type=float, default=60.0)
    parser.add_argument('--batch-sleep-max', type=float, default=90.0)
    parser.add_argument('--max-errors', type=int, default=80)
    parser.add_argument('--timeout', type=float, default=20.0, help='Per-provider request timeout seconds for one stock.')
    parser.add_argument('--workers', type=int, default=1, help='Number of concurrent stock downloads.')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end_date = args.end_date
    start_date = args.start_date
    if args.years is not None:
        start_date = (date.fromisoformat(end_date) - timedelta(days=365 * args.years + 30)).isoformat()
    cache_dir = Path(args.cache_dir)
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    run_id = f'{start_date}_{end_date}_{args.adjust}'
    log_path = state_dir / f'bulk_daily_download_{run_id}.jsonl'
    summary_path = state_dir / f'bulk_daily_download_{run_id}.summary.json'

    stocks = load_pool(Path(args.pool_file))
    if args.offset:
        stocks = stocks[args.offset :]
    if args.limit is not None:
        stocks = stocks[: args.limit]

    stats: dict[str, Any] = {
        'pool_file': args.pool_file,
        'start_date': start_date,
        'end_date': end_date,
        'adjust': args.adjust,
        'requested': len(stocks),
        'downloaded': 0,
        'skipped_cached': 0,
        'failed': 0,
        'empty': 0,
        'dry_run': args.dry_run,
        'timeout': args.timeout,
        'workers': args.workers,
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    print(json.dumps({'event': 'start', **stats}, ensure_ascii=False), flush=True)
    errors = 0
    indexed_stocks = list(enumerate(stocks, start=1))
    workers = max(1, args.workers)
    for batch_start in range(0, len(indexed_stocks), args.batch_size):
        batch = indexed_stocks[batch_start : batch_start + args.batch_size]
        if workers == 1:
            for index, stock in batch:
                sleep_after = index < len(stocks) and index % args.batch_size != 0
                event = download_stock(index, stock, args, cache_dir, start_date, end_date, sleep_after=sleep_after)
                errors = record_event(log_path, stats, event, errors)
                if errors >= args.max_errors:
                    break
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        download_stock,
                        index,
                        stock,
                        args,
                        cache_dir,
                        start_date,
                        end_date,
                        True,
                    ): index
                    for index, stock in batch
                }
                for future in as_completed(futures):
                    event = future.result()
                    errors = record_event(log_path, stats, event, errors)

        if errors >= args.max_errors:
            print(json.dumps({'event': 'stop_max_errors', 'errors': errors}, ensure_ascii=False), flush=True)
            break

        last_index = batch[-1][0]
        if last_index < len(stocks):
            sleep_seconds = random.uniform(args.batch_sleep_min, args.batch_sleep_max)
            print(json.dumps({'event': 'batch_sleep', 'seconds': round(sleep_seconds, 2), 'index': last_index}, ensure_ascii=False), flush=True)
            time.sleep(sleep_seconds)

    stats['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    stats['log_path'] = str(log_path)
    summary_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'event': 'finish', **stats}, ensure_ascii=False), flush=True)


def record_event(log_path: Path, stats: dict[str, Any], event: dict[str, Any], errors: int) -> int:
    status = event.get('status')
    if status == 'downloaded':
        stats['downloaded'] += 1
    elif status == 'skipped_cached':
        stats['skipped_cached'] += 1
    elif status == 'failed':
        errors += 1
        stats['failed'] += 1
    elif status == 'empty':
        stats['empty'] += 1

    append_jsonl(log_path, event)
    print(json.dumps(event, ensure_ascii=False), flush=True)
    return errors


def download_stock(
    index: int,
    stock: dict[str, Any],
    args: argparse.Namespace,
    cache_dir: Path,
    start_date: str,
    end_date: str,
    sleep_after: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    code = stock['code']
    name = stock['name']
    ts_code = to_ts_code(code)
    cache_path = cache_dir / args.adjust / f'{ts_code}_{start_date}_{end_date}.csv'
    event: dict[str, Any] = {
        'event': 'stock',
        'index': index,
        'rank': stock.get('rank'),
        'code': code,
        'ts_code': ts_code,
        'name': name,
        'cache_path': str(cache_path),
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    try:
        if cache_path.exists() and cache_path.stat().st_size > 0:
            event |= {'status': 'skipped_cached', 'bytes': cache_path.stat().st_size}
        elif args.dry_run:
            event |= {'status': 'dry_run_missing'}
        else:
            rows = fetch_daily_bars(ts_code, name, start_date, end_date, timeout=args.timeout, adjust=args.adjust)
            if rows:
                write_daily_bars(cache_path, rows)
                event |= {
                    'status': 'downloaded',
                    'rows': len(rows),
                    'first_date': rows[0].trade_date,
                    'last_date': rows[-1].trade_date,
                    'bytes': cache_path.stat().st_size,
                }
            else:
                event |= {'status': 'empty'}
    except Exception as exc:  # noqa: BLE001 - keep downloader resilient and resumable.
        event |= {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)}
    finally:
        event['duration_seconds'] = round(time.monotonic() - started, 3)
        if sleep_after:
            time.sleep(random.uniform(args.sleep_min, args.sleep_max))
    return event


def load_pool(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        frame = pd.read_excel(path)
    else:
        frame = pd.read_csv(path)
    columns = {str(column).strip(): column for column in frame.columns}
    code_col = pick_column(columns, ['代码', 'code', '证券代码'])
    name_col = pick_column(columns, ['名称', 'name', '证券名称'])
    rank_col = columns.get('序') or columns.get('rank') or columns.get('排名')
    if rank_col is not None:
        frame = frame.sort_values(rank_col, kind='stable')
    stocks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(frame.to_dict('records'), start=1):
        code = normalize_stock_code(str(row.get(code_col) or ''))
        if not code or code in seen:
            continue
        seen.add(code)
        rank = row.get(rank_col) if rank_col is not None else index
        try:
            rank = int(float(rank))
        except (TypeError, ValueError):
            rank = index
        stocks.append({'rank': rank, 'code': code, 'name': str(row.get(name_col) or '').strip()})
    return stocks


def pick_column(columns: dict[str, Any], names: list[str]) -> Any:
    lowered = {key.lower(): value for key, value in columns.items()}
    for name in names:
        if name in columns:
            return columns[name]
        if name.lower() in lowered:
            return lowered[name.lower()]
    raise ValueError(f'missing expected column: {names}')


def write_daily_bars(path: Path, rows: list[DailyBar]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def append_jsonl(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(event, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
