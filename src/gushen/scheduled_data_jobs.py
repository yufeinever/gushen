from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from gushen.akshare_spot_daily_update import update_akshare_spot_daily
from gushen.bulk_daily_download import DEFAULT_FULL_HISTORY_START_DATE, DEFAULT_POOL, load_pool
from gushen.data_update_status import DEFAULT_STATUS_PATH, append_job_log, update_job_status
from gushen.incremental_daily_update import update_incremental_daily_bars
from gushen.trade_calendar import latest_research_trade_date


def planned_weekly_sleep_bounds(
    stock_count: int,
    target_hours: float = 3.0,
    workers: int = 3,
    expected_request_seconds: float = 2.0,
) -> tuple[float, float]:
    if stock_count <= 0:
        return 0.0, 0.0
    budget_seconds = target_hours * 3600.0
    per_worker_count = stock_count / max(workers, 1)
    planned_gap = max(0.0, budget_seconds / per_worker_count - expected_request_seconds)
    return max(0.2, planned_gap * 0.75), max(0.5, planned_gap * 1.25)


def run_daily_spot_job(args: argparse.Namespace) -> dict[str, Any]:
    try:
        result = update_akshare_spot_daily(
            trade_date=args.trade_date,
            output_root=Path(args.daily_output_root),
            status_path=Path(args.status_path),
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        update_job_status(
            "daily_spot",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            },
            Path(args.status_path),
        )
        raise
    return asdict(result)


def run_daily_gap_fill_job(args: argparse.Namespace) -> dict[str, Any]:
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
    return asdict(result)


def run_weekly_qfq_job(args: argparse.Namespace) -> dict[str, Any]:
    trade_date = args.trade_date or latest_research_trade_date()
    pool_file = Path(args.pool_file)
    stocks = load_pool(pool_file)
    requested = min(len(stocks), args.limit) if args.limit is not None else len(stocks)
    sleep_min, sleep_max = planned_weekly_sleep_bounds(
        stock_count=requested,
        target_hours=args.target_hours,
        workers=args.workers,
        expected_request_seconds=args.expected_request_seconds,
    )
    started_at = datetime.now().isoformat(timespec="seconds")
    job_fields: dict[str, Any] = {
        "job_id": "weekly_qfq_full_refresh",
        "name": "Weekly full-market qfq refresh",
        "status": "running",
        "trade_date": trade_date,
        "started_at": started_at,
        "pool_file": str(pool_file),
        "requested": requested,
        "processed": 0,
        "downloaded": 0,
        "skipped_cached": 0,
        "failed": 0,
        "empty": 0,
        "workers": args.workers,
        "target_hours": args.target_hours,
        "sleep_min": round(sleep_min, 3),
        "sleep_max": round(sleep_max, 3),
        "dry_run": args.dry_run,
    }
    update_job_status("weekly_qfq_full_refresh", job_fields, Path(args.status_path))
    command = [
        sys.executable,
        "-m",
        "gushen.bulk_daily_download",
        "--pool-file",
        str(pool_file),
        "--start-date",
        args.start_date,
        "--end-date",
        trade_date,
        "--adjust",
        args.adjust,
        "--cache-dir",
        args.cache_dir,
        "--state-dir",
        args.state_dir,
        "--workers",
        str(args.workers),
        "--timeout",
        str(args.timeout),
        "--batch-size",
        str(args.batch_size),
        "--sleep-min",
        f"{sleep_min:.3f}",
        "--sleep-max",
        f"{sleep_max:.3f}",
        "--batch-sleep-min",
        str(args.batch_sleep_min),
        "--batch-sleep-max",
        str(args.batch_sleep_max),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.dry_run:
        command.append("--dry-run")
    append_job_log("weekly_qfq_full_refresh", "starting: " + " ".join(command), Path(args.status_path))
    process = subprocess.Popen(
        command,
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    stats = dict(job_fields)
    assert process.stdout is not None
    for line in process.stdout:
        line = line.strip()
        if not line:
            continue
        append_job_log("weekly_qfq_full_refresh", line, Path(args.status_path))
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "stock":
            stats["processed"] = int(stats.get("processed", 0)) + 1
            status = event.get("status")
            if status in {"downloaded", "skipped_cached", "failed", "empty"}:
                stats[status] = int(stats.get(status, 0)) + 1
            stats["last_stock"] = {
                "index": event.get("index"),
                "ts_code": event.get("ts_code"),
                "name": event.get("name"),
                "status": status,
            }
            stats["progress_pct"] = round(stats["processed"] / max(requested, 1) * 100, 2)
            update_job_status("weekly_qfq_full_refresh", stats, Path(args.status_path))
    return_code = process.wait()
    finished_at = datetime.now().isoformat(timespec="seconds")
    stats["status"] = "success" if return_code == 0 else "failed"
    stats["return_code"] = return_code
    stats["finished_at"] = finished_at
    update_job_status("weekly_qfq_full_refresh", stats, Path(args.status_path))
    if return_code != 0:
        raise RuntimeError(f"weekly qfq refresh failed with exit code {return_code}")
    return stats


def install_systemd_user_units(args: argparse.Namespace) -> None:
    project_dir = Path.cwd()
    python_path = project_dir / ".venv" / "bin" / "python"
    if not python_path.exists():
        python_path = Path(sys.executable)
    user_dir = Path.home() / ".config" / "systemd" / "user"
    user_dir.mkdir(parents=True, exist_ok=True)
    daily_service = user_dir / "gushen-daily-spot.service"
    daily_timer = user_dir / "gushen-daily-spot.timer"
    weekly_service = user_dir / "gushen-weekly-qfq-refresh.service"
    weekly_timer = user_dir / "gushen-weekly-qfq-refresh.timer"
    status_service = user_dir / "gushen-data-status-web.service"
    daily_service.write_text(
        f"""[Unit]
Description=Gushen AKShare daily market snapshot
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory={project_dir}
ExecStart={python_path} -m gushen.scheduled_data_jobs daily-gap-fill
""",
        encoding="utf-8",
    )
    daily_timer.write_text(
        """[Unit]
Description=Run Gushen daily market snapshot after A-share close

[Timer]
OnCalendar=Mon..Fri 16:00
Persistent=true

[Install]
WantedBy=timers.target
""",
        encoding="utf-8",
    )
    weekly_service.write_text(
        f"""[Unit]
Description=Gushen weekly full-market qfq refresh
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory={project_dir}
ExecStart={python_path} -m gushen.scheduled_data_jobs weekly-qfq
""",
        encoding="utf-8",
    )
    weekly_timer.write_text(
        """[Unit]
Description=Run Gushen weekly full-market qfq refresh

[Timer]
OnCalendar=Fri 18:00
Persistent=true

[Install]
WantedBy=timers.target
""",
        encoding="utf-8",
    )
    status_service.write_text(
        f"""[Unit]
Description=Gushen data update status web
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={project_dir}
ExecStart={python_path} -m gushen.data_status_web --host 127.0.0.1 --port {args.status_port}
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
""",
        encoding="utf-8",
    )
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", "gushen-daily-spot.timer"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", "gushen-weekly-qfq-refresh.timer"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", "gushen-data-status-web.service"], check=True)


def default_weekly_trade_date() -> str:
    return latest_research_trade_date()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run or install Gushen scheduled data jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    daily = subparsers.add_parser("daily-spot")
    daily.add_argument("--trade-date", default=None)
    daily.add_argument("--daily-output-root", default="data/local/akshare_market")
    daily.add_argument("--status-path", default=str(DEFAULT_STATUS_PATH))
    daily.add_argument("--dry-run", action="store_true")
    daily.add_argument("--overwrite", action="store_true")
    gap = subparsers.add_parser("daily-gap-fill")
    gap.add_argument("--trade-date", default=None)
    gap.add_argument("--pool-file", default=str(DEFAULT_POOL))
    gap.add_argument("--cache-dir", default="data/local/guided_factor_backtests/daily_bars")
    gap.add_argument("--state-dir", default="data/local/incremental_daily_updates")
    gap.add_argument("--status-path", default=str(DEFAULT_STATUS_PATH))
    gap.add_argument("--adjust", default="qfq")
    gap.add_argument("--workers", type=int, default=3)
    gap.add_argument("--timeout", type=float, default=8.0)
    gap.add_argument("--overlap-days", type=int, default=7)
    gap.add_argument("--sleep-min", type=float, default=0.8)
    gap.add_argument("--sleep-max", type=float, default=1.5)
    gap.add_argument("--limit", type=int, default=None)
    gap.add_argument("--dry-run", action="store_true")
    weekly = subparsers.add_parser("weekly-qfq")
    weekly.add_argument("--trade-date", default=None)
    weekly.add_argument("--pool-file", default=str(DEFAULT_POOL))
    weekly.add_argument("--start-date", default=DEFAULT_FULL_HISTORY_START_DATE)
    weekly.add_argument("--adjust", default="qfq")
    weekly.add_argument("--cache-dir", default="data/local/guided_factor_backtests/daily_bars")
    weekly.add_argument("--state-dir", default="data/local/bulk_daily_downloads")
    weekly.add_argument("--status-path", default=str(DEFAULT_STATUS_PATH))
    weekly.add_argument("--workers", type=int, default=3)
    weekly.add_argument("--timeout", type=float, default=8.0)
    weekly.add_argument("--target-hours", type=float, default=3.0)
    weekly.add_argument("--expected-request-seconds", type=float, default=2.0)
    weekly.add_argument("--batch-size", type=int, default=100)
    weekly.add_argument("--batch-sleep-min", type=float, default=0.0)
    weekly.add_argument("--batch-sleep-max", type=float, default=0.0)
    weekly.add_argument("--limit", type=int, default=None)
    weekly.add_argument("--dry-run", action="store_true")
    install = subparsers.add_parser("install-systemd")
    install.add_argument("--status-port", type=int, default=18088)
    args = parser.parse_args(argv)
    if args.command == "daily-spot":
        result = run_daily_spot_job(args)
    elif args.command == "daily-gap-fill":
        result = run_daily_gap_fill_job(args)
    elif args.command == "weekly-qfq":
        result = run_weekly_qfq_job(args)
    elif args.command == "install-systemd":
        install_systemd_user_units(args)
        result = {"status": "installed", "status_port": args.status_port}
    else:
        raise ValueError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
