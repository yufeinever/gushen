from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_STATUS_DIR = Path("data/local/data_update_status")
DEFAULT_STATUS_PATH = DEFAULT_STATUS_DIR / "status.json"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_status(path: Path = DEFAULT_STATUS_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"updated_at": None, "jobs": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_status(status: dict[str, Any], path: Path = DEFAULT_STATUS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status["updated_at"] = now_iso()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def update_job_status(
    job_id: str,
    fields: dict[str, Any],
    path: Path = DEFAULT_STATUS_PATH,
) -> dict[str, Any]:
    status = load_status(path)
    jobs = status.setdefault("jobs", {})
    job = jobs.setdefault(job_id, {})
    job.update(fields)
    job["updated_at"] = now_iso()
    save_status(status, path)
    return job


def append_job_log(
    job_id: str,
    message: str,
    status_path: Path = DEFAULT_STATUS_PATH,
    max_lines: int = 80,
) -> None:
    status = load_status(status_path)
    job = status.setdefault("jobs", {}).setdefault(job_id, {})
    lines = list(job.get("recent_logs", []))
    lines.append(f"{now_iso()} {message}")
    job["recent_logs"] = lines[-max_lines:]
    job["updated_at"] = now_iso()
    save_status(status, status_path)
