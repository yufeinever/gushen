from __future__ import annotations

import argparse
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from gushen.data_update_status import DEFAULT_STATUS_PATH, load_status


class StatusHandler(BaseHTTPRequestHandler):
    status_path = DEFAULT_STATUS_PATH

    def do_GET(self) -> None:
        if self.path.startswith("/api/status"):
            self._send_json(load_status(self.status_path))
            return
        if self.path in {"/", "/index.html"}:
            self._send_html(render_status_page(load_status(self.status_path)))
            return
        self.send_error(404)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def render_status_page(status: dict[str, Any]) -> str:
    jobs = status.get("jobs", {})
    rows = "\n".join(render_job(job_id, job) for job_id, job in sorted(jobs.items()))
    if not rows:
        rows = "<section><h2>No jobs yet</h2><p>No data update job has written status.</p></section>"
    updated_at = html.escape(str(status.get("updated_at") or "never"))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="10">
  <title>Gushen Data Updates</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      background: #f6f7f9;
      color: #1f2933;
    }}
    header {{
      background: #111827;
      color: #fff;
      padding: 18px 28px;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }}
    section {{
      background: #fff;
      border: 1px solid #d8dee8;
      border-radius: 8px;
      margin-bottom: 18px;
      padding: 18px;
    }}
    h1 {{ margin: 0; font-size: 22px; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    .meta {{ color: #667085; margin-top: 6px; font-size: 13px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
    }}
    .item {{
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      padding: 10px;
      background: #fafafa;
    }}
    .label {{ color: #667085; font-size: 12px; }}
    .value {{ font-size: 16px; margin-top: 4px; overflow-wrap: anywhere; }}
    .status {{
      display: inline-block;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 13px;
      background: #e5e7eb;
    }}
    .status.success, .status.cached {{ background: #dcfce7; color: #166534; }}
    .status.running {{ background: #dbeafe; color: #1d4ed8; }}
    .status.failed {{ background: #fee2e2; color: #991b1b; }}
    .status.dry_run {{ background: #fef3c7; color: #92400e; }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #0f172a;
      color: #e5e7eb;
      border-radius: 6px;
      padding: 12px;
      max-height: 320px;
      overflow: auto;
    }}
    a {{ color: #2563eb; }}
  </style>
</head>
<body>
  <header>
    <h1>Gushen Data Updates</h1>
    <div class="meta">Last status write: {updated_at}. Page refreshes every 10 seconds.</div>
  </header>
  <main>
    {rows}
  </main>
</body>
</html>"""


def render_job(job_id: str, job: dict[str, Any]) -> str:
    fields = [
        ("Status", f"<span class=\"status {html.escape(str(job.get('status', 'unknown')))}\">{html.escape(str(job.get('status', 'unknown')))}</span>"),
        ("Trade date", html.escape(str(job.get("trade_date", "")))),
        ("Progress", html.escape(str(job.get("progress_pct", ""))) + ("%" if job.get("progress_pct") is not None else "")),
        ("Requested", html.escape(str(job.get("requested", "")))),
        ("Processed", html.escape(str(job.get("processed", "")))),
        ("Downloaded", html.escape(str(job.get("downloaded", "")))),
        ("Cached", html.escape(str(job.get("skipped_cached", "")))),
        ("Failed", html.escape(str(job.get("failed", "")))),
        ("Rows", html.escape(str(job.get("rows", "")))),
        ("Valid rows", html.escape(str(job.get("valid_rows", "")))),
        ("Started", html.escape(str(job.get("started_at", "")))),
        ("Finished", html.escape(str(job.get("finished_at", "")))),
        ("Updated", html.escape(str(job.get("updated_at", "")))),
    ]
    cards = "\n".join(
        f"<div class=\"item\"><div class=\"label\">{label}</div><div class=\"value\">{value}</div></div>"
        for label, value in fields
    )
    logs = "\n".join(html.escape(str(line)) for line in job.get("recent_logs", []))
    last_stock = html.escape(json.dumps(job.get("last_stock", {}), ensure_ascii=False, indent=2))
    title = html.escape(str(job.get("name") or job_id))
    return f"""<section>
  <h2>{title}</h2>
  <div class="grid">{cards}</div>
  <h3>Last stock</h3>
  <pre>{last_stock}</pre>
  <h3>Recent logs</h3>
  <pre>{logs}</pre>
</section>"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve Gushen data update status.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18088)
    parser.add_argument("--status-path", default=str(DEFAULT_STATUS_PATH))
    args = parser.parse_args(argv)
    StatusHandler.status_path = Path(args.status_path)
    server = ThreadingHTTPServer((args.host, args.port), StatusHandler)
    print(f"Serving Gushen data status on http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
