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
    rows = "\n".join(render_job_row(job_id, job) for job_id, job in sorted(jobs.items()))
    details = "\n".join(render_job_detail(job_id, job) for job_id, job in sorted(jobs.items()))
    if not rows:
        rows = """<tr><td colspan="9" class="empty">暂无数据更新记录</td></tr>"""
        details = ""
    updated_at = html.escape(str(status.get("updated_at") or "从未更新"))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="10">
  <title>股神数据更新状态</title>
  <style>
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      background: #f4f6f8;
      color: #1f2937;
    }}
    header {{
      background: #172033;
      color: #fff;
      padding: 18px 28px 16px;
      border-bottom: 1px solid #263348;
    }}
    main {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 22px;
    }}
    section {{
      background: #fff;
      border: 1px solid #d9e0ea;
      border-radius: 8px;
      margin-bottom: 18px;
      overflow: hidden;
    }}
    h1 {{
      margin: 0;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0;
      padding: 15px 18px;
      font-size: 17px;
      border-bottom: 1px solid #e5e7eb;
      background: #fbfcfd;
    }}
    h3 {{
      margin: 0 0 10px;
      font-size: 15px;
    }}
    .meta {{
      color: #cbd5e1;
      margin-top: 7px;
      font-size: 13px;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }}
    th, td {{
      border-bottom: 1px solid #edf0f5;
      padding: 12px 14px;
      text-align: left;
      vertical-align: top;
      font-size: 14px;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }}
    th {{
      background: #f7f9fb;
      color: #475467;
      font-weight: 600;
      white-space: nowrap;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    .job-name {{
      font-weight: 700;
      color: #111827;
    }}
    .subtle {{
      color: #667085;
      font-size: 12px;
      margin-top: 3px;
    }}
    .status {{
      display: inline-block;
      border-radius: 999px;
      padding: 3px 9px;
      font-size: 13px;
      background: #e5e7eb;
      color: #374151;
      white-space: nowrap;
    }}
    .status.success, .status.cached {{ background: #dcfce7; color: #166534; }}
    .status.running {{ background: #dbeafe; color: #1d4ed8; }}
    .status.failed {{ background: #fee2e2; color: #991b1b; }}
    .status.dry_run {{ background: #fef3c7; color: #92400e; }}
    .progress {{
      height: 8px;
      border-radius: 999px;
      background: #e5e7eb;
      overflow: hidden;
      margin-top: 8px;
    }}
    .progress-bar {{
      height: 100%;
      background: #2563eb;
    }}
    .detail {{
      padding: 16px 18px 18px;
      border-top: 1px solid #edf0f5;
    }}
    .detail + .detail {{
      border-top: 1px solid #d9e0ea;
    }}
    .detail-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}
    .detail-item {{
      background: #f8fafc;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      padding: 10px;
      min-height: 58px;
    }}
    .label {{
      color: #667085;
      font-size: 12px;
      margin-bottom: 4px;
    }}
    .value {{
      font-size: 14px;
      color: #111827;
      overflow-wrap: anywhere;
    }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #111827;
      color: #e5e7eb;
      border-radius: 6px;
      padding: 12px;
      max-height: 260px;
      overflow: auto;
      margin: 8px 0 0;
      font-size: 12px;
    }}
    details summary {{
      cursor: pointer;
      color: #2563eb;
      font-size: 14px;
    }}
    .empty {{
      text-align: center;
      color: #667085;
      padding: 30px;
    }}
    @media (max-width: 760px) {{
      header {{ padding: 16px; }}
      main {{ padding: 14px; }}
      th, td {{ padding: 10px; font-size: 13px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>股神数据更新状态</h1>
    <div class="meta">状态最后写入：{updated_at}；页面每 10 秒自动刷新。</div>
  </header>
  <main>
    <section>
      <h2>今日更新列表</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th style="width: 170px;">更新任务</th>
              <th style="width: 240px;">更新内容</th>
              <th style="width: 96px;">状态</th>
              <th style="width: 160px;">进度</th>
              <th style="width: 150px;">数据量</th>
              <th style="width: 120px;">异常</th>
              <th style="width: 150px;">时间</th>
              <th>输出位置</th>
              <th style="width: 160px;">最近处理</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
      </div>
    </section>
    <section>
      <h2>任务明细</h2>
      {details}
    </section>
  </main>
</body>
</html>"""


def render_job_row(job_id: str, job: dict[str, Any]) -> str:
    progress = job_progress(job)
    status = str(job.get("status", "unknown"))
    output_path = job.get("output_path") or job.get("cache_dir") or job.get("pool_file") or ""
    last_stock = job.get("last_stock") or {}
    last_text = ""
    if last_stock:
        last_text = f"{last_stock.get('index', '')} {last_stock.get('ts_code', '')} {last_stock.get('name', '')}"
    started = str(job.get("started_at") or "")
    finished = str(job.get("finished_at") or "")
    time_text = html.escape(started)
    if finished:
        time_text += f"<div class=\"subtle\">完成：{html.escape(finished)}</div>"
    return f"""<tr>
  <td><div class="job-name">{html.escape(job_title(job_id, job))}</div><div class="subtle">{html.escape(str(job.get("trade_date", "")))}</div></td>
  <td>{html.escape(job_content(job_id, job))}</td>
  <td>{render_status(status)}</td>
  <td>{html.escape(progress)}<div class="progress"><div class="progress-bar" style="width: {html.escape(progress_width(job))};"></div></div></td>
  <td>{html.escape(data_volume(job))}</td>
  <td>{html.escape(error_summary(job))}</td>
  <td>{time_text}</td>
  <td>{html.escape(str(output_path))}</td>
  <td>{html.escape(last_text)}</td>
</tr>"""


def render_job_detail(job_id: str, job: dict[str, Any]) -> str:
    logs = "\n".join(html.escape(str(line)) for line in job.get("recent_logs", []))
    last_stock = html.escape(json.dumps(job.get("last_stock", {}), ensure_ascii=False, indent=2))
    fields = [
        ("任务编号", job_id),
        ("任务名称", job_title(job_id, job)),
        ("交易日期", job.get("trade_date", "")),
        ("状态", status_label(str(job.get("status", "unknown")))),
        ("进度", job_progress(job)),
        ("请求数量", job.get("requested", "")),
        ("已处理", job.get("processed", "")),
        ("下载成功", job.get("downloaded", "")),
        ("缓存跳过", job.get("skipped_cached", "")),
        ("失败", job.get("failed", "")),
        ("有效行数", job.get("valid_rows", "")),
        ("输出文件", job.get("output_path") or job.get("manifest_path") or ""),
        ("开始时间", job.get("started_at", "")),
        ("完成时间", job.get("finished_at", "")),
        ("更新时间", job.get("updated_at", "")),
    ]
    items = "\n".join(
        f"""<div class="detail-item"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(str(value))}</div></div>"""
        for label, value in fields
    )
    return f"""<div class="detail">
  <h3>{html.escape(job_title(job_id, job))}</h3>
  <div class="detail-grid">{items}</div>
  <details>
    <summary>展开最近处理日志</summary>
    <pre>{logs}</pre>
  </details>
  <details>
    <summary>展开最近处理股票</summary>
    <pre>{last_stock}</pre>
  </details>
</div>"""


def job_title(job_id: str, job: dict[str, Any]) -> str:
    titles = {
        "daily_spot": "每日全市场日线",
        "weekly_qfq_full_refresh": "每周复权全量刷新",
    }
    return titles.get(job_id, str(job.get("name") or job_id))


def job_content(job_id: str, job: dict[str, Any]) -> str:
    if job_id == "daily_spot":
        return "收盘后通过 AKShare 抓取全市场当日 raw 日线快照"
    if job_id == "weekly_qfq_full_refresh":
        workers = job.get("workers", "")
        return f"全市场前复权 qfq 历史数据重拉，workers={workers}"
    return str(job.get("note") or job.get("name") or job_id)


def status_label(status: str) -> str:
    labels = {
        "success": "成功",
        "running": "运行中",
        "failed": "失败",
        "cached": "已缓存",
        "dry_run": "演练",
        "unknown": "未知",
    }
    return labels.get(status, status)


def render_status(status: str) -> str:
    return f"""<span class="status {html.escape(status)}">{html.escape(status_label(status))}</span>"""


def job_progress(job: dict[str, Any]) -> str:
    if job.get("progress_pct") is not None:
        return f"{job.get('progress_pct')}%"
    processed = job.get("processed")
    requested = job.get("requested")
    if processed is not None and requested:
        return f"{processed}/{requested}"
    rows = job.get("rows")
    if rows is not None:
        return f"{rows} 行"
    return "-"


def progress_width(job: dict[str, Any]) -> str:
    value = job.get("progress_pct")
    if value is None and job.get("status") in {"success", "cached", "dry_run"}:
        value = 100
    try:
        pct = max(0.0, min(float(value), 100.0))
    except (TypeError, ValueError):
        pct = 0.0
    return f"{pct:.1f}%"


def data_volume(job: dict[str, Any]) -> str:
    if job.get("rows") is not None:
        return f"总行 {job.get('rows')} / 有效 {job.get('valid_rows', 0)}"
    requested = job.get("requested", "")
    processed = job.get("processed", "")
    downloaded = job.get("downloaded", "")
    skipped = job.get("skipped_cached", "")
    return f"请求 {requested} / 已处理 {processed} / 下载 {downloaded} / 跳过 {skipped}"


def error_summary(job: dict[str, Any]) -> str:
    failed = job.get("failed", 0) or 0
    empty = job.get("empty", 0) or 0
    if failed or empty:
        return f"失败 {failed} / 空数据 {empty}"
    return "无"


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
