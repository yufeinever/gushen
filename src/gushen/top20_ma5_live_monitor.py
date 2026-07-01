from __future__ import annotations

import argparse
import html
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests

from gushen.domestic_network import domestic_data_no_proxy
from gushen.top20_ma5_intraday_backtest import dynamic_ma5_boundary
from gushen.top20_ma5_pullback_strategy import DEFAULT_CACHE_DIR, choose_latest_paths_by_code


DEFAULT_BACKTEST_DIR = Path("reports/generated/top20_ma5_intraday_last_month")
DEFAULT_STATE_DIR = Path("data/local/live_monitor/top20_ma5")
EASTMONEY_QUOTE_URL = "https://push2.eastmoney.com/api/qt/ulist.np/get"
TENCENT_MINUTE_URL = "http://ifzq.gtimg.cn/appstock/app/kline/mkline"


@dataclass(frozen=True)
class MonitorCandidate:
    signal_date: str
    code: str
    name: str
    amount_rank: int
    close: float
    signal_ma5: float
    boundary_price: float


@dataclass(frozen=True)
class QuoteRow:
    code: str
    name: str
    latest: float | None
    pct_change: float | None
    amount: float | None
    high: float | None
    low: float | None
    open: float | None
    prev_close: float | None
    source_time: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live monitor for Top20 MA5 next-day boundary entries.")
    parser.add_argument("--backtest-dir", type=Path, default=DEFAULT_BACKTEST_DIR)
    parser.add_argument("--daily-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--signal-date", default=None)
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--per-position", type=float, default=20_000)
    parser.add_argument("--refresh-seconds", type=int, default=8)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signal_date = args.signal_date or infer_latest_signal_date(args.backtest_dir)
    app = MonitorApp(
        backtest_dir=args.backtest_dir,
        daily_cache_dir=args.daily_cache_dir,
        state_dir=args.state_dir,
        signal_date=signal_date,
        capital=args.capital,
        per_position=args.per_position,
        refresh_seconds=args.refresh_seconds,
    )
    handler = make_handler(app)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(json.dumps({"url": f"http://{args.host}:{args.port}", "signal_date": signal_date}, ensure_ascii=False))
    server.serve_forever()


class MonitorApp:
    def __init__(
        self,
        backtest_dir: Path,
        daily_cache_dir: Path,
        state_dir: Path,
        signal_date: str,
        capital: float,
        per_position: float,
        refresh_seconds: int,
    ) -> None:
        self.backtest_dir = backtest_dir
        self.daily_cache_dir = daily_cache_dir
        self.state_dir = state_dir
        self.signal_date = signal_date
        self.capital = capital
        self.per_position = per_position
        self.refresh_seconds = refresh_seconds
        self.state_path = state_dir / f"{signal_date}.json"
        self.candidates = load_monitor_candidates(backtest_dir, daily_cache_dir, signal_date)
        self.state = self._load_state()

    def snapshot(self) -> dict[str, Any]:
        quotes, quote_meta = fetch_quotes([item.code for item in self.candidates])
        now = datetime.now().isoformat(timespec="seconds")
        changed = False
        rows = []
        used_amount = self._used_amount()
        for candidate in self.candidates:
            quote = quotes.get(candidate.code)
            event = self.state["events"].setdefault(candidate.code, {})
            trigger = detect_trigger(candidate, quote)
            if trigger and "triggered_at" not in event:
                triggered_at = quote.source_time if quote and quote.source_time else now
                event.update(
                    {
                        "status": "triggered",
                        "triggered_at": triggered_at,
                        "trigger_price": round(candidate.boundary_price, 4),
                    }
                )
                changed = True
            recommendation = recommend_lot(candidate.boundary_price, self.per_position)
            if event.get("status") == "bought":
                status = "已标记买入"
            elif event.get("status") == "skipped":
                status = "已标记跳过"
            elif event.get("status") == "triggered":
                if used_amount + recommendation["amount"] <= self.capital + 1e-6:
                    status = "已触发，待手动确认"
                else:
                    status = "已触发，资金上限"
            else:
                status = "待触发"
            rows.append(
                {
                    "candidate": asdict(candidate),
                    "quote": asdict(quote) if quote else None,
                    "event": event,
                    "recommendation": recommendation,
                    "status": status,
                }
            )
        if changed:
            self._save_state()
        return {
            "signal_date": self.signal_date,
            "monitor_date": now[:10],
            "updated_at": now,
            "refresh_seconds": self.refresh_seconds,
            "capital": self.capital,
            "per_position": self.per_position,
            "used_amount": used_amount,
            "quote_meta": quote_meta,
            "rows": rows,
        }

    def mark(self, code: str, status: str) -> None:
        if status not in {"bought", "skipped", "reset"}:
            return
        events = self.state["events"]
        if status == "reset":
            events.pop(code, None)
        else:
            event = events.setdefault(code, {})
            event["status"] = status
            event["marked_at"] = datetime.now().isoformat(timespec="seconds")
        self._save_state()

    def _used_amount(self) -> float:
        total = 0.0
        for candidate in self.candidates:
            event = self.state.get("events", {}).get(candidate.code, {})
            if event.get("status") == "bought":
                total += recommend_lot(candidate.boundary_price, self.per_position)["amount"]
        return total

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"signal_date": self.signal_date, "events": {}}

    def _save_state(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")


def make_handler(app: MonitorApp) -> type[BaseHTTPRequestHandler]:
    class MonitorHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/status":
                self._send_json(app.snapshot())
                return
            if parsed.path == "/mark":
                query = parse_qs(parsed.query)
                app.mark(str(query.get("code", [""])[0]), str(query.get("status", [""])[0]))
                self.send_response(302)
                self.send_header("Location", "/")
                self.end_headers()
                return
            if parsed.path in {"/", "/index.html"}:
                self._send_html(render_page(app.snapshot()))
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

    return MonitorHandler


def load_monitor_candidates(backtest_dir: Path, daily_cache_dir: Path, signal_date: str) -> list[MonitorCandidate]:
    candidates = pd.read_csv(backtest_dir / "top20_ma5_candidates.csv")
    selected = candidates[candidates["signal_date"].astype(str) == signal_date].copy()
    if selected.empty:
        raise RuntimeError(f"no candidates found for signal date {signal_date}")
    paths = choose_latest_paths_by_code(daily_cache_dir)
    rows: list[MonitorCandidate] = []
    for row in selected.sort_values(["amount_rank", "code"]).to_dict("records"):
        path = paths.get(str(row["code"]))
        if path is None:
            continue
        frame = read_daily_frame(path)
        matches = frame.index[frame["trade_date"] == signal_date].tolist()
        if not matches:
            continue
        boundary = dynamic_ma5_boundary(frame, matches[0], 5)
        if boundary is None:
            continue
        rows.append(
            MonitorCandidate(
                signal_date=signal_date,
                code=str(row["code"]),
                name=str(row["name"]),
                amount_rank=int(row["amount_rank"]),
                close=float(row["close"]),
                signal_ma5=float(row["ma5"]),
                boundary_price=boundary,
            )
        )
    return rows


def read_daily_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d")
    for column in ["open", "high", "low", "close", "amount"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("trade_date").reset_index(drop=True)


def infer_latest_signal_date(backtest_dir: Path) -> str:
    candidates = pd.read_csv(backtest_dir / "top20_ma5_candidates.csv")
    return str(candidates["signal_date"].max())


def fetch_quotes(codes: list[str]) -> tuple[dict[str, QuoteRow], dict[str, Any]]:
    if not codes:
        return {}, {"source": "none", "rows": 0}
    secids = ",".join(eastmoney_secid(code) for code in codes)
    params = {
        "secids": secids,
        "fields": "f12,f14,f2,f3,f4,f5,f6,f15,f16,f17,f18,f124",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
    }
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
    with domestic_data_no_proxy():
        response = requests.get(EASTMONEY_QUOTE_URL, params=params, headers=headers, timeout=12)
    response.raise_for_status()
    payload = response.json()
    rows = (payload.get("data") or {}).get("diff", [])
    result = {}
    for row in rows:
        code = normalize_ts_code(str(row.get("f12") or ""))
        result[code] = QuoteRow(
            code=code,
            name=str(row.get("f14") or ""),
            latest=float_or_none(row.get("f2")),
            pct_change=float_or_none(row.get("f3")),
            amount=float_or_none(row.get("f6")),
            high=float_or_none(row.get("f15")),
            low=float_or_none(row.get("f16")),
            open=float_or_none(row.get("f17")),
            prev_close=float_or_none(row.get("f18")),
            source_time=source_time(int_or_none(row.get("f124"))),
        )
    return result, {"source": "eastmoney_ulist", "rows": len(result)}


def detect_trigger(candidate: MonitorCandidate, quote: QuoteRow | None) -> bool:
    if quote is None or quote.low is None or quote.high is None:
        return False
    return quote.low <= candidate.boundary_price <= quote.high


def find_first_trigger_minute(candidate: MonitorCandidate) -> str | None:
    symbol = tencent_symbol(candidate.code)
    params = {"param": f"{symbol},m1,,500"}
    try:
        with domestic_data_no_proxy():
            response = requests.get(TENCENT_MINUTE_URL, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        rows = ((data.get("data") or {}).get(symbol) or {}).get("m1", [])
    except Exception:
        return None
    today = datetime.now().strftime("%Y%m%d")
    for row in rows:
        if not row or not str(row[0]).startswith(today):
            continue
        high = float_or_none(row[3])
        low = float_or_none(row[4])
        if high is not None and low is not None and low <= candidate.boundary_price <= high:
            stamp = str(row[0])
            return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[8:10]}:{stamp[10:12]}"
    return None


def recommend_lot(price: float, per_position: float) -> dict[str, Any]:
    shares = int(per_position // (price * 100)) * 100 if price > 0 else 0
    amount = shares * price
    return {"shares": shares, "amount": amount}


def render_page(snapshot: dict[str, Any]) -> str:
    rows = "\n".join(render_row(row) for row in snapshot["rows"])
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="{int(snapshot['refresh_seconds'])}">
  <title>Top20 MA5 实时监控</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f8fafc; color: #111827; }}
    header {{ padding: 20px 28px; background: #111827; color: white; }}
    main {{ padding: 18px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; min-width: 1320px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 14px; white-space: nowrap; }}
    th {{ background: #eef2f7; color: #334155; }}
    .wrap {{ overflow-x: auto; border: 1px solid #e5e7eb; }}
    .badge {{ display: inline-block; border-radius: 999px; padding: 4px 10px; }}
    .waiting {{ background: #fef3c7; color: #92400e; }}
    .triggered {{ background: #dbeafe; color: #1d4ed8; }}
    .bought {{ background: #dcfce7; color: #166534; }}
    .skipped {{ background: #f1f5f9; color: #475569; }}
    a {{ color: #2563eb; text-decoration: none; margin-right: 8px; }}
    .meta {{ color: #cbd5e1; margin-top: 6px; }}
  </style>
</head>
<body>
  <header>
    <h1>Top20 MA5 次日实时监控</h1>
    <div class="meta">信号日 {escape(snapshot['signal_date'])}；更新 {escape(snapshot['updated_at'])}；资金 {format_number(snapshot['capital'])}；已标记买入 {format_number(snapshot['used_amount'])}</div>
  </header>
  <main><div class="wrap"><table>
    <thead><tr><th>排名</th><th>代码</th><th>名称</th><th>交界价</th><th>建议股数</th><th>建议金额</th><th>最新</th><th>涨跌幅</th><th>日低</th><th>日高</th><th>成交额(亿)</th><th>行情时间</th><th>状态</th><th>操作</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div></main>
</body>
</html>"""


def render_row(row: dict[str, Any]) -> str:
    candidate = row["candidate"]
    quote = row["quote"] or {}
    event = row["event"] or {}
    rec = row["recommendation"]
    status = str(row["status"])
    status_class = "waiting"
    if "已触发" in status:
        status_class = "triggered"
    if "已标记买入" in status:
        status_class = "bought"
    if "跳过" in status:
        status_class = "skipped"
    detail = event.get("triggered_at")
    if detail:
        status = f"{status}；{detail}"
    code = escape(candidate["code"])
    return (
        "<tr>"
        f"<td>{candidate['amount_rank']}</td>"
        f"<td>{code}</td>"
        f"<td>{escape(candidate['name'])}</td>"
        f"<td>{format_number(candidate['boundary_price'])}</td>"
        f"<td>{rec['shares']}</td>"
        f"<td>{format_number(rec['amount'])}</td>"
        f"<td>{format_number(quote.get('latest'))}</td>"
        f"<td>{format_number(quote.get('pct_change'))}%</td>"
        f"<td>{format_number(quote.get('low'))}</td>"
        f"<td>{format_number(quote.get('high'))}</td>"
        f"<td>{format_number((quote.get('amount') or 0) / 100000000)}</td>"
        f"<td>{escape(quote.get('source_time') or '')}</td>"
        f"<td><span class=\"badge {status_class}\">{escape(status)}</span></td>"
        f"<td><a href=\"/mark?code={code}&status=bought\">标记已买</a><a href=\"/mark?code={code}&status=skipped\">跳过</a><a href=\"/mark?code={code}&status=reset\">重置</a></td>"
        "</tr>"
    )


def eastmoney_secid(code: str) -> str:
    raw, market = code.split(".")
    return ("1." if market == "SH" else "0.") + raw


def tencent_symbol(code: str) -> str:
    raw, market = code.split(".")
    return ("sh" if market == "SH" else "sz") + raw


def normalize_ts_code(raw: str) -> str:
    if raw.startswith("6"):
        return f"{raw}.SH"
    return f"{raw}.SZ"


def source_time(value: int | None) -> str:
    if not value:
        return ""
    return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def float_or_none(value: Any) -> float | None:
    try:
        if value in {None, "", "-"}:
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def format_number(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
