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
import plotly.graph_objects as go
import requests

from gushen.domestic_network import domestic_data_no_proxy
from gushen.top20_ma5_intraday_backtest import dynamic_ma5_boundary
from gushen.top20_ma5_pullback_strategy import DEFAULT_CACHE_DIR, choose_latest_paths_by_code


DEFAULT_BACKTEST_DIR = Path("reports/generated/top20_ma5_intraday_last_month")
DEFAULT_STATE_DIR = Path("data/local/live_monitor/top20_ma5")
EASTMONEY_QUOTE_URL = "https://push2.eastmoney.com/api/qt/ulist.np/get"
TENCENT_QUOTE_URL = "http://qt.gtimg.cn/q="
TENCENT_MINUTE_URL = "http://ifzq.gtimg.cn/appstock/app/kline/mkline"
LOCAL_PROXY = "http://127.0.0.1:7890"


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
        self.candidate_by_code = {item.code: item for item in self.candidates}
        self.last_quotes: dict[str, QuoteRow] = {}
        self.state = self._load_state()

    def snapshot(self) -> dict[str, Any]:
        quotes, quote_meta = fetch_quotes([item.code for item in self.candidates])
        quotes, quote_meta = self._with_cached_quotes(quotes, quote_meta, [item.code for item in self.candidates])
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
            pnl = estimate_pnl(candidate, quote, recommendation, event)
            rows.append(
                {
                    "candidate": asdict(candidate),
                    "quote": asdict(quote) if quote else None,
                    "event": event,
                    "recommendation": recommendation,
                    "pnl": pnl,
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

    def detail(self, code: str) -> dict[str, Any]:
        candidate = self.candidate_by_code.get(code)
        if candidate is None:
            raise KeyError(code)
        quotes, quote_meta = fetch_quotes([code])
        quotes, quote_meta = self._with_cached_quotes(quotes, quote_meta, [code])
        return {
            "candidate": asdict(candidate),
            "quote": asdict(quotes[code]) if code in quotes else None,
            "quote_meta": quote_meta,
            "daily": load_daily_chart_frame(self.daily_cache_dir, code, self.signal_date),
            "minute": fetch_tencent_minute_frame(code),
        }

    def _with_cached_quotes(
        self,
        quotes: dict[str, QuoteRow],
        quote_meta: dict[str, Any],
        codes: list[str],
    ) -> tuple[dict[str, QuoteRow], dict[str, Any]]:
        if quotes:
            self.last_quotes.update(quotes)

        missing = [code for code in codes if code not in quotes]
        cached = {code: self.last_quotes[code] for code in missing if code in self.last_quotes}
        if not cached:
            return quotes, quote_meta

        merged = dict(quotes)
        merged.update(cached)
        meta = dict(quote_meta)
        meta["cached_rows"] = len(cached)
        if len(cached) == len(codes):
            meta["source"] = "last_successful_quote"
            meta["stale"] = True
        else:
            meta["partial_stale"] = True
        return merged, meta

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
            if parsed.path == "/stock":
                query = parse_qs(parsed.query)
                code = str(query.get("code", [""])[0])
                try:
                    self._send_html(render_stock_detail(app.detail(code)))
                except KeyError:
                    self.send_error(404)
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
    error: Exception | None = None
    try:
        with domestic_data_no_proxy():
            response = requests.get(EASTMONEY_QUOTE_URL, params=params, headers=headers, timeout=12)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        error = exc
        try:
            response = requests.get(
                EASTMONEY_QUOTE_URL,
                params=params,
                headers=headers,
                timeout=12,
                proxies={"http": LOCAL_PROXY, "https": LOCAL_PROXY},
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as proxy_exc:
            fallback_quotes, fallback_meta = fetch_tencent_quotes(codes)
            fallback_meta["eastmoney_error"] = f"direct failed: {error}; proxy failed: {proxy_exc}"
            return fallback_quotes, fallback_meta
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
    meta: dict[str, Any] = {"source": "eastmoney_ulist", "rows": len(result)}
    if error is not None:
        meta["fallback"] = "proxy"
    return result, meta


def fetch_tencent_quotes(codes: list[str]) -> tuple[dict[str, QuoteRow], dict[str, Any]]:
    symbols = ",".join(tencent_symbol(code) for code in codes)
    try:
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(TENCENT_QUOTE_URL + symbols, headers={"User-Agent": "Mozilla/5.0"}, timeout=12, proxies={})
        response.raise_for_status()
    except Exception as exc:
        return {}, {"source": "tencent_quote", "rows": 0, "fallback": "tencent", "error": str(exc)}

    result = {}
    for line in response.text.split(";"):
        if "~" not in line:
            continue
        try:
            body = line.split('="', 1)[1].rsplit('"', 1)[0]
        except IndexError:
            continue
        parts = body.split("~")
        if len(parts) < 38:
            continue
        code = normalize_ts_code(parts[2])
        result[code] = QuoteRow(
            code=code,
            name=parts[1],
            latest=float_or_none(parts[3]),
            pct_change=float_or_none(parts[32]),
            amount=(float_or_none(parts[37]) or 0.0) * 10000,
            high=float_or_none(parts[33]),
            low=float_or_none(parts[34]),
            open=float_or_none(parts[5]),
            prev_close=float_or_none(parts[4]),
            source_time=tencent_source_time(parts[30]),
        )
    return result, {"source": "tencent_quote", "rows": len(result), "fallback": "tencent"}


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


def fetch_tencent_minute_frame(code: str) -> pd.DataFrame:
    symbol = tencent_symbol(code)
    params = {"param": f"{symbol},m1,,500"}
    try:
        with domestic_data_no_proxy():
            response = requests.get(TENCENT_MINUTE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        rows = ((data.get("data") or {}).get(symbol) or {}).get("m1", [])
    except Exception:
        return pd.DataFrame(columns=["time", "open", "close", "high", "low", "volume"])
    today = datetime.now().strftime("%Y%m%d")
    parsed = []
    for row in rows:
        if not row or not str(row[0]).startswith(today):
            continue
        stamp = str(row[0])
        parsed.append(
            {
                "time": f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[8:10]}:{stamp[10:12]}",
                "open": float_or_none(row[1]),
                "close": float_or_none(row[2]),
                "high": float_or_none(row[3]),
                "low": float_or_none(row[4]),
                "volume": float_or_none(row[5]),
            }
        )
    return pd.DataFrame(parsed).dropna(subset=["open", "close", "high", "low"])


def load_daily_chart_frame(daily_cache_dir: Path, code: str, signal_date: str) -> pd.DataFrame:
    paths = choose_latest_paths_by_code(daily_cache_dir)
    path = paths.get(code)
    if path is None:
        return pd.DataFrame()
    frame = read_daily_frame(path)
    frame["ma5"] = pd.to_numeric(frame["close"], errors="coerce").rolling(5).mean()
    start = (pd.Timestamp(signal_date) - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    return frame[(frame["trade_date"] >= start) & (frame["trade_date"] <= signal_date)].tail(80).copy()


def recommend_lot(price: float, per_position: float) -> dict[str, Any]:
    shares = int(per_position // (price * 100)) * 100 if price > 0 else 0
    amount = shares * price
    return {"shares": shares, "amount": amount}


def estimate_pnl(
    candidate: MonitorCandidate,
    quote: QuoteRow | None,
    recommendation: dict[str, Any],
    event: dict[str, Any],
) -> dict[str, float] | None:
    if event.get("status") not in {"triggered", "bought"}:
        return None
    if quote is None or quote.latest is None:
        return None
    shares = float(recommendation.get("shares") or 0)
    entry_price = float(event.get("trigger_price") or candidate.boundary_price)
    if shares <= 0 or entry_price <= 0:
        return None
    amount = (quote.latest - entry_price) * shares
    pct = (quote.latest / entry_price - 1.0) * 100
    return {"amount": amount, "pct": pct}


def render_page(snapshot: dict[str, Any]) -> str:
    rows = "\n".join(render_row(row) for row in snapshot["rows"])
    quote_meta = snapshot.get("quote_meta", {})
    notice = render_quote_notice(quote_meta)
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
    .alert {{ margin-bottom: 12px; background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; padding: 10px 12px; }}
    .notice {{ margin-bottom: 12px; background: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe; padding: 10px 12px; }}
  </style>
</head>
<body>
  <header>
    <h1>Top20 MA5 次日实时监控</h1>
    <div class="meta">信号日 {escape(snapshot['signal_date'])}；更新 {escape(snapshot['updated_at'])}；资金 {format_number(snapshot['capital'])}；已标记买入 {format_number(snapshot['used_amount'])}</div>
  </header>
  <main>{notice}<div class="wrap"><table>
    <thead><tr><th>排名</th><th>代码</th><th>名称</th><th>交界价</th><th>建议股数</th><th>建议金额</th><th>最新</th><th>当前盈亏</th><th>盈亏率</th><th>涨跌幅</th><th>日低</th><th>日高</th><th>成交额(亿)</th><th>行情时间</th><th>状态</th><th>操作</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div></main>
</body>
</html>"""


def render_row(row: dict[str, Any]) -> str:
    candidate = row["candidate"]
    quote = row["quote"] or {}
    event = row["event"] or {}
    rec = row["recommendation"]
    pnl = row["pnl"] or {}
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
    detail_href = f"/stock?code={code}"
    return (
        "<tr>"
        f"<td>{candidate['amount_rank']}</td>"
        f"<td><a href=\"{detail_href}\" target=\"_blank\">{code}</a></td>"
        f"<td><a href=\"{detail_href}\" target=\"_blank\">{escape(candidate['name'])}</a></td>"
        f"<td>{format_number(candidate['boundary_price'])}</td>"
        f"<td>{rec['shares']}</td>"
        f"<td>{format_number(rec['amount'])}</td>"
        f"<td>{format_number(quote.get('latest'))}</td>"
        f"<td>{format_signed(pnl.get('amount'))}</td>"
        f"<td>{format_signed(pnl.get('pct'), suffix='%')}</td>"
        f"<td>{format_number(quote.get('pct_change'))}%</td>"
        f"<td>{format_number(quote.get('low'))}</td>"
        f"<td>{format_number(quote.get('high'))}</td>"
        f"<td>{format_number((quote.get('amount') or 0) / 100000000)}</td>"
        f"<td>{escape(quote.get('source_time') or '')}</td>"
        f"<td><span class=\"badge {status_class}\">{escape(status)}</span></td>"
        f"<td><a href=\"/mark?code={code}&status=bought\">标记已买</a><a href=\"/mark?code={code}&status=skipped\">跳过</a><a href=\"/mark?code={code}&status=reset\">重置</a></td>"
        "</tr>"
    )


def render_quote_notice(quote_meta: dict[str, Any]) -> str:
    if quote_meta.get("stale"):
        rows = int(quote_meta.get("cached_rows") or 0)
        return f"<div class=\"notice\">当前行情接口暂未返回新数据，页面显示最后一次成功行情，共 {rows} 条。午休或收盘后这是正常情况。</div>"
    if quote_meta.get("partial_stale"):
        rows = int(quote_meta.get("cached_rows") or 0)
        return f"<div class=\"notice\">部分股票使用最后一次成功行情，共 {rows} 条。</div>"
    quote_error = quote_meta.get("error")
    if quote_error:
        return f"<div class=\"alert\">行情接口暂不可用：{escape(str(quote_error))}</div>"
    if quote_meta.get("fallback") == "tencent":
        return "<div class=\"notice\">EastMoney 暂不可用，当前使用腾讯行情兜底。</div>"
    if quote_meta.get("fallback") == "proxy":
        return "<div class=\"notice\">EastMoney 直连暂不可用，当前使用本机代理兜底。</div>"
    return ""


def render_stock_detail(payload: dict[str, Any]) -> str:
    candidate = payload["candidate"]
    quote = payload["quote"] or {}
    notice = render_quote_notice(payload.get("quote_meta", {}))
    daily_chart = chart_html(build_daily_figure(payload["daily"], candidate), include_plotly=True)
    minute_chart = chart_html(build_minute_figure(payload["minute"], candidate), include_plotly=False)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(candidate['code'])} {escape(candidate['name'])}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f8fafc; color: #111827; }}
    header {{ padding: 18px 26px; background: #111827; color: #fff; }}
    main {{ padding: 16px; max-width: 1280px; margin: 0 auto; }}
    section {{ background: white; border: 1px solid #e5e7eb; margin-bottom: 16px; padding: 14px; }}
    h1, h2 {{ margin: 0 0 8px; }}
    .meta {{ color: #cbd5e1; }}
    .alert {{ margin-bottom: 12px; background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; padding: 10px 12px; }}
    .notice {{ margin-bottom: 12px; background: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe; padding: 10px 12px; }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(candidate['code'])} {escape(candidate['name'])}</h1>
    <div class="meta">交界价 {format_number(candidate['boundary_price'])}；最新 {format_number(quote.get('latest'))}；行情时间 {escape(quote.get('source_time') or '')}</div>
  </header>
  <main>
    {notice}
    <section><h2>日K线</h2>{daily_chart}</section>
    <section><h2>今日1分钟走势</h2>{minute_chart}</section>
  </main>
</body>
</html>"""


def build_daily_figure(frame: pd.DataFrame, candidate: dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    if not frame.empty:
        fig.add_trace(
            go.Candlestick(
                x=frame["trade_date"],
                open=frame["open"],
                high=frame["high"],
                low=frame["low"],
                close=frame["close"],
                name="日K",
            )
        )
        fig.add_trace(go.Scatter(x=frame["trade_date"], y=frame["ma5"], mode="lines", name="MA5"))
    fig.add_hline(y=float(candidate["boundary_price"]), line_dash="dot", line_color="#16a34a", annotation_text="交界价")
    fig.update_layout(height=420, margin={"l": 30, "r": 20, "t": 20, "b": 30}, xaxis_rangeslider_visible=False, template="plotly_white")
    return fig


def build_minute_figure(frame: pd.DataFrame, candidate: dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    if not frame.empty:
        fig.add_trace(
            go.Candlestick(
                x=frame["time"],
                open=frame["open"],
                high=frame["high"],
                low=frame["low"],
                close=frame["close"],
                name="1分钟",
            )
        )
    fig.add_hline(y=float(candidate["boundary_price"]), line_dash="dot", line_color="#16a34a", annotation_text="交界价")
    fig.update_layout(height=420, margin={"l": 30, "r": 20, "t": 20, "b": 30}, xaxis_rangeslider_visible=False, template="plotly_white")
    return fig


def chart_html(fig: go.Figure, include_plotly: bool) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if include_plotly else False, config={"displayModeBar": False})


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


def tencent_source_time(value: str) -> str:
    if len(value) != 14 or not value.isdigit():
        return value
    return f"{value[:4]}-{value[4:6]}-{value[6:8]} {value[8:10]}:{value[10:12]}:{value[12:14]}"


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


def format_signed(value: Any, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    sign = "+" if number > 0 else ""
    return f"{sign}{number:,.2f}{suffix}"


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
