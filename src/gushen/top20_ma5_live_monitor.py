from __future__ import annotations

import argparse
import html
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import plotly.graph_objects as go
import requests

from gushen.domestic_network import domestic_data_no_proxy
from gushen.top20_ma5_intraday_backtest import dynamic_ma5_boundary
from gushen.top20_ma5_pullback_strategy import (
    DEFAULT_CACHE_DIR,
    choose_latest_paths_by_code,
    infer_latest_cache_date,
    is_excluded_board,
    is_st_name,
)


DEFAULT_BACKTEST_DIR = Path("reports/generated/top20_ma5_intraday_last_month")
DEFAULT_STATE_DIR = Path("data/local/live_monitor/top20_ma5")
EASTMONEY_QUOTE_URL = "https://push2.eastmoney.com/api/qt/ulist.np/get"
TENCENT_QUOTE_URL = "http://qt.gtimg.cn/q="
TENCENT_MINUTE_URL = "http://ifzq.gtimg.cn/appstock/app/kline/mkline"
LOCAL_PROXY = "http://127.0.0.1:7890"


@dataclass(frozen=True)
class MonitorCandidate:
    signal_date: str
    signal_dates: tuple[str, ...]
    amount_ranks: tuple[int, ...]
    monitor_date: str
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
    parser = argparse.ArgumentParser(description="Live monitor for Top20 MA5 morning boundary entries.")
    parser.add_argument("--backtest-dir", type=Path, default=DEFAULT_BACKTEST_DIR)
    parser.add_argument("--daily-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--signal-date", default=None)
    parser.add_argument("--monitor-date", default=None)
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--per-position", type=float, default=10_000)
    parser.add_argument("--refresh-seconds", type=int, default=8)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signal_date = args.signal_date or infer_latest_signal_date(args.backtest_dir)
    monitor_date = args.monitor_date or date.today().isoformat()
    app = MonitorApp(
        backtest_dir=args.backtest_dir,
        daily_cache_dir=args.daily_cache_dir,
        state_dir=args.state_dir,
        signal_date=signal_date,
        monitor_date=monitor_date,
        capital=args.capital,
        per_position=args.per_position,
        refresh_seconds=args.refresh_seconds,
    )
    handler = make_handler(app)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(json.dumps({"url": f"http://{args.host}:{args.port}", "monitor_date": monitor_date}, ensure_ascii=False))
    server.serve_forever()


class MonitorApp:
    def __init__(
        self,
        backtest_dir: Path,
        daily_cache_dir: Path,
        state_dir: Path,
        signal_date: str,
        monitor_date: str,
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
        self._load_monitor_date(monitor_date)

    def _load_monitor_date(self, monitor_date: str) -> None:
        self.monitor_date = monitor_date
        self.state_path = self.state_dir / f"{monitor_date}.json"
        self.candidates = load_monitor_candidates(self.backtest_dir, self.daily_cache_dir, self.signal_date, monitor_date)
        self.candidate_by_code = {item.code: item for item in self.candidates}
        self.last_quotes: dict[str, QuoteRow] = {}
        self.signal_decisions_cache: dict[str, list[dict[str, Any]]] = {}
        self.state = self._load_state()

    def set_monitor_date(self, monitor_date: str) -> None:
        if monitor_date != self.monitor_date:
            self._load_monitor_date(monitor_date)

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
            recommendation = recommend_lot(candidate.boundary_price, self.per_position)
            if recommendation["shares"] <= 0 and event.get("status") == "triggered":
                event.pop("status", None)
                event.pop("triggered_at", None)
                event.pop("trigger_price", None)
                changed = True
            trigger_time = find_morning_trigger_minute(candidate) if recommendation["shares"] > 0 else None
            trigger = trigger_time is not None
            if trigger and "triggered_at" not in event:
                event.update(
                    {
                        "status": "triggered",
                        "triggered_at": trigger_time,
                        "trigger_price": round(candidate.boundary_price, 4),
                    }
                )
                changed = True
            if recommendation["shares"] <= 0:
                status = "价格>110，跳过"
            elif event.get("status") == "bought":
                status = "已标记买入"
            elif event.get("status") == "skipped":
                status = "已标记跳过"
            elif event.get("status") == "triggered":
                if used_amount + recommendation["amount"] <= self.capital + 1e-6:
                    status = "已触发，待手动确认"
                else:
                    status = "已触发，资金上限"
            elif quote is not None and quote.latest is not None and quote.latest < candidate.boundary_price:
                status = "当前低于MA5，跳过"
            else:
                status = morning_status(candidate.monitor_date, trigger_time, now)
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
            "monitor_date": self.monitor_date,
            "decision_date": previous_trade_dates(self.monitor_date, 1)[0],
            "updated_at": now,
            "refresh_seconds": self.refresh_seconds,
            "capital": self.capital,
            "per_position": self.per_position,
            "used_amount": used_amount,
            "quote_meta": quote_meta,
            "signal_decisions": self._signal_decisions(previous_trade_dates(self.monitor_date, 1)[0]),
            "rows": rows,
        }

    def _signal_decisions(self, signal_date: str) -> list[dict[str, Any]]:
        cached = self.signal_decisions_cache.get(signal_date)
        if cached is None:
            cache_path = self.state_dir / f"signal_decisions_{signal_date}.json"
            if cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                cached = load_signal_decisions(self.daily_cache_dir, signal_date)
                self.state_dir.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8")
            self.signal_decisions_cache[signal_date] = cached
        return cached

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
            "daily": load_daily_chart_frame(self.daily_cache_dir, code, self.monitor_date),
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
        return {"monitor_date": self.monitor_date, "events": {}}

    def _save_state(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")


def make_handler(app: MonitorApp) -> type[BaseHTTPRequestHandler]:
    class MonitorHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            selected_date = normalize_date_query(str(query.get("date", [app.monitor_date])[0]), app.monitor_date)
            app.set_monitor_date(selected_date)
            if parsed.path == "/api/status":
                self._send_json(app.snapshot())
                return
            if parsed.path == "/mark":
                app.mark(str(query.get("code", [""])[0]), str(query.get("status", [""])[0]))
                self.send_response(302)
                self.send_header("Location", f"/?date={selected_date}")
                self.end_headers()
                return
            if parsed.path == "/stock":
                code = str(query.get("code", [""])[0])
                try:
                    self._send_html(render_stock_detail(app.detail(code)))
                except KeyError:
                    self.send_error(404)
                return
            if parsed.path == "/decision":
                self._send_html(render_decision_page(app))
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


def load_monitor_candidates(
    backtest_dir: Path,
    daily_cache_dir: Path,
    signal_date: str,
    monitor_date: str,
) -> list[MonitorCandidate]:
    candidates = pd.read_csv(backtest_dir / "top20_ma5_candidates.csv")
    signal_dates = previous_trade_dates(monitor_date, 3)
    selected = candidates[candidates["signal_date"].astype(str).isin(signal_dates)].copy()
    if selected.empty:
        raise RuntimeError(f"no candidates found for monitor date {monitor_date}")
    paths = choose_latest_paths_by_code(daily_cache_dir)
    rows: list[MonitorCandidate] = []
    for code, group in selected.sort_values(["signal_date", "amount_rank", "code"]).groupby("code"):
        path = paths.get(str(code))
        if path is None:
            continue
        frame = read_daily_frame(path)
        frame["ma5"] = pd.to_numeric(frame["close"], errors="coerce").rolling(5).mean()
        valid_group = group[
            group["signal_date"].astype(str).map(
                lambda item: signal_survives_ma5_observation(frame, item, monitor_date)
            )
        ].copy()
        if valid_group.empty:
            continue
        boundary = current_dynamic_ma5_boundary(frame, monitor_date)
        if boundary is None:
            continue
        first = valid_group.sort_values(["signal_date", "amount_rank"]).iloc[0]
        signal_date_items = tuple(str(item) for item in valid_group["signal_date"].astype(str).tolist())
        amount_rank_items = tuple(int(item) for item in valid_group["amount_rank"].tolist())
        rows.append(
            MonitorCandidate(
                signal_date=signal_date_items[0],
                signal_dates=signal_date_items,
                amount_ranks=amount_rank_items,
                monitor_date=monitor_date,
                code=str(code),
                name=str(first["name"]),
                amount_rank=min(amount_rank_items),
                close=float(first["close"]),
                signal_ma5=float(first["ma5"]),
                boundary_price=boundary,
            )
        )
    return sorted(rows, key=lambda item: (item.amount_rank, item.code))


def signal_survives_ma5_observation(frame: pd.DataFrame, signal_date: str, monitor_date: str) -> bool:
    observation = frame[(frame["trade_date"] > signal_date) & (frame["trade_date"] < monitor_date)].copy()
    if observation.empty:
        return True
    close = pd.to_numeric(observation["close"], errors="coerce")
    ma5 = pd.to_numeric(observation["ma5"], errors="coerce")
    return not bool((close < ma5).any())


def load_signal_decisions(daily_cache_dir: Path, signal_date: str, top_n: int = 20) -> list[dict[str, Any]]:
    paths = choose_latest_paths_by_code(daily_cache_dir)
    rows = []
    for code, path in paths.items():
        frame = read_recent_daily_frame(path, min_rows=12)
        match = frame[frame["trade_date"] == signal_date]
        if match.empty:
            continue
        frame["ma5"] = pd.to_numeric(frame["close"], errors="coerce").rolling(5).mean()
        row = frame.loc[match.index[0]]
        amount = float_or_none(row.get("amount"))
        if amount is None or amount <= 0:
            continue
        close = float_or_none(row.get("close"))
        ma5 = float_or_none(row.get("ma5"))
        name = str(row.get("name") or "")
        if is_excluded_board(code) or is_st_name(name):
            continue
        if close is None or ma5 is None or close <= ma5:
            decision = "剔除：未站上MA5"
        else:
            decision = "入选观察"
        rows.append(
            {
                "amount": amount,
                "code": code,
                "name": name,
                "close": close,
                "ma5": ma5,
                "decision": decision,
            }
        )
    rows = sorted(rows, key=lambda item: item["amount"], reverse=True)[:top_n]
    for index, row in enumerate(rows, start=1):
        row["amount_rank"] = index
    return rows


def read_recent_daily_frame(path: Path, min_rows: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if len(frame) > min_rows:
        frame = frame.tail(min_rows).copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d")
    for column in ["open", "high", "low", "close", "amount"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("trade_date").reset_index(drop=True)


def previous_trade_dates(day: str, count: int) -> list[str]:
    current = datetime.strptime(day, "%Y-%m-%d").date()
    result: list[str] = []
    probe = current - timedelta(days=1)
    while len(result) < count:
        if probe.weekday() < 5:
            result.append(probe.isoformat())
        probe -= timedelta(days=1)
    return result


def normalize_date_query(value: str, fallback: str) -> str:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    except ValueError:
        return fallback


def current_dynamic_ma5_boundary(frame: pd.DataFrame, monitor_date: str) -> float | None:
    previous = frame[frame["trade_date"] < monitor_date].tail(4)
    closes = pd.to_numeric(previous["close"], errors="coerce").dropna()
    if len(closes) != 4:
        return None
    value = float(closes.mean())
    return value if math.isfinite(value) else None


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


def find_morning_trigger_minute(candidate: MonitorCandidate) -> str | None:
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
    today = candidate.monitor_date.replace("-", "")
    for row in rows:
        if not row or not str(row[0]).startswith(today):
            continue
        stamp = str(row[0])
        minute = f"{stamp[8:10]}:{stamp[10:12]}"
        if minute < "09:30" or minute > "10:00":
            continue
        high = float_or_none(row[3])
        low = float_or_none(row[4])
        if high is not None and low is not None and low <= candidate.boundary_price <= high:
            return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[8:10]}:{stamp[10:12]}"
    return None


def morning_status(monitor_date: str, trigger_time: str | None, now: str) -> str:
    if trigger_time:
        return "已触发，待手动确认"
    current_date, current_time = split_iso_minute(now)
    if current_date < monitor_date:
        return f"等待{monitor_date} 09:30开始观察"
    if current_date > monitor_date:
        return f"{monitor_date} 09:30-10:00未触发，不再买入"
    if current_time > "10:00":
        return "今日09:30-10:00未触发，不再买入"
    if current_time < "09:30":
        return "今日09:30开始观察"
    return "今日执行窗口观察中"


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
    if price <= 0 or price > 110:
        return {"shares": 0, "amount": 0.0}
    shares = int(per_position // (price * 100)) * 100
    if shares <= 0:
        shares = 100
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
    decision_rows = "\n".join(render_signal_decision_row(row) for row in snapshot.get("signal_decisions", []))
    quote_meta = snapshot.get("quote_meta", {})
    notice = render_quote_notice(quote_meta)
    source_dates = sorted(
        {
            signal_date
            for row in snapshot["rows"]
            for signal_date in row["candidate"].get("signal_dates", [])
        }
    )
    source_text = " / ".join(source_dates)
    execution_note = execution_window_note(str(snapshot["monitor_date"]), str(snapshot["updated_at"]))
    date_toolbar = render_date_toolbar(str(snapshot["monitor_date"]))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="{int(snapshot['refresh_seconds'])}">
  <title>{escape(snapshot['monitor_date'])} Top20 MA5 今日执行</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f8fafc; color: #111827; }}
    header {{ padding: 20px 28px; background: #111827; color: white; }}
    main {{ padding: 18px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; min-width: 1580px; }}
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
    .rule {{ margin-bottom: 12px; background: #fff7ed; color: #9a3412; border: 1px solid #fed7aa; padding: 10px 12px; }}
    .toolbar {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 12px; background: white; border: 1px solid #e5e7eb; padding: 10px 12px; }}
    input[type=date] {{ padding: 7px 9px; border: 1px solid #cbd5e1; }}
    button {{ padding: 8px 12px; border: 1px solid #cbd5e1; background: #f8fafc; cursor: pointer; }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(snapshot['monitor_date'])} 今日执行清单</h1>
    <div class="meta">今天只在 09:30-10:00 执行买入观察；昨日选股日：{escape(snapshot['decision_date'])}；延续观察来源：{escape(source_text)}；每票约1万；交界价<=110；更新 {escape(snapshot['updated_at'])}</div>
  </header>
  <main>{date_toolbar}{notice}<div class="rule">{escape(execution_note)} <a href="/decision">查看明日决策</a></div>
  <h2>{escape(snapshot['monitor_date'])} 执行观察池</h2>
  <div class="wrap"><table>
    <thead><tr><th>最佳排名</th><th>观察池来源</th><th>来源排名</th><th>执行日期</th><th>执行窗口</th><th>代码</th><th>名称</th><th>今日交界价</th><th>建议股数</th><th>建议金额</th><th>最新</th><th>当前盈亏</th><th>盈亏率</th><th>涨跌幅</th><th>日低</th><th>日高</th><th>成交额(亿)</th><th>行情时间</th><th>状态</th><th>操作</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
  <h2>{escape(snapshot['decision_date'])} Top20 选股决策</h2>
  <div class="wrap"><table>
    <thead><tr><th>成交额排名</th><th>代码</th><th>名称</th><th>收盘价</th><th>MA5</th><th>成交额(亿)</th><th>判断</th></tr></thead>
    <tbody>{decision_rows}</tbody>
  </table></div></main>
</body>
</html>"""


def render_signal_decision_row(row: dict[str, Any]) -> str:
    decision = str(row.get("decision") or "")
    status_class = "bought" if decision.startswith("入选") else "skipped"
    return (
        "<tr>"
        f"<td>{int(row.get('amount_rank') or 0)}</td>"
        f"<td>{escape(row.get('code') or '')}</td>"
        f"<td>{escape(row.get('name') or '')}</td>"
        f"<td>{format_number(row.get('close'))}</td>"
        f"<td>{format_number(row.get('ma5'))}</td>"
        f"<td>{format_number((row.get('amount') or 0) / 100000000)}</td>"
        f"<td><span class=\"badge {status_class}\">{escape(decision)}</span></td>"
        "</tr>"
    )


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
    if detail and "跳过" not in status:
        status = f"{status}；{detail}"
    code = escape(candidate["code"])
    monitor_date = escape(candidate["monitor_date"])
    detail_href = f"/stock?code={code}&date={monitor_date}"
    amount_ranks = ",".join(str(item) for item in candidate.get("amount_ranks", []))
    observation_source = observation_source_text(candidate["monitor_date"], candidate.get("signal_dates", []))
    return (
        "<tr>"
        f"<td>{candidate['amount_rank']}</td>"
        f"<td>{escape(observation_source)}</td>"
        f"<td>{escape(amount_ranks)}</td>"
        f"<td>{monitor_date}</td>"
        "<td>09:30-10:00</td>"
        f"<td><a href=\"{detail_href}\" target=\"_blank\">{code}</a></td>"
        f"<td><a href=\"{detail_href}\" target=\"_blank\">{escape(candidate['name'])}</a></td>"
        f"<td>{format_boundary_with_spread(candidate['boundary_price'], quote.get('latest'))}</td>"
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
        f"<td><a href=\"/mark?code={code}&status=bought&date={monitor_date}\">标记已买</a><a href=\"/mark?code={code}&status=skipped&date={monitor_date}\">跳过</a><a href=\"/mark?code={code}&status=reset&date={monitor_date}\">重置</a></td>"
        "</tr>"
    )


def render_date_toolbar(monitor_date: str) -> str:
    previous_day = previous_weekday(monitor_date)
    next_day = next_weekday(monitor_date)
    return f"""
<form class="toolbar" method="get" action="/">
  <strong>执行日期</strong>
  <a href="/?date={escape(previous_day)}">上一交易日</a>
  <input type="date" name="date" value="{escape(monitor_date)}">
  <button type="submit">切换</button>
  <a href="/?date={escape(next_day)}">下一交易日</a>
  <span>本页展示该日期 09:30-10:00 的执行清单，来源是此前最多 3 个交易日的 Top20 观察池。</span>
</form>"""


def observation_source_text(monitor_date: str, signal_dates: list[str]) -> str:
    previous_dates = previous_trade_dates(monitor_date, 3)
    labels = []
    for signal_date in signal_dates:
        try:
            index = previous_dates.index(str(signal_date)) + 1
        except ValueError:
            labels.append(str(signal_date))
            continue
        labels.append(f"第{index}")
    return " / ".join(labels)


def render_decision_page(app: MonitorApp) -> str:
    latest_date = infer_latest_cache_date(app.daily_cache_dir)
    today = date.today().isoformat()
    tomorrow = next_weekday(today)
    expected_signal_date = today
    decision_ready = latest_date == expected_signal_date
    if decision_ready:
        decision_title = f"{tomorrow} 明日执行清单已可生成"
        decision_text = (
            f"已经有 {expected_signal_date} 收盘日线数据。明日 {tomorrow} 的执行清单，"
            f"应由 {expected_signal_date} 的 Top20 选股结果，加上仍在 3 日观察期内的历史信号合并生成。"
        )
    else:
        decision_title = f"{tomorrow} 明日决策尚未生成"
        decision_text = (
            f"当前日线缓存最新是 {latest_date or '-'}，还没有 {expected_signal_date} 收盘数据。"
            f"因此现在不能严谨生成 {tomorrow} 的明日执行清单。需要先更新日线数据，再用 {expected_signal_date} "
            f"的成交额 Top20、收盘站上 MA5 结果作为新信号。"
        )
    source_dates = previous_trade_dates(tomorrow, 3)
    steps = [
        ("1. 收盘后选股", f"用 {expected_signal_date} 收盘后的沪深成交额 Top20，剔除创业板/科创板/北交所/ST，保留收盘价站上 MA5 的股票。"),
        ("2. 合并观察池", f"把来源信号日 {', '.join(source_dates)} 中仍在 3 日观察期内的同一股票合并成一条。"),
        ("3. 生成明日买点", f"对 {tomorrow} 计算动态 MA5 交界价，也就是 {tomorrow} 盘中价格等于 MA5 时的价格。"),
        ("4. 明日执行", f"{tomorrow} 只在 09:30-10:00 观察，触碰交界价才考虑买入；交界价 >110 或 1 万买不了一手则跳过。"),
    ]
    step_rows = "".join(f"<tr><td>{escape(title)}</td><td>{escape(text)}</td></tr>" for title, text in steps)
    status_class = "ok" if decision_ready else "pending"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(tomorrow)} 明日决策</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f8fafc; color: #111827; }}
    header {{ padding: 20px 28px; background: #111827; color: white; }}
    main {{ padding: 18px 26px; }}
    a {{ color: #2563eb; text-decoration: none; }}
    .card {{ background: white; border: 1px solid #e5e7eb; padding: 14px; margin-bottom: 14px; }}
    .pending {{ border-color: #fed7aa; background: #fff7ed; color: #9a3412; }}
    .ok {{ border-color: #bbf7d0; background: #f0fdf4; color: #166534; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 10px 12px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; color: #334155; }}
    .meta {{ color: #cbd5e1; margin-top: 6px; }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(tomorrow)} 明日决策</h1>
    <div class="meta">今天是 {escape(today)}；日线缓存最新 {escape(str(latest_date or '-'))}</div>
  </header>
  <main>
    <div class="card {status_class}">
      <h2>{escape(decision_title)}</h2>
      <p>{escape(decision_text)}</p>
    </div>
    <div class="card">
      <p><a href="/">返回今日执行清单</a></p>
      <table>
        <thead><tr><th>步骤</th><th>规则</th></tr></thead>
        <tbody>{step_rows}</tbody>
      </table>
    </div>
  </main>
</body>
</html>"""


def next_weekday(day: str) -> str:
    current = datetime.strptime(day, "%Y-%m-%d").date() + timedelta(days=1)
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return current.isoformat()


def previous_weekday(day: str) -> str:
    current = datetime.strptime(day, "%Y-%m-%d").date() - timedelta(days=1)
    while current.weekday() >= 5:
        current -= timedelta(days=1)
    return current.isoformat()


def execution_window_note(monitor_date: str, updated_at: str) -> str:
    current_date, current_time = split_iso_minute(updated_at)
    if current_date > monitor_date or (current_date == monitor_date and current_time > "10:00"):
        return f"{monitor_date} 的买入执行窗口已经结束；未触发的票今天不再买，明天需要重新生成明日执行清单。"
    if current_date == monitor_date and "09:30" <= current_time <= "10:00":
        return f"现在是 {monitor_date} 今日执行窗口，只在 09:30-10:00 内触碰交界价才考虑买入。"
    if current_date == monitor_date and current_time < "09:30":
        return f"{monitor_date} 今日执行窗口尚未开始，09:30 后再观察是否触碰交界价。"
    return f"本页是 {monitor_date} 今日执行清单，不是明日计划。"


def split_iso_minute(value: str) -> tuple[str, str]:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value[:10], value[11:16]
    return parsed.date().isoformat(), parsed.strftime("%H:%M")


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


def format_boundary_with_spread(boundary: Any, latest: Any) -> str:
    boundary_text = format_number(boundary)
    try:
        spread = float(latest) - float(boundary)
    except (TypeError, ValueError):
        return boundary_text
    if not math.isfinite(spread):
        return boundary_text
    return f"{boundary_text}（{spread:+.2f}）"


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
