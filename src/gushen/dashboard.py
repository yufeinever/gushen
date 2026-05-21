from __future__ import annotations

import csv
import json
import threading
from dataclasses import dataclass, asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from gushen.macro_regime import MacroRegime, build_macro_regime, load_or_build_macro_regime
from gushen.tradingagents_dataset import build_tradingagents_dataset


DEFAULT_TRADE_DATE = "2026-05-20"
SERVER_STATE: dict[str, Any] = {"running": False, "message": "", "last_error": ""}


def _zh(key: str) -> str:
    values = {
        "title": "\u80a1\u795e\u00b7A\u80a1 AI Agent \u7814\u7a76\u770b\u677f",
        "subtitle": "\u6210\u4ea4\u989d Top100 \u5168\u6c60\u5206\u6790\uff1a\u6280\u672f\u3001\u4ea4\u6613\u9650\u5236\u3001\u4f30\u503c\u3001\u4e8b\u4ef6\u3001\u56de\u6d4b",
        "rerun": "\u91cd\u65b0\u8dd1\u5206\u6790",
        "refresh": "\u5237\u65b0\u9875\u9762\u6570\u636e",
        "good": "\u76f8\u5bf9\u66f4\u597d",
        "watch": "\u7ee7\u7eed\u89c2\u5bdf",
        "bad": "\u76f8\u5bf9\u4e0d\u597d",
        "avoid": "\u56de\u907f",
        "insufficient": "\u6570\u636e\u4e0d\u8db3",
        "sources": "\u6570\u636e\u6765\u6e90",
        "evidence": "\u8ba1\u7b97\u4f9d\u636e",
        "risks": "\u98ce\u9669",
        "missing": "\u7f3a\u53e3",
        "entry": "\u6a21\u62df\u89c2\u5bdf\u903b\u8f91",
        "all": "\u5168\u90e8",
        "search": "\u8f93\u5165\u4ee3\u7801\u6216\u540d\u79f0",
        "macro": "\u5b8f\u89c2\u73af\u5883",
    }
    return values[key]


@dataclass(frozen=True)
class StockScore:
    rank: int
    code: str
    name: str
    close: float
    amount: float
    score: float
    label: str
    label_text: str
    evidence: list[str]
    risks: list[str]
    missing: list[str]
    sources: list[str]
    metrics: dict[str, float | int | str]
    event_summary: str
    plan: dict[str, str]


def build_dashboard_payload(trade_date: str = DEFAULT_TRADE_DATE) -> dict[str, Any]:
    dataset_dir = Path(f"reports/generated/tradingagents_dataset_{trade_date}")
    if not dataset_dir.exists():
        build_tradingagents_dataset(trade_date)
    macro = load_or_build_macro_regime(trade_date)
    rows = score_top100(dataset_dir, macro)
    counts = {"good": 0, "watch": 0, "bad": 0, "avoid": 0, "insufficient": 0}
    for row in rows:
        counts[row.label] = counts.get(row.label, 0) + 1
    return {
        "trade_date": trade_date,
        "data_sufficiency": {
            "status": "partial",
            "note": (
                "\u6280\u672f\u3001\u4ea4\u6613\u9650\u5236\u3001\u4e8b\u4ef6\u548c\u56de\u6d4b\u5df2\u63a5\u5165\uff1b"
                "\u5df2\u52a0\u5165 MacroRegimeAgent\uff0c\u4f7f\u7528\u7f8e\u503a\u3001\u6c47\u7387\u3001LPR/SHIBOR\u3001PMI\u3001QVIX \u8bc4\u4f30\u5e02\u573a\u73af\u5883\uff1b"
                "\u57fa\u672c\u9762\u591a\u4e3a valuation_fallback\uff0c\u8fd8\u4e0d\u662f\u5b8c\u6574\u8d22\u62a5\u5b57\u6bb5\u3002"
                "\u8f93\u51fa\u4ec5\u7528\u4e8e\u7814\u7a76\u548c\u6a21\u62df\u89c2\u5bdf\uff0c\u4e0d\u662f\u5b9e\u76d8\u4e70\u5356\u5efa\u8bae\u3002"
            ),
        },
        "macro": asdict(macro),
        "counts": counts,
        "rows": [asdict(row) for row in rows],
        "state": SERVER_STATE,
    }


def score_top100(dataset_dir: Path, macro: MacroRegime | None = None) -> list[StockScore]:
    market = _by_code(_read_csv(dataset_dir / "market_technical.csv"))
    risk = _by_code(_read_csv(dataset_dir / "risk_tradability.csv"))
    fundamentals = _by_code(_read_csv(dataset_dir / "fundamentals.csv"))
    events = _by_code(_read_csv(dataset_dir / "events.csv"))
    backtests = _by_code(_read_csv(dataset_dir / "backtests.csv"))
    rows = [
        _score_stock(code, market[code], risk, fundamentals, events, backtests, macro)
        for code in market
    ]
    return sorted(rows, key=lambda item: item.rank)


def _score_stock(
    code: str,
    market_row: dict[str, str],
    risk_rows: dict[str, dict[str, str]],
    fundamental_rows: dict[str, dict[str, str]],
    event_rows: dict[str, dict[str, str]],
    backtest_rows: dict[str, dict[str, str]],
    macro: MacroRegime | None = None,
) -> StockScore:
    risk = risk_rows.get(code, {})
    fundamentals = fundamental_rows.get(code, {})
    events = event_rows.get(code, {})
    backtest = backtest_rows.get(code, {})
    score = 0.0
    evidence: list[str] = []
    risks: list[str] = []
    missing: list[str] = []
    sources = [
        "market_technical.csv / EastMoney daily OHLCV",
        "risk_tradability.csv / AKShare ST, suspension, limit pools",
        "fundamentals.csv / EastMoney valuation fallback",
        "events.csv / AKShare stock_news_em and stock_individual_notice_report",
        "backtests.csv / local momentum-volume signal backtest",
        "macro_regime.json / AKShare bond, forex, LPR, SHIBOR, PMI, QVIX",
    ]

    ret_5d = _float(market_row.get("ret_5d"))
    ret_20d = _float(market_row.get("ret_20d"))
    ma5_gap = _float(market_row.get("ma5_gap"))
    amount_ratio_5d = _float(market_row.get("amount_ratio_5d"))
    volatility_20d = _float(market_row.get("volatility_20d"))
    close = _float(market_row.get("close"))
    amount = _float(market_row.get("amount"))
    turnover = _float(market_row.get("turnover"))
    pe_dynamic = _float(fundamentals.get("pe_dynamic"))
    pb = _float(fundamentals.get("pb"))
    bt_sample = _int(backtest.get("sample_count"))
    bt_avg_3d = _float(backtest.get("avg_return_3d"))
    bt_win_3d = _float(backtest.get("win_rate_3d"))
    bt_drawdown_5d = _float(backtest.get("max_drawdown_5d"))
    macro_penalty = _macro_penalty(market_row.get("name", ""), pe_dynamic, macro)
    if macro:
        score += macro.score_adjustment
        if macro.score_adjustment >= 0:
            evidence.append(f"MacroRegimeAgent \u52a0\u5206\uff1a{macro.score_adjustment:+.1f}")
        else:
            risks.append(f"MacroRegimeAgent \u603b\u4f53\u6263\u5206\uff1a{macro.score_adjustment:+.1f}")
    if macro_penalty < 0:
        score += macro_penalty
        risks.append(
            f"\u5b8f\u89c2\u654f\u611f\u6263\u5206\uff1a{macro_penalty:+.1f}\uff08\u9ad8\u4f30\u503c/\u79d1\u6280\u6210\u957f\u9047\u7f8e\u503a\u9ad8\u4f4d\uff09"
        )

    if risk.get("tradability_note") == "normal":
        score += 18
        evidence.append("\u4ea4\u6613\u72b6\u6001\u6b63\u5e38\uff1a\u975e ST\u3001\u975e\u505c\u724c\u3001\u975e\u6da8\u8dcc\u505c")
    else:
        risks.append(f"\u4ea4\u6613\u9650\u5236\uff1a{risk.get('tradability_note', 'missing')}")

    if fundamentals.get("source_status") in {"ok", "spot_fallback", "valuation_fallback"}:
        score += 10
        evidence.append(
            f"\u4f30\u503c\u53ef\u7528\uff1aPE={pe_dynamic:.2f}, PB={pb:.2f}, source={fundamentals.get('source_status')}"
        )
    else:
        missing.append("\u57fa\u672c\u9762/\u4f30\u503c\u5b57\u6bb5\u7f3a\u5931")

    if events.get("event_status") == "loaded":
        score += 10
        evidence.append("\u4e8b\u4ef6/\u516c\u544a/\u65b0\u95fb\u5df2\u52a0\u8f7d")
        event_text = events.get("event_summary", "")
        hits = [token for token in _risk_tokens() if token in event_text]
        if hits:
            risks.append("\u4e8b\u4ef6\u98ce\u9669\u8bcd\uff1a" + "\u3001".join(hits[:4]))
    else:
        missing.append("\u4e8b\u4ef6/\u516c\u544a/\u65b0\u95fb\u7f3a\u5931")

    if backtest.get("source_status") == "ok" and bt_sample >= 5:
        if bt_avg_3d > 0 and bt_win_3d >= 0.6:
            score += 28
            evidence.append(f"\u56de\u6d4b\u504f\u5f3a\uff1a3\u65e5\u5747\u503c={bt_avg_3d:.2%}, \u80dc\u7387={bt_win_3d:.0%}, n={bt_sample}")
        elif bt_avg_3d > 0 and bt_win_3d >= 0.5:
            score += 14
            risks.append("\u56de\u6d4b\u4f18\u52bf\u504f\u5f31")
        else:
            risks.append("\u56de\u6d4b 3 \u65e5\u6536\u76ca/\u80dc\u7387\u4e0d\u8db3")
    else:
        missing.append(f"\u56de\u6d4b\u6837\u672c\u4e0d\u8db3\uff1a{backtest.get('source_status', 'missing')}, n={bt_sample}")

    if 0.02 <= ret_5d <= 0.18:
        score += 10
        evidence.append(f"5\u65e5\u52a8\u91cf\u9002\u4e2d\uff1a{ret_5d:.2%}")
    elif ret_5d > 0.25:
        risks.append(f"5\u65e5\u8fc7\u70ed\uff1a{ret_5d:.2%}")
    elif ret_5d < -0.08:
        risks.append(f"5\u65e5\u8d8b\u52bf\u8f6c\u5f31\uff1a{ret_5d:.2%}")

    if 0.03 <= ret_20d <= 0.35:
        score += 8
        evidence.append(f"20\u65e5\u8d8b\u52bf\u9002\u4e2d\uff1a{ret_20d:.2%}")
    elif ret_20d > 0.60:
        risks.append(f"20\u65e5\u6da8\u5e45\u8fc7\u70ed\uff1a{ret_20d:.2%}")
    elif ret_20d < -0.12:
        risks.append(f"20\u65e5\u8d8b\u52bf\u504f\u5f31\uff1a{ret_20d:.2%}")

    if 0 <= ma5_gap <= 0.08:
        score += 6
        evidence.append(f"\u8d34\u8fd1 5 \u65e5\u5747\u7ebf\uff1a{ma5_gap:.2%}")
    elif ma5_gap > 0.12:
        risks.append(f"\u504f\u79bb 5 \u65e5\u5747\u7ebf\u8fc7\u5927\uff1a{ma5_gap:.2%}")

    if 1.1 <= amount_ratio_5d <= 2.8:
        score += 6
        evidence.append(f"\u6210\u4ea4\u989d\u653e\u91cf\uff1a{amount_ratio_5d:.2f}x")
    elif amount_ratio_5d < 0.8:
        risks.append(f"\u6210\u4ea4\u989d\u7f29\u91cf\uff1a{amount_ratio_5d:.2f}x")

    if volatility_20d <= 0.055:
        score += 4
    elif volatility_20d > 0.08:
        risks.append(f"20\u65e5\u6ce2\u52a8\u8fc7\u9ad8\uff1a{volatility_20d:.2%}")

    if bt_drawdown_5d < -0.12:
        risks.append(f"\u56de\u6d4b 5 \u65e5\u6700\u5927\u56de\u64a4\u6df1\uff1a{bt_drawdown_5d:.2%}")

    if risk.get("tradability_note") != "normal":
        label = "avoid"
    elif not risks and not missing and score >= 80:
        label = "good"
    elif score < 45 or len(risks) >= 3:
        label = "bad"
    elif missing:
        label = "insufficient"
    else:
        label = "watch"

    stop_loss = max(0.035, min(0.07, volatility_20d * 1.5 if volatility_20d else 0.05))
    take_profit = max(0.06, stop_loss * 1.8)
    return StockScore(
        rank=_int(market_row.get("amount_rank")),
        code=code,
        name=market_row.get("name", ""),
        close=close,
        amount=amount,
        score=round(score, 2),
        label=label,
        label_text=_label_text(label),
        evidence=evidence,
        risks=risks or ["none"],
        missing=missing or ["none"],
        sources=sources,
        metrics={
            "ret_5d": ret_5d,
            "ret_20d": ret_20d,
            "ma5_gap": ma5_gap,
            "amount_ratio_5d": amount_ratio_5d,
            "volatility_20d": volatility_20d,
            "turnover": turnover,
            "pe_dynamic": pe_dynamic,
            "pb": pb,
            "bt_sample": bt_sample,
            "bt_avg_3d": bt_avg_3d,
            "bt_win_3d": bt_win_3d,
            "bt_drawdown_5d": bt_drawdown_5d,
            "macro_adjustment": macro.score_adjustment if macro else 0,
            "macro_sensitive_penalty": macro_penalty,
        },
        event_summary=events.get("event_summary", ""),
        plan={
            "entry": (
                "\u4ec5\u6a21\u62df\uff1a\u6b21\u4e00\u4ea4\u6613\u65e5\u4ecd\u5728\u6210\u4ea4\u989d Top100\uff0c"
                "\u4e0d\u6da8\u505c/\u8dcc\u505c\uff0c\u5f00\u76d8\u540e 30 \u5206\u949f\u4e0d\u8dcc\u7834\u524d\u6536\uff0c"
                "\u4e14\u653e\u91cf\u91cd\u56de\u5206\u65f6\u5747\u4ef7\u4e0a\u65b9\u624d\u8bb0\u5f55\u3002"
            ),
            "exit": (
                f"\u4ec5\u6a21\u62df\uff1a\u8dcc\u7834\u5165\u573a\u4ef7 {stop_loss:.1%} \u6b62\u635f\uff1b"
                f"\u76c8\u5229\u8fbe {take_profit:.1%} \u5206\u6279\u6b62\u76c8\uff1b"
                "3 \u4e2a\u4ea4\u6613\u65e5\u672a\u8d70\u5f3a\u51cf\u4ed3\u89c2\u5bdf\uff0c5 \u4e2a\u4ea4\u6613\u65e5\u5f3a\u5236\u590d\u76d8\u3002"
            ),
            "position": "\u5355\u7968\u6a21\u62df\u4ed3\u4e0d\u8d85\u8fc7 5%\uff0c\u540c\u9898\u6750\u5408\u8ba1\u4e0d\u8d85\u8fc7 10%\u3002",
        },
    )


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.startswith("/api/analysis"):
            self._send_json(build_dashboard_payload())
            return
        self._send_html(_html())

    def do_POST(self) -> None:
        if self.path.startswith("/api/rerun"):
            if SERVER_STATE["running"]:
                self._send_json({"ok": False, "message": "rerun already running"}, HTTPStatus.CONFLICT)
                return
            thread = threading.Thread(target=_rerun_worker, daemon=True)
            thread.start()
            self._send_json({"ok": True, "message": "rerun started"})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_html(self, html: str) -> None:
        payload = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _rerun_worker() -> None:
    SERVER_STATE.update({"running": True, "message": "\u6b63\u5728\u91cd\u8dd1\u6570\u636e\u96c6...", "last_error": ""})
    try:
        build_tradingagents_dataset(DEFAULT_TRADE_DATE)
        build_macro_regime(DEFAULT_TRADE_DATE)
        SERVER_STATE.update({"message": "\u91cd\u8dd1\u5b8c\u6210"})
    except Exception as exc:
        SERVER_STATE.update({"last_error": f"{type(exc).__name__}: {exc}", "message": "\u91cd\u8dd1\u5931\u8d25"})
    finally:
        SERVER_STATE["running"] = False


def _html() -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_zh("title")}</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d8dee8;
      --good: #0b7a53;
      --watch: #7a5b00;
      --bad: #a43820;
      --avoid: #8f1f32;
      --accent: #2364aa;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Microsoft YaHei", Segoe UI, Arial, sans-serif; background: var(--bg); color: var(--ink); }}
    header {{ padding: 18px 24px 14px; background: #ffffff; border-bottom: 1px solid var(--line); position: sticky; top: 0; z-index: 3; }}
    h1 {{ font-size: 22px; margin: 0 0 6px; letter-spacing: 0; }}
    .sub {{ color: var(--muted); font-size: 13px; }}
    .bar {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 14px; }}
    button, input, select {{ height: 34px; border: 1px solid var(--line); border-radius: 6px; background: #fff; color: var(--ink); padding: 0 10px; font-size: 13px; }}
    button {{ cursor: pointer; background: var(--accent); color: white; border-color: var(--accent); }}
    button.secondary {{ background: #fff; color: var(--accent); }}
    button:disabled {{ opacity: .55; cursor: wait; }}
    main {{ display: grid; grid-template-columns: 360px 1fr; gap: 14px; padding: 14px; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; min-width: 0; }}
    .summary {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; padding: 12px; }}
    .metric {{ border: 1px solid var(--line); border-radius: 6px; padding: 8px; background: #fbfcfe; }}
    .metric b {{ display:block; font-size: 18px; }}
    .note {{ margin: 0 12px 12px; color: var(--muted); font-size: 13px; line-height: 1.5; }}
    .macro {{ margin: 0 12px 12px; border: 1px solid var(--line); border-radius: 6px; padding: 10px; background: #fbfcfe; font-size: 13px; line-height: 1.55; }}
    .macro b {{ display: block; margin-bottom: 4px; }}
    .list {{ max-height: calc(100vh - 204px); overflow: auto; }}
    .row {{ display: grid; grid-template-columns: 42px 1fr 64px; gap: 8px; padding: 10px 12px; border-top: 1px solid var(--line); cursor: pointer; }}
    .row:hover, .row.active {{ background: #eef4fb; }}
    .rank {{ color: var(--muted); font-variant-numeric: tabular-nums; }}
    .name {{ font-weight: 700; }}
    .code {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}
    .score {{ text-align: right; font-weight: 700; }}
    .tag {{ display: inline-block; margin-top: 4px; padding: 2px 6px; border-radius: 999px; font-size: 12px; border: 1px solid var(--line); }}
    .tag.good {{ color: var(--good); border-color: #a8d8c4; background: #eef8f4; }}
    .tag.watch {{ color: var(--watch); border-color: #e5d28c; background: #fff8db; }}
    .tag.bad {{ color: var(--bad); border-color: #ecc0b4; background: #fff1ed; }}
    .tag.avoid {{ color: var(--avoid); border-color: #edb9c1; background: #fff0f3; }}
    .tag.insufficient {{ color: #595f6b; background: #f1f3f5; }}
    .detail {{ padding: 16px; }}
    .detail h2 {{ margin: 0; font-size: 24px; }}
    .detail-grid {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 10px; margin: 14px 0; }}
    .kv {{ border: 1px solid var(--line); border-radius: 6px; padding: 9px; background: #fbfcfe; }}
    .kv span {{ display: block; color: var(--muted); font-size: 12px; }}
    .kv b {{ display: block; margin-top: 4px; font-size: 15px; }}
    section {{ border-top: 1px solid var(--line); padding-top: 12px; margin-top: 12px; }}
    h3 {{ font-size: 15px; margin: 0 0 8px; }}
    ul {{ margin: 0; padding-left: 18px; line-height: 1.7; }}
    .event {{ color: #334155; line-height: 1.6; white-space: pre-wrap; }}
    @media (max-width: 900px) {{ main {{ grid-template-columns: 1fr; }} .list {{ max-height: 420px; }} .detail-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
  </style>
</head>
<body>
  <header>
    <h1>{_zh("title")}</h1>
    <div class="sub">{_zh("subtitle")}</div>
    <div class="bar">
      <input id="search" placeholder="{_zh("search")}" />
      <select id="filter">
        <option value="all">{_zh("all")}</option>
        <option value="good">{_zh("good")}</option>
        <option value="watch">{_zh("watch")}</option>
        <option value="bad">{_zh("bad")}</option>
        <option value="avoid">{_zh("avoid")}</option>
        <option value="insufficient">{_zh("insufficient")}</option>
      </select>
      <button id="rerun">{_zh("rerun")}</button>
      <button class="secondary" id="refresh">{_zh("refresh")}</button>
      <span class="sub" id="status"></span>
    </div>
  </header>
  <main>
    <aside class="panel">
      <div class="summary" id="summary"></div>
      <p class="note" id="note"></p>
      <div class="macro" id="macro"></div>
      <div class="list" id="list"></div>
    </aside>
    <article class="panel detail" id="detail"></article>
  </main>
  <script>
    let payload = null;
    let selected = null;
    const labels = {{good:"{_zh("good")}", watch:"{_zh("watch")}", bad:"{_zh("bad")}", avoid:"{_zh("avoid")}", insufficient:"{_zh("insufficient")}"}};
    const fmtPct = v => (Number(v) * 100).toFixed(2) + "%";
    const fmtNum = v => Number(v).toLocaleString("zh-CN", {{maximumFractionDigits: 2}});
    async function loadData() {{
      const res = await fetch("/api/analysis");
      payload = await res.json();
      selected = selected || payload.rows[0]?.code;
      render();
    }}
    function render() {{
      document.getElementById("note").textContent = payload.data_sufficiency.note;
      renderMacro();
      document.getElementById("status").textContent = payload.state.running ? payload.state.message : payload.state.last_error || payload.state.message || "";
      document.getElementById("summary").innerHTML = ["good","watch","bad","avoid","insufficient"].map(k => `<div class="metric"><span>${{labels[k]}}</span><b>${{payload.counts[k] || 0}}</b></div>`).join("");
      const q = document.getElementById("search").value.trim().toLowerCase();
      const f = document.getElementById("filter").value;
      const rows = payload.rows.filter(r => (f === "all" || r.label === f) && (!q || r.code.toLowerCase().includes(q) || r.name.toLowerCase().includes(q)));
      document.getElementById("list").innerHTML = rows.map(r => `<div class="row ${{r.code===selected?"active":""}}" data-code="${{r.code}}"><div class="rank">#${{r.rank}}</div><div><div class="name">${{r.name}}</div><div class="code">${{r.code}}</div><span class="tag ${{r.label}}">${{r.label_text}}</span></div><div class="score">${{r.score}}</div></div>`).join("");
      document.querySelectorAll(".row").forEach(el => el.onclick = () => {{ selected = el.dataset.code; render(); }});
      const item = payload.rows.find(r => r.code === selected) || rows[0] || payload.rows[0];
      if (item) selected = item.code;
      renderDetail(item);
    }}
    function renderDetail(r) {{
      if (!r) {{ document.getElementById("detail").innerHTML = ""; return; }}
      const m = r.metrics;
      document.getElementById("detail").innerHTML = `
        <h2>${{r.name}} <span class="sub">${{r.code}} / #${{r.rank}}</span></h2>
        <span class="tag ${{r.label}}">${{r.label_text}}</span>
        <div class="detail-grid">
          <div class="kv"><span>Score</span><b>${{r.score}}</b></div>
          <div class="kv"><span>Close</span><b>${{fmtNum(r.close)}}</b></div>
          <div class="kv"><span>Ret 5D / 20D</span><b>${{fmtPct(m.ret_5d)}} / ${{fmtPct(m.ret_20d)}}</b></div>
          <div class="kv"><span>Amount 5D Ratio</span><b>${{Number(m.amount_ratio_5d).toFixed(2)}}x</b></div>
          <div class="kv"><span>Vol 20D</span><b>${{fmtPct(m.volatility_20d)}}</b></div>
          <div class="kv"><span>Turnover</span><b>${{fmtPct(m.turnover)}}</b></div>
          <div class="kv"><span>PE / PB</span><b>${{Number(m.pe_dynamic).toFixed(2)}} / ${{Number(m.pb).toFixed(2)}}</b></div>
          <div class="kv"><span>Backtest</span><b>n=${{m.bt_sample}}, avg3=${{fmtPct(m.bt_avg_3d)}}, win3=${{fmtPct(m.bt_win_3d)}}</b></div>
          <div class="kv"><span>Macro Adj / Sensitivity</span><b>${{Number(m.macro_adjustment).toFixed(1)}} / ${{Number(m.macro_sensitive_penalty).toFixed(1)}}</b></div>
        </div>
        ${{section("{_zh('evidence')}", r.evidence)}}
        ${{section("{_zh('risks')}", r.risks)}}
        ${{section("{_zh('missing')}", r.missing)}}
        <section><h3>{_zh("entry")}</h3><ul><li>${{r.plan.entry}}</li><li>${{r.plan.exit}}</li><li>${{r.plan.position}}</li></ul></section>
        <section><h3>{_zh("sources")}</h3><ul>${{r.sources.map(s => `<li>${{s}}</li>`).join("")}}</ul></section>
        <section><h3>事件摘要</h3><div class="event">${{r.event_summary || "none"}}</div></section>
      `;
    }}
    function section(title, rows) {{
      return `<section><h3>${{title}}</h3><ul>${{rows.map(x => `<li>${{x}}</li>`).join("")}}</ul></section>`;
    }}
    function renderMacro() {{
      const m = payload.macro || {{}};
      const risks = (m.risks || []).slice(0, 2).join("；") || "none";
      const supports = (m.supports || []).slice(0, 2).join("；") || "none";
      document.getElementById("macro").innerHTML = `<b>{_zh("macro")}：${{m.status || "unknown"}} (${{Number(m.score_adjustment || 0).toFixed(1)}})</b>
        <div>US10Y/30Y：${{m.us_10y ?? "-"}} / ${{m.us_30y ?? "-"}}，USDCNH：${{m.usdcnh ?? "-"}}</div>
        <div>LPR1Y/5Y：${{m.lpr_1y ?? "-"}} / ${{m.lpr_5y ?? "-"}}，SHIBOR O/N：${{m.shibor_on ?? "-"}}</div>
        <div>PMI：${{m.china_pmi ?? "-"}}，300ETF QVIX：${{m.qvix_300etf ?? "-"}}</div>
        <div>风险：${{risks}}</div><div>支撑：${{supports}}</div>`;
    }}
    document.getElementById("search").oninput = render;
    document.getElementById("filter").onchange = render;
    document.getElementById("refresh").onclick = loadData;
    document.getElementById("rerun").onclick = async () => {{
      const btn = document.getElementById("rerun");
      btn.disabled = true;
      await fetch("/api/rerun", {{method:"POST"}});
      const timer = setInterval(async () => {{
        await loadData();
        if (!payload.state.running) {{ clearInterval(timer); btn.disabled = false; }}
      }}, 2500);
    }};
    loadData();
  </script>
</body>
</html>"""


def _risk_tokens() -> tuple[str, ...]:
    return (
        "\u4e8f\u635f",
        "\u51cf\u6301",
        "\u98ce\u9669\u63d0\u793a",
        "\u95ee\u8be2\u51fd",
        "\u7acb\u6848",
        "\u5f02\u52a8",
        "\u6f84\u6e05",
        "\u9000\u5e02",
        "\u5904\u7f5a",
        "\u76d1\u7ba1",
    )


def _label_text(label: str) -> str:
    return {
        "good": _zh("good"),
        "watch": _zh("watch"),
        "bad": _zh("bad"),
        "avoid": _zh("avoid"),
        "insufficient": _zh("insufficient"),
    }[label]


def _macro_penalty(name: str, pe_dynamic: float, macro: MacroRegime | None) -> float:
    if not macro or not macro.us_30y or macro.us_30y < 5.0:
        return 0.0
    growth_keywords = (
        "\u79d1\u6280",
        "\u82af",
        "\u5fae",
        "\u5149",
        "\u7535\u5b50",
        "\u534a\u5bfc\u4f53",
        "\u4fe1\u606f",
        "\u901a\u4fe1",
        "\u5149\u7535",
        "\u521b",
        "\u673a\u5668",
    )
    penalty = 0.0
    if pe_dynamic >= 80:
        penalty -= 5
    elif pe_dynamic >= 50:
        penalty -= 3
    if any(keyword in name for keyword in growth_keywords):
        penalty -= 3
    return penalty


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _by_code(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["code"]: row for row in rows}


def _float(value: str | None) -> float:
    try:
        return float(value or 0)
    except ValueError:
        return 0.0


def _int(value: str | None) -> int:
    try:
        return int(float(value or 0))
    except ValueError:
        return 0


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Dashboard: http://{host}:{port}")
    server.serve_forever()


def main() -> None:
    run_server()


if __name__ == "__main__":
    main()
