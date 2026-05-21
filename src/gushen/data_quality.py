from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


CORE_FILES = {
    "market": "market_technical.csv",
    "tradability": "risk_tradability.csv",
    "fundamentals": "fundamentals.csv",
    "events": "events.csv",
    "backtests": "backtests.csv",
}


@dataclass(frozen=True)
class DataQualityCheck:
    component: str
    status: str
    score: float
    evidence: list[str]
    missing: list[str]
    sources: list[str]


@dataclass(frozen=True)
class DataQualityReport:
    trade_date: str
    status: str
    action_gate: str
    score: float
    note: str
    checks: list[DataQualityCheck]
    hard_gaps: list[str]


def assess_dataset_quality(dataset_dir: Path, trade_date: str) -> DataQualityReport:
    checks = [
        _check_market(dataset_dir),
        _check_tradability(dataset_dir),
        _check_fundamentals(dataset_dir),
        _check_events(dataset_dir),
        _check_backtests(dataset_dir),
        _check_macro(trade_date),
        _missing_check(
            "SectorThemeAgent",
            "industry/sector strength dataset is not wired yet",
            "AKShare sector/industry/theme interfaces",
        ),
        _missing_check(
            "FundFlowAgent",
            "main fund flow, margin financing, northbound flow and dragon-tiger data are not wired yet",
            "AKShare fund flow / margin / dragon-tiger interfaces",
        ),
        _missing_check(
            "A-share SentimentNarrativeAgent raw feed",
            "forum/social raw feed is not wired yet; current events are news/announcement summaries only",
            "EastMoney Guba, Xueqiu, exchange announcements, web search",
        ),
    ]
    weights = {
        "MarketAnalyst": 0.18,
        "TradabilityRisk": 0.14,
        "FundamentalsAnalyst": 0.13,
        "NewsEventAnalyst": 0.12,
        "BacktestValidator": 0.18,
        "MacroRegimeAgent": 0.10,
        "SectorThemeAgent": 0.05,
        "FundFlowAgent": 0.05,
        "A-share SentimentNarrativeAgent raw feed": 0.05,
    }
    score = sum(check.score * weights.get(check.component, 0.0) for check in checks)
    hard_gaps = [
        gap
        for check in checks
        if check.status in {"missing", "insufficient"}
        for gap in check.missing
    ]
    core_blocked = any(
        check.component in {"MarketAnalyst", "TradabilityRisk", "BacktestValidator"}
        and check.status in {"missing", "insufficient"}
        for check in checks
    )
    if core_blocked or score < 55:
        status = "insufficient_for_research"
        action_gate = "blocked"
        note = "Core market, tradability or backtest data is insufficient; only data repair is allowed."
    elif hard_gaps:
        status = "research_only"
        action_gate = "no_real_trade"
        note = (
            "Dataset can support comparative research, but missing A-share sector, fund-flow or social "
            "feeds means the system must not output real buy/sell actions."
        )
    else:
        status = "paper_trade_ready"
        action_gate = "paper_trade_allowed"
        note = "Dataset passes the current quality gate for simulated paper-trade validation only."
    report = DataQualityReport(
        trade_date=trade_date,
        status=status,
        action_gate=action_gate,
        score=round(score, 2),
        note=note,
        checks=checks,
        hard_gaps=hard_gaps,
    )
    _write_report(report)
    return report


def load_or_assess_dataset_quality(dataset_dir: Path, trade_date: str) -> DataQualityReport:
    path = Path(f"reports/generated/data_quality_{trade_date}.json")
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return DataQualityReport(
            checks=[DataQualityCheck(**item) for item in data["checks"]],
            **{key: value for key, value in data.items() if key != "checks"},
        )
    return assess_dataset_quality(dataset_dir, trade_date)


def _check_market(dataset_dir: Path) -> DataQualityCheck:
    rows = _read_rows(dataset_dir / CORE_FILES["market"])
    missing = []
    if len(rows) < 100:
        missing.append(f"market Top100 rows are incomplete: {len(rows)}/100")
    required = ["code", "amount_rank", "close", "amount", "ret_5d", "ret_20d", "volatility_20d"]
    missing_cells = _missing_cells(rows, required)
    if missing_cells:
        missing.append(f"market required cells missing: {missing_cells}")
    score = 100.0 if not missing else max(0.0, len(rows))
    return DataQualityCheck(
        component="MarketAnalyst",
        status="ok" if not missing else "insufficient",
        score=min(score, 100.0),
        evidence=[f"market rows={len(rows)}", "technical fields include returns, MA gap, volatility and amount ratio"],
        missing=missing,
        sources=["market_technical.csv / AKShare-EastMoney OHLCV"],
    )


def _check_tradability(dataset_dir: Path) -> DataQualityCheck:
    rows = _read_rows(dataset_dir / CORE_FILES["tradability"])
    missing = []
    if len(rows) < 100:
        missing.append(f"tradability rows are incomplete: {len(rows)}/100")
    note_count = sum(1 for row in rows if row.get("tradability_note"))
    if note_count < len(rows):
        missing.append("some tradability notes are empty")
    score = 100.0 if not missing else max(0.0, note_count)
    return DataQualityCheck(
        component="TradabilityRisk",
        status="ok" if not missing else "insufficient",
        score=min(score, 100.0),
        evidence=[f"tradability rows={len(rows)}", f"rows with note={note_count}"],
        missing=missing,
        sources=["risk_tradability.csv / ST, suspension, limit-up, limit-down pools"],
    )


def _check_fundamentals(dataset_dir: Path) -> DataQualityCheck:
    rows = _read_rows(dataset_dir / CORE_FILES["fundamentals"])
    usable = [
        row
        for row in rows
        if row.get("source_status") in {"ok", "spot_fallback", "valuation_fallback"}
        and (row.get("pe_dynamic") or row.get("pb"))
    ]
    coverage = _coverage(usable, rows)
    missing = [] if coverage >= 0.8 else [f"fundamental valuation coverage is low: {coverage:.0%}"]
    return DataQualityCheck(
        component="FundamentalsAnalyst",
        status="ok" if not missing else "insufficient",
        score=round(coverage * 100, 2),
        evidence=[f"fundamental rows={len(rows)}", f"usable valuation rows={len(usable)}"],
        missing=missing + ["full balance sheet/income/cash-flow fields are not wired yet"],
        sources=["fundamentals.csv / EastMoney individual info and valuation fallback"],
    )


def _check_events(dataset_dir: Path) -> DataQualityCheck:
    rows = _read_rows(dataset_dir / CORE_FILES["events"])
    loaded = [row for row in rows if row.get("event_status") == "loaded"]
    coverage = _coverage(loaded, rows)
    missing = [] if coverage >= 0.6 else [f"news/announcement coverage is low: {coverage:.0%}"]
    return DataQualityCheck(
        component="NewsEventAnalyst",
        status="ok" if not missing else "insufficient",
        score=round(coverage * 100, 2),
        evidence=[f"event rows={len(rows)}", f"loaded event rows={len(loaded)}"],
        missing=missing + ["full announcement body extraction is not wired yet"],
        sources=["events.csv / AKShare stock_news_em and stock_individual_notice_report"],
    )


def _check_backtests(dataset_dir: Path) -> DataQualityCheck:
    rows = _read_rows(dataset_dir / CORE_FILES["backtests"])
    usable = [
        row
        for row in rows
        if row.get("source_status") == "ok" and _to_int(row.get("sample_count")) >= 5
    ]
    coverage = _coverage(usable, rows)
    missing = [] if coverage >= 0.3 else [f"local signal backtest coverage is weak: {coverage:.0%}"]
    return DataQualityCheck(
        component="BacktestValidator",
        status="ok" if not missing else "insufficient",
        score=round(coverage * 100, 2),
        evidence=[f"backtest rows={len(rows)}", f"usable signal rows={len(usable)}"],
        missing=missing + ["current backtest is a simple signal check, not execution-grade simulation"],
        sources=["backtests.csv / local momentum-volume signal replay"],
    )


def _check_macro(trade_date: str) -> DataQualityCheck:
    path = Path(f"reports/generated/macro_regime_{trade_date}.json")
    if not path.exists():
        return DataQualityCheck(
            component="MacroRegimeAgent",
            status="missing",
            score=0.0,
            evidence=[],
            missing=["macro regime file is missing"],
            sources=["macro_regime.json / AKShare macro interfaces"],
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    populated = sum(1 for key, value in data.items() if key not in {"risks", "supports", "sources"} and value not in {None, ""})
    score = min(100.0, populated / 11 * 100)
    return DataQualityCheck(
        component="MacroRegimeAgent",
        status="ok" if score >= 70 else "insufficient",
        score=round(score, 2),
        evidence=[f"macro status={data.get('status')}", f"populated macro fields={populated}"],
        missing=[] if score >= 70 else ["macro fields are sparse"],
        sources=["macro_regime.json / US rates, FX, LPR, SHIBOR, PMI, QVIX"],
    )


def _missing_check(component: str, missing: str, source: str) -> DataQualityCheck:
    return DataQualityCheck(
        component=component,
        status="missing",
        score=0.0,
        evidence=[],
        missing=[missing],
        sources=[source],
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _missing_cells(rows: Iterable[dict[str, str]], required: list[str]) -> int:
    return sum(1 for row in rows for key in required if row.get(key) in {None, ""})


def _coverage(usable: list[dict[str, str]], rows: list[dict[str, str]]) -> float:
    if not rows:
        return 0.0
    return len(usable) / len(rows)


def _to_int(value: str | None) -> int:
    try:
        return int(float(value or 0))
    except ValueError:
        return 0


def _write_report(report: DataQualityReport) -> None:
    output_dir = Path("reports/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"data_quality_{report.trade_date}.json").write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
