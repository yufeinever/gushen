from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from gushen.trade_calendar import latest_research_trade_date


@dataclass(frozen=True)
class CandidatePlan:
    trade_date: str
    code: str
    name: str
    amount_rank: int
    close: float
    action: str
    data_sufficiency: str
    score: float
    entry_logic: str
    exit_logic: str
    position_logic: str
    evidence: str
    risks: str
    missing_data: str


def build_candidate_plans(trade_date: str | None = None, limit: int = 10) -> list[CandidatePlan]:
    console = Console()
    trade_date = trade_date or latest_research_trade_date()
    dataset_dir = Path(f"reports/generated/tradingagents_dataset_{trade_date}")
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)

    market = _by_code(_read_csv(dataset_dir / "market_technical.csv"))
    risk = _by_code(_read_csv(dataset_dir / "risk_tradability.csv"))
    fundamentals = _by_code(_read_csv(dataset_dir / "fundamentals.csv"))
    events = _by_code(_read_csv(dataset_dir / "events.csv"))
    backtests = _by_code(_read_csv(dataset_dir / "backtests.csv"))

    rows = [
        _score_candidate(code, market[code], risk, fundamentals, events, backtests)
        for code in market
        if float(market[code]["close"]) < 50
    ]
    rows = sorted(rows, key=lambda item: item.score, reverse=True)[:limit]
    _write_rows(trade_date, rows)
    _print_rows(console, trade_date, rows)
    return rows


def _score_candidate(
    code: str,
    market_row: dict[str, str],
    risk_rows: dict[str, dict[str, str]],
    fundamental_rows: dict[str, dict[str, str]],
    event_rows: dict[str, dict[str, str]],
    backtest_rows: dict[str, dict[str, str]],
) -> CandidatePlan:
    risk = risk_rows.get(code, {})
    fundamentals = fundamental_rows.get(code, {})
    events = event_rows.get(code, {})
    backtest = backtest_rows.get(code, {})

    missing = []
    risks = []
    score = 0.0
    if risk.get("tradability_note") != "normal":
        risks.append(f"交易限制={risk.get('tradability_note', 'missing')}")
    else:
        score += 15
    if fundamentals.get("source_status") not in {"ok", "spot_fallback", "valuation_fallback"}:
        missing.append("基本面字段缺失")
    else:
        score += 15
    if events.get("event_status") != "loaded":
        missing.append("事件/公告/新闻缺失")
    else:
        score += 15
        event_text = events.get("event_summary", "")
        if any(token in event_text for token in ["亏损", "减持", "风险提示", "问询函", "立案"]):
            risks.append("事件面存在亏损/减持/监管等风险词")
    if backtest.get("source_status") != "ok":
        missing.append(f"回测不足={backtest.get('source_status', 'missing')}")
    else:
        score += 20

    ret_5d = _float(market_row.get("ret_5d"))
    ret_20d = _float(market_row.get("ret_20d"))
    ma5_gap = _float(market_row.get("ma5_gap"))
    amount_ratio_5d = _float(market_row.get("amount_ratio_5d"))
    volatility_20d = _float(market_row.get("volatility_20d"))
    if 0.02 <= ret_5d <= 0.18:
        score += 10
    elif ret_5d > 0.25:
        risks.append("5日涨幅过热")
    elif ret_5d < -0.08:
        risks.append("5日趋势破坏")
    if 0.03 <= ret_20d <= 0.35:
        score += 7
    elif ret_20d > 0.6:
        risks.append("20日涨幅过热")
    elif ret_20d < -0.12:
        risks.append("20日趋势偏弱")
    if 0 <= ma5_gap <= 0.08:
        score += 6
    elif ma5_gap > 0.12:
        risks.append("价格偏离5日均线过大")
    if 1.1 <= amount_ratio_5d <= 2.8:
        score += 7
    if volatility_20d <= 0.055:
        score += 5
    if volatility_20d > 0.08:
        risks.append("20日波动过高")

    bt_sample = _int(backtest.get("sample_count"))
    bt_avg_3d = _float(backtest.get("avg_return_3d"))
    bt_win_3d = _float(backtest.get("win_rate_3d"))
    if backtest.get("source_status") == "ok" and (bt_sample < 5 or bt_avg_3d <= 0 or bt_win_3d < 0.6):
        risks.append("本地信号回测优势不足")

    data_sufficiency = "sufficient_for_paper_validation" if not missing and not risks else "insufficient"
    action = "模拟验证" if data_sufficiency == "sufficient_for_paper_validation" and score >= 65 else "研究观察"
    if missing:
        action = "数据不足"
    if risks and risk.get("tradability_note") != "normal":
        action = "回避"

    close = _float(market_row.get("close"))
    stop_loss = max(0.035, min(0.07, volatility_20d * 1.5 if volatility_20d else 0.05))
    take_profit = max(0.06, stop_loss * 1.8)
    entry = (
        "仅模拟：次一交易日不涨停且仍在成交额Top100；开盘后30分钟不跌破前收，"
        "价格回踩不破5日均线或重新放量上穿分时均价时记录纸面入场。"
    )
    exit_logic = (
        f"仅模拟：跌破入场价{stop_loss:.1%}止损；盈利达到{take_profit:.1%}分批止盈；"
        "3个交易日未走强减仓观察，5个交易日强制复盘退出。"
    )
    position = "单票模拟仓不超过5%，同题材合计不超过10%；未完成复盘前不得实盘。"
    evidence = (
        f"rank={market_row.get('amount_rank')}, ret5={ret_5d:.2%}, ret20={ret_20d:.2%}, "
        f"amt5x={amount_ratio_5d:.2f}, bt_n={bt_sample}, "
        f"bt_3d_avg={bt_avg_3d:.2%}, bt_3d_win={bt_win_3d:.2%}"
    )
    return CandidatePlan(
        trade_date=market_row["trade_date"],
        code=code,
        name=market_row["name"],
        amount_rank=_int(market_row["amount_rank"]),
        close=close,
        action=action,
        data_sufficiency=data_sufficiency,
        score=round(score, 2),
        entry_logic=entry,
        exit_logic=exit_logic,
        position_logic=position,
        evidence=evidence,
        risks="; ".join(risks) if risks else "none",
        missing_data="; ".join(missing) if missing else "none",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _by_code(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["code"]: row for row in rows}


def _write_rows(trade_date: str, rows: list[CandidatePlan]) -> None:
    output = Path(f"reports/generated/candidate_plans_{trade_date}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(CandidatePlan.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _print_rows(console: Console, trade_date: str, rows: list[CandidatePlan]) -> None:
    table = Table(title=f"{trade_date} Top100 close<50 candidate plans")
    table.add_column("Rank", justify="right")
    table.add_column("Code")
    table.add_column("Name")
    table.add_column("Action")
    table.add_column("Sufficiency")
    table.add_column("Score", justify="right")
    table.add_column("Evidence")
    for row in rows:
        table.add_row(
            str(row.amount_rank),
            row.code,
            row.name,
            row.action,
            row.data_sufficiency,
            f"{row.score:.2f}",
            row.evidence,
        )
    console.print(table)
    console.print(f"Report: reports/generated/candidate_plans_{trade_date}.csv")


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


def main() -> None:
    build_candidate_plans()


if __name__ == "__main__":
    main()
