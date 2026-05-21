from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

from rich.console import Console
from rich.table import Table

from gushen.data import DailyBar, fetch_a_share_code_names, fetch_daily_bar


@dataclass(frozen=True)
class AnalysisRow:
    trade_date: str
    code: str
    name: str
    amount_rank: int
    close: float
    amount: float
    pct_change: float
    amplitude: float
    turnover: float
    score: float
    action: str
    reason: str


def run_yesterday_top100_price_under_50(trade_date: str = "2026-05-20") -> list[AnalysisRow]:
    console = Console()
    bars = load_or_fetch_daily_snapshot(trade_date=trade_date)
    top100 = sorted(bars, key=lambda item: item.amount, reverse=True)[:100]
    filtered = [bar for bar in top100 if bar.close < 50]
    rows = score_candidates(filtered, top100)
    write_analysis(trade_date, rows)
    print_analysis(console, trade_date, rows)
    return rows


def load_or_fetch_daily_snapshot(trade_date: str) -> list[DailyBar]:
    cache_path = Path(f"data/local/snapshots/a_share_daily_{trade_date}.csv")
    if cache_path.exists():
        return read_snapshot(cache_path)

    codes = fetch_a_share_code_names()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    bars: list[DailyBar] = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(fetch_daily_bar, code, name, trade_date): (code, name)
            for code, name in codes
            if not _is_excluded_name(name)
        }
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                bar = future.result()
            except Exception:
                bar = None
            if bar is not None:
                bars.append(bar)
            if index % 500 == 0:
                print(f"Fetched {index}/{len(futures)} symbols, valid bars={len(bars)}")

    write_snapshot(cache_path, bars)
    return bars


def score_candidates(candidates: list[DailyBar], top100: list[DailyBar]) -> list[AnalysisRow]:
    amounts = [bar.amount for bar in top100]
    turnovers = [bar.turnover for bar in top100 if bar.turnover > 0]
    amount_min, amount_max = min(amounts), max(amounts)
    turnover_mean = mean(turnovers) if turnovers else 0.0
    turnover_std = pstdev(turnovers) if len(turnovers) > 1 else 1.0

    rows = []
    amount_rank = {bar.code: index + 1 for index, bar in enumerate(top100)}
    for bar in candidates:
        liquidity_score = _scale(bar.amount, amount_min, amount_max)
        momentum_score = _score_momentum(bar.pct_change)
        volatility_score = _score_amplitude(bar.amplitude)
        turnover_score = _score_turnover(bar.turnover, turnover_mean, turnover_std)
        price_score = 1.0 if 5 <= bar.close < 50 else 0.3
        score = (
            liquidity_score * 30
            + momentum_score * 25
            + volatility_score * 20
            + turnover_score * 15
            + price_score * 10
        )
        action = "paper_watch" if score >= 70 else "research" if score >= 55 else "observe"
        reason = (
            f"amount_rank={amount_rank[bar.code]}, pct={bar.pct_change:.2%}, "
            f"amplitude={bar.amplitude:.2%}, turnover={bar.turnover:.2%}"
        )
        rows.append(
            AnalysisRow(
                trade_date=bar.trade_date,
                code=bar.code,
                name=bar.name,
                amount_rank=amount_rank[bar.code],
                close=bar.close,
                amount=bar.amount,
                pct_change=bar.pct_change,
                amplitude=bar.amplitude,
                turnover=bar.turnover,
                score=round(score, 2),
                action=action,
                reason=reason,
            )
        )

    return sorted(rows, key=lambda item: item.score, reverse=True)


def write_snapshot(path: Path, bars: list[DailyBar]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(DailyBar.__dataclass_fields__))
        writer.writeheader()
        for bar in bars:
            writer.writerow(bar.__dict__)


def read_snapshot(path: Path) -> list[DailyBar]:
    with path.open(newline="", encoding="utf-8") as file:
        return [
            DailyBar(
                trade_date=row["trade_date"],
                code=row["code"],
                name=row["name"],
                open=float(row["open"]),
                close=float(row["close"]),
                high=float(row["high"]),
                low=float(row["low"]),
                volume=float(row["volume"]),
                amount=float(row["amount"]),
                amplitude=float(row["amplitude"]),
                pct_change=float(row["pct_change"]),
                turnover=float(row["turnover"]),
            )
            for row in csv.DictReader(file)
        ]


def write_analysis(trade_date: str, rows: list[AnalysisRow]) -> None:
    output = Path(f"reports/generated/top100_under50_analysis_{trade_date}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(AnalysisRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def print_analysis(console: Console, trade_date: str, rows: list[AnalysisRow]) -> None:
    table = Table(title=f"{trade_date} Top100 amount, close < 50 analysis")
    table.add_column("Rank", justify="right")
    table.add_column("Code")
    table.add_column("Name")
    table.add_column("Close", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Action")
    table.add_column("Reason")
    for row in rows[:15]:
        table.add_row(
            str(row.amount_rank),
            row.code,
            row.name,
            f"{row.close:.2f}",
            f"{row.score:.2f}",
            row.action,
            row.reason,
        )
    console.print(table)
    console.print(f"Candidates close < 50: {len(rows)}")
    console.print("Report: reports/generated/top100_under50_analysis_" + trade_date + ".csv")


def _scale(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _score_momentum(pct_change: float) -> float:
    if pct_change <= -0.03 or pct_change >= 0.095:
        return 0.2
    if 0.005 <= pct_change <= 0.06:
        return 1.0
    if -0.01 <= pct_change < 0.005:
        return 0.7
    return 0.5


def _score_amplitude(amplitude: float) -> float:
    if amplitude <= 0.015:
        return 0.4
    if amplitude <= 0.06:
        return 1.0
    if amplitude <= 0.09:
        return 0.7
    return 0.3


def _score_turnover(turnover: float, avg: float, std: float) -> float:
    if turnover <= 0:
        return 0.5
    z = (turnover - avg) / (std or 1.0)
    if -0.5 <= z <= 1.5:
        return 1.0
    if 1.5 < z <= 3:
        return 0.7
    return 0.4


def _is_excluded_name(name: str) -> bool:
    upper = name.upper()
    return "ST" in upper or "\u9000" in name


def main() -> None:
    run_yesterday_top100_price_under_50()


if __name__ == "__main__":
    main()
