from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, pstdev

from rich.console import Console
from rich.table import Table

from gushen.data import DailyBar, fetch_daily_bars
from gushen.research import load_or_fetch_daily_snapshot


@dataclass(frozen=True)
class DataSufficiency:
    status: str
    available: list[str]
    missing: list[str]
    note: str


@dataclass(frozen=True)
class DeepFeatureRow:
    trade_date: str
    code: str
    name: str
    amount_rank: int
    close: float
    amount: float
    pct_1d: float
    ret_5d: float
    ret_10d: float
    ret_20d: float
    ma5_gap: float
    ma10_gap: float
    ma20_gap: float
    volatility_20d: float
    amount_ratio_5d: float
    turnover: float
    ai_readiness: str
    data_note: str


def run_deep_analysis_pack(trade_date: str = "2026-05-20") -> list[DeepFeatureRow]:
    console = Console()
    sufficiency = assess_data_sufficiency()
    console.print(f"Data sufficiency: [yellow]{sufficiency.status}[/]")
    console.print(sufficiency.note)

    snapshot = load_or_fetch_daily_snapshot(trade_date)
    top100 = sorted(snapshot, key=lambda item: item.amount, reverse=True)[:100]
    history = load_or_fetch_histories(top100, trade_date)
    rows = build_deep_features(top100, history, trade_date)
    write_deep_features(trade_date, rows, sufficiency)
    print_deep_features(console, trade_date, rows)
    if _has_llm_config():
        console.print("[green]LLM config detected, but LLM execution is not enabled yet.[/]")
    else:
        console.print("[yellow]No LLM config detected. Generated AI analysis pack only.[/]")
    return rows


def assess_data_sufficiency() -> DataSufficiency:
    available = [
        "AKShare/EastMoney daily OHLCV",
        "Top100 amount universe",
        "close price filter",
        "60-calendar-day local history fetch for Top100",
    ]
    missing = [
        "exchange announcement/event extraction",
        "ST/suspension verification for target date",
        "limit-up/limit-down tradability for target date",
        "industry/sector strength",
        "fund flow/financing/dragon-tiger data",
        "historical strategy backtest",
        "LLM API configuration",
    ]
    return DataSufficiency(
        status="insufficient_for_recommendation",
        available=available,
        missing=missing,
        note="Only research/observation analysis is allowed until missing data and backtest are added.",
    )


def load_or_fetch_histories(top100: list[DailyBar], trade_date: str) -> dict[str, list[DailyBar]]:
    cache_dir = Path(f"data/local/histories/{trade_date}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    start_date = (date.fromisoformat(trade_date) - timedelta(days=120)).isoformat()
    histories: dict[str, list[DailyBar]] = {}
    to_fetch = []
    by_code = {bar.code: bar for bar in top100}
    for bar in top100:
        path = cache_dir / f"{bar.code}.csv"
        if path.exists():
            histories[bar.code] = _read_bars(path)
        else:
            to_fetch.append(bar)

    if to_fetch:
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = {
                executor.submit(fetch_daily_bars, bar.code, bar.name, start_date, trade_date): bar
                for bar in to_fetch
            }
            for future in as_completed(futures):
                bar = futures[future]
                try:
                    rows = future.result()
                except Exception:
                    rows = [by_code[bar.code]]
                histories[bar.code] = rows
                _write_bars(cache_dir / f"{bar.code}.csv", rows)
    return histories


def build_deep_features(
    top100: list[DailyBar],
    history: dict[str, list[DailyBar]],
    trade_date: str,
) -> list[DeepFeatureRow]:
    amount_rank = {bar.code: index + 1 for index, bar in enumerate(top100)}
    rows = []
    for bar in top100:
        bars = sorted(history.get(bar.code, []), key=lambda item: item.trade_date)
        bars = [item for item in bars if item.trade_date <= trade_date]
        closes = [item.close for item in bars]
        amounts = [item.amount for item in bars]
        pct_changes = [item.pct_change for item in bars]
        row = DeepFeatureRow(
            trade_date=trade_date,
            code=bar.code,
            name=bar.name,
            amount_rank=amount_rank[bar.code],
            close=bar.close,
            amount=bar.amount,
            pct_1d=bar.pct_change,
            ret_5d=_return(closes, 5),
            ret_10d=_return(closes, 10),
            ret_20d=_return(closes, 20),
            ma5_gap=_ma_gap(closes, 5),
            ma10_gap=_ma_gap(closes, 10),
            ma20_gap=_ma_gap(closes, 20),
            volatility_20d=_volatility(pct_changes, 20),
            amount_ratio_5d=_amount_ratio(amounts, 5),
            turnover=bar.turnover,
            ai_readiness="research_only",
            data_note="missing event/tradability/backtest data",
        )
        rows.append(row)
    return sorted(rows, key=lambda item: item.amount_rank)


def write_deep_features(
    trade_date: str,
    rows: list[DeepFeatureRow],
    sufficiency: DataSufficiency,
) -> None:
    output_dir = Path("reports/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"top100_deep_features_{trade_date}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(DeepFeatureRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    json_path = output_dir / f"top100_ai_analysis_pack_{trade_date}.json"
    payload = {
        "trade_date": trade_date,
        "data_sufficiency": asdict(sufficiency),
        "instructions": (
            "Do not recommend buy/sell. Only produce research observations until missing data "
            "and backtest are available."
        ),
        "rows": [asdict(row) for row in rows],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_deep_features(console: Console, trade_date: str, rows: list[DeepFeatureRow]) -> None:
    table = Table(title=f"{trade_date} Top100 deep feature pack")
    table.add_column("Rank", justify="right")
    table.add_column("Code")
    table.add_column("Name")
    table.add_column("Close", justify="right")
    table.add_column("Ret5", justify="right")
    table.add_column("Ret20", justify="right")
    table.add_column("Vol20", justify="right")
    table.add_column("Amt5x", justify="right")
    for row in rows[:20]:
        table.add_row(
            str(row.amount_rank),
            row.code,
            row.name,
            f"{row.close:.2f}",
            f"{row.ret_5d:.2%}",
            f"{row.ret_20d:.2%}",
            f"{row.volatility_20d:.2%}",
            f"{row.amount_ratio_5d:.2f}",
        )
    console.print(table)
    console.print(f"Feature rows: {len(rows)}")
    console.print(f"Pack: reports/generated/top100_ai_analysis_pack_{trade_date}.json")


def _read_bars(path: Path) -> list[DailyBar]:
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


def _write_bars(path: Path, bars: list[DailyBar]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(DailyBar.__dataclass_fields__))
        writer.writeheader()
        for bar in bars:
            writer.writerow(asdict(bar))


def _return(values: list[float], periods: int) -> float:
    if len(values) <= periods or values[-periods - 1] == 0:
        return 0.0
    return values[-1] / values[-periods - 1] - 1


def _ma_gap(values: list[float], periods: int) -> float:
    if len(values) < periods:
        return 0.0
    avg = mean(values[-periods:])
    return values[-1] / avg - 1 if avg else 0.0


def _volatility(values: list[float], periods: int) -> float:
    sample = values[-periods:]
    if len(sample) < 2:
        return 0.0
    return pstdev(sample)


def _amount_ratio(values: list[float], periods: int) -> float:
    if len(values) < periods + 1:
        return 1.0
    avg = mean(values[-periods - 1 : -1])
    return values[-1] / avg if avg else 1.0


def _has_llm_config() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_BASE_URL"))


def main() -> None:
    run_deep_analysis_pack()


if __name__ == "__main__":
    main()
