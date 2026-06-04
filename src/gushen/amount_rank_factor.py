from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from gushen.guided_factor_backtest import normalize_stock_code, to_ts_code


DEFAULT_POOL_FILE = Path("data/local/Table_4860_2026-06-03.xlsx")
DEFAULT_CACHE_DIR = Path("data/local/guided_factor_backtests/daily_bars/qfq")
DEFAULT_OUTPUT_DIR = Path("reports/generated/amount_rank_factor")
DEFAULT_TRADE_DATE = "2026-06-03"
DEFAULT_LOOKBACKS = (100, 200)
DEFAULT_FORWARD_HORIZONS = (20, 60)


@dataclass(frozen=True)
class PoolStock:
    rank: int
    code: str
    ts_code: str
    name: str


@dataclass(frozen=True)
class AmountRankPoint:
    ts_code: str
    name: str
    trade_date: str
    amount: float
    close: float


@dataclass(frozen=True)
class CurrentAmountRankRow:
    pool_rank: int
    code: str
    ts_code: str
    name: str
    trade_date: str
    current_amount_rank: int | None
    current_amount: float | None
    current_close: float | None
    lookback_days: int
    anchor_date: str
    anchor_amount_rank: int | None
    anchor_amount: float | None
    anchor_close: float | None
    rank_improvement: int | None
    amount_ratio: float | None
    close_return_pct: float | None
    rank_bucket: str
    improvement_bucket: str


@dataclass(frozen=True)
class BucketSummary:
    kind: str
    lookback_days: int
    forward_days: int
    bucket: str
    count: int
    avg_return_pct: float | None
    median_return_pct: float | None
    win_rate_pct: float | None


@dataclass(frozen=True)
class SimulationRow:
    eval_date: str
    ts_code: str
    name: str
    current_rank: int
    lookback_days: int
    anchor_date: str
    anchor_rank: int | None
    rank_improvement: int | None
    rank_bucket: str
    improvement_bucket: str
    forward_days: int
    forward_date: str
    forward_return_pct: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research historical amount-rank factors for current or rolling Top100 pools."
    )
    parser.add_argument("--pool-file", default=str(DEFAULT_POOL_FILE))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--trade-date", default=DEFAULT_TRADE_DATE)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--lookbacks", default=",".join(str(item) for item in DEFAULT_LOOKBACKS))
    parser.add_argument(
        "--forward-horizons",
        default=",".join(str(item) for item in DEFAULT_FORWARD_HORIZONS),
    )
    parser.add_argument(
        "--simulation-step",
        type=int,
        default=20,
        help="Evaluate rolling TopN every N trading days in the holdout window.",
    )
    parser.add_argument(
        "--simulation-start",
        default="2024-06-03",
        help="First rolling evaluation date for factor simulation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lookbacks = _parse_ints(args.lookbacks)
    forward_horizons = _parse_ints(args.forward_horizons)
    output = run_amount_rank_factor_research(
        pool_file=Path(args.pool_file),
        cache_dir=Path(args.cache_dir),
        output_dir=Path(args.output_dir),
        trade_date=args.trade_date,
        limit=args.limit,
        top_n=args.top_n,
        lookbacks=lookbacks,
        forward_horizons=forward_horizons,
        simulation_start=args.simulation_start,
        simulation_step=args.simulation_step,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


def run_amount_rank_factor_research(
    pool_file: Path,
    cache_dir: Path,
    output_dir: Path,
    trade_date: str,
    limit: int,
    top_n: int,
    lookbacks: list[int],
    forward_horizons: list[int],
    simulation_start: str,
    simulation_step: int,
) -> dict[str, Any]:
    pool = load_pool(pool_file, limit)
    trading_dates = load_reference_trading_dates(cache_dir, trade_date)
    eval_dates = build_simulation_eval_dates(
        trading_dates=trading_dates,
        simulation_start=simulation_start,
        trade_date=trade_date,
        lookbacks=lookbacks,
        forward_horizons=forward_horizons,
        step=simulation_step,
    )
    current_target_dates = build_current_target_dates(trading_dates, trade_date, lookbacks)
    simulation_target_dates = build_simulation_target_dates(
        trading_dates, eval_dates, lookbacks, forward_horizons
    )
    target_dates = current_target_dates | simulation_target_dates
    points_by_date, points_by_stock, loaded_files = load_target_points(cache_dir, target_dates)
    ranks_by_date = build_amount_ranks(points_by_date)
    current_rows = build_current_amount_rank_rows(
        pool=pool,
        trade_date=trade_date,
        lookbacks=lookbacks,
        trading_dates=trading_dates,
        points_by_stock=points_by_stock,
        ranks_by_date=ranks_by_date,
    )
    simulation_rows = run_rolling_top_amount_simulation(
        eval_dates=eval_dates,
        top_n=top_n,
        lookbacks=lookbacks,
        forward_horizons=forward_horizons,
        trading_dates=trading_dates,
        points_by_stock=points_by_stock,
        ranks_by_date=ranks_by_date,
    )
    summaries = summarize_simulation(simulation_rows)
    current_distribution = summarize_current_distribution(current_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"amount_rank_factor_{trade_date}"
    current_path = output_dir / f"{stem}_current_top{limit}.csv"
    summary_path = output_dir / f"{stem}_simulation_summary.csv"
    simulation_path = output_dir / f"{stem}_simulation_rows.csv"
    report_path = output_dir / f"{stem}.json"
    write_dataclass_csv(current_path, current_rows, CurrentAmountRankRow)
    write_dataclass_csv(summary_path, summaries, BucketSummary)
    write_dataclass_csv(simulation_path, simulation_rows, SimulationRow)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_sufficiency": {
            "status": "research_only",
            "note": (
                "Uses cached qfq daily bars and cross-sectional amount ranks. "
                "It does not include announcements, tradability replay, limit rules or execution costs."
            ),
            "pool_size": len(pool),
            "loaded_history_files": loaded_files,
            "trade_date": trade_date,
            "lookbacks": lookbacks,
            "forward_horizons": forward_horizons,
            "simulation_eval_dates": len(eval_dates),
        },
        "current_distribution": current_distribution,
        "simulation_summary": [asdict(item) for item in summaries],
        "outputs": {
            "current_top_rows": str(current_path),
            "simulation_summary": str(summary_path),
            "simulation_rows": str(simulation_path),
            "report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def load_pool(path: Path, limit: int) -> list[PoolStock]:
    frame = _read_pool_frame(path)
    columns = {str(column).strip(): column for column in frame.columns}
    code_col = _pick_required_column(columns, ["code", "代码", "证券代码"])
    name_col = _pick_required_column(columns, ["name", "名称", "证券名称"])
    rank_col = _pick_optional_column(columns, ["rank", "amount_rank", "序", "排名"])
    if rank_col is not None:
        frame = frame.sort_values(rank_col, kind="stable")
    stocks: list[PoolStock] = []
    seen: set[str] = set()
    for index, row in enumerate(frame.head(limit).to_dict("records"), start=1):
        code = normalize_stock_code(str(row.get(code_col) or ""))
        if not code or code in seen:
            continue
        seen.add(code)
        try:
            rank = int(float(row.get(rank_col))) if rank_col is not None else index
        except (TypeError, ValueError):
            rank = index
        stocks.append(
            PoolStock(
                rank=rank,
                code=code,
                ts_code=to_ts_code(code),
                name=str(row.get(name_col) or "").strip(),
            )
        )
    return stocks


def _read_pool_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _pick_required_column(columns: dict[str, Any], names: list[str]) -> Any:
    column = _pick_optional_column(columns, names)
    if column is None:
        raise ValueError(f"missing required column, expected one of: {', '.join(names)}")
    return column


def _pick_optional_column(columns: dict[str, Any], names: list[str]) -> Any | None:
    lowered = {key.lower(): value for key, value in columns.items()}
    for name in names:
        if name in columns:
            return columns[name]
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def load_reference_trading_dates(cache_dir: Path, trade_date: str) -> list[str]:
    preferred = sorted(cache_dir.glob(f"000001.*_*_{trade_date}.csv"))
    candidates = preferred or sorted(cache_dir.glob(f"*_*_{trade_date}.csv"))[:20]
    best_dates: list[str] = []
    for path in candidates:
        dates = read_dates(path)
        if len(dates) > len(best_dates):
            best_dates = dates
    if not best_dates:
        raise RuntimeError(f"no cached daily bar files found under {cache_dir}")
    return sorted(date for date in best_dates if date <= trade_date)


def read_dates(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as file:
        return [row["trade_date"] for row in csv.DictReader(file) if row.get("trade_date")]


def build_current_target_dates(
    trading_dates: list[str], trade_date: str, lookbacks: Iterable[int]
) -> set[str]:
    targets = {trade_date}
    for lookback in lookbacks:
        anchor = offset_trade_date(trading_dates, trade_date, -lookback)
        if anchor is not None:
            targets.add(anchor)
    return targets


def build_simulation_eval_dates(
    trading_dates: list[str],
    simulation_start: str,
    trade_date: str,
    lookbacks: list[int],
    forward_horizons: list[int],
    step: int,
) -> list[str]:
    min_lookback = max(lookbacks)
    max_forward = max(forward_horizons)
    dates = [date for date in trading_dates if simulation_start <= date <= trade_date]
    selected: list[str] = []
    start_index = min_lookback
    end_index = len(trading_dates) - max_forward - 1
    allowed = set(dates)
    for index in range(start_index, max(start_index, end_index + 1), max(1, step)):
        date = trading_dates[index]
        if date in allowed:
            selected.append(date)
    return selected


def build_simulation_target_dates(
    trading_dates: list[str],
    eval_dates: Iterable[str],
    lookbacks: Iterable[int],
    forward_horizons: Iterable[int],
) -> set[str]:
    targets: set[str] = set()
    for eval_date in eval_dates:
        targets.add(eval_date)
        for lookback in lookbacks:
            anchor = offset_trade_date(trading_dates, eval_date, -lookback)
            if anchor is not None:
                targets.add(anchor)
        for horizon in forward_horizons:
            forward = offset_trade_date(trading_dates, eval_date, horizon)
            if forward is not None:
                targets.add(forward)
    return targets


def offset_trade_date(trading_dates: list[str], trade_date: str, offset: int) -> str | None:
    try:
        index = trading_dates.index(trade_date)
    except ValueError:
        eligible = [date for date in trading_dates if date <= trade_date]
        if not eligible:
            return None
        index = trading_dates.index(eligible[-1])
    target = index + offset
    if target < 0 or target >= len(trading_dates):
        return None
    return trading_dates[target]


def load_target_points(
    cache_dir: Path, target_dates: set[str]
) -> tuple[dict[str, dict[str, AmountRankPoint]], dict[str, dict[str, AmountRankPoint]], int]:
    points_by_date: dict[str, dict[str, AmountRankPoint]] = defaultdict(dict)
    points_by_stock: dict[str, dict[str, AmountRankPoint]] = defaultdict(dict)
    loaded_files = 0
    for path in sorted(cache_dir.glob("*.csv")):
        ts_code = path.name.split("_")[0]
        loaded = False
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                trade_date = row.get("trade_date", "")
                if trade_date not in target_dates:
                    continue
                amount = _float_or_none(row.get("amount"))
                close = _float_or_none(row.get("close"))
                if amount is None or close is None or close <= 0:
                    continue
                point = AmountRankPoint(
                    ts_code=ts_code,
                    name=str(row.get("name") or ""),
                    trade_date=trade_date,
                    amount=amount,
                    close=close,
                )
                points_by_date[trade_date][ts_code] = point
                points_by_stock[ts_code][trade_date] = point
                loaded = True
        if loaded:
            loaded_files += 1
    return dict(points_by_date), dict(points_by_stock), loaded_files


def build_amount_ranks(
    points_by_date: dict[str, dict[str, AmountRankPoint]]
) -> dict[str, dict[str, int]]:
    ranks_by_date: dict[str, dict[str, int]] = {}
    for trade_date, points in points_by_date.items():
        ranked = sorted(
            (point for point in points.values() if point.amount > 0),
            key=lambda item: item.amount,
            reverse=True,
        )
        ranks_by_date[trade_date] = {
            point.ts_code: rank for rank, point in enumerate(ranked, start=1)
        }
    return ranks_by_date


def build_current_amount_rank_rows(
    pool: list[PoolStock],
    trade_date: str,
    lookbacks: Iterable[int],
    trading_dates: list[str],
    points_by_stock: dict[str, dict[str, AmountRankPoint]],
    ranks_by_date: dict[str, dict[str, int]],
) -> list[CurrentAmountRankRow]:
    rows: list[CurrentAmountRankRow] = []
    for stock in pool:
        current_point = points_by_stock.get(stock.ts_code, {}).get(trade_date)
        current_rank = ranks_by_date.get(trade_date, {}).get(stock.ts_code)
        for lookback in lookbacks:
            anchor_date = offset_trade_date(trading_dates, trade_date, -lookback)
            anchor_point = (
                points_by_stock.get(stock.ts_code, {}).get(anchor_date) if anchor_date else None
            )
            anchor_rank = ranks_by_date.get(anchor_date or "", {}).get(stock.ts_code)
            rows.append(
                CurrentAmountRankRow(
                    pool_rank=stock.rank,
                    code=stock.code,
                    ts_code=stock.ts_code,
                    name=stock.name,
                    trade_date=trade_date,
                    current_amount_rank=current_rank,
                    current_amount=current_point.amount if current_point else None,
                    current_close=current_point.close if current_point else None,
                    lookback_days=lookback,
                    anchor_date=anchor_date or "",
                    anchor_amount_rank=anchor_rank,
                    anchor_amount=anchor_point.amount if anchor_point else None,
                    anchor_close=anchor_point.close if anchor_point else None,
                    rank_improvement=_rank_improvement(anchor_rank, current_rank),
                    amount_ratio=_safe_ratio(
                        current_point.amount if current_point else None,
                        anchor_point.amount if anchor_point else None,
                    ),
                    close_return_pct=_return_pct(
                        anchor_point.close if anchor_point else None,
                        current_point.close if current_point else None,
                    ),
                    rank_bucket=rank_bucket(anchor_rank),
                    improvement_bucket=improvement_bucket(
                        _rank_improvement(anchor_rank, current_rank)
                    ),
                )
            )
    return rows


def run_rolling_top_amount_simulation(
    eval_dates: Iterable[str],
    top_n: int,
    lookbacks: Iterable[int],
    forward_horizons: Iterable[int],
    trading_dates: list[str],
    points_by_stock: dict[str, dict[str, AmountRankPoint]],
    ranks_by_date: dict[str, dict[str, int]],
) -> list[SimulationRow]:
    rows: list[SimulationRow] = []
    for eval_date in eval_dates:
        current_rank_map = ranks_by_date.get(eval_date, {})
        top_items = sorted(current_rank_map.items(), key=lambda item: item[1])[:top_n]
        for ts_code, current_rank in top_items:
            current_point = points_by_stock.get(ts_code, {}).get(eval_date)
            if current_point is None:
                continue
            for lookback in lookbacks:
                anchor_date = offset_trade_date(trading_dates, eval_date, -lookback)
                anchor_rank = ranks_by_date.get(anchor_date or "", {}).get(ts_code)
                change = _rank_improvement(anchor_rank, current_rank)
                for horizon in forward_horizons:
                    forward_date = offset_trade_date(trading_dates, eval_date, horizon)
                    forward_point = (
                        points_by_stock.get(ts_code, {}).get(forward_date) if forward_date else None
                    )
                    forward_return = _return_pct(
                        current_point.close,
                        forward_point.close if forward_point else None,
                    )
                    if forward_date is None or forward_return is None:
                        continue
                    rows.append(
                        SimulationRow(
                            eval_date=eval_date,
                            ts_code=ts_code,
                            name=current_point.name,
                            current_rank=current_rank,
                            lookback_days=lookback,
                            anchor_date=anchor_date or "",
                            anchor_rank=anchor_rank,
                            rank_improvement=change,
                            rank_bucket=rank_bucket(anchor_rank),
                            improvement_bucket=improvement_bucket(change),
                            forward_days=horizon,
                            forward_date=forward_date,
                            forward_return_pct=round(forward_return, 4),
                        )
                    )
    return rows


def summarize_simulation(rows: list[SimulationRow]) -> list[BucketSummary]:
    grouped: dict[tuple[str, int, int, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[("anchor_rank", row.lookback_days, row.forward_days, row.rank_bucket)].append(
            row.forward_return_pct
        )
        grouped[("rank_improvement", row.lookback_days, row.forward_days, row.improvement_bucket)].append(
            row.forward_return_pct
        )
    summaries: list[BucketSummary] = []
    for key, values in sorted(grouped.items()):
        kind, lookback, forward, bucket = key
        summaries.append(
            BucketSummary(
                kind=kind,
                lookback_days=lookback,
                forward_days=forward,
                bucket=bucket,
                count=len(values),
                avg_return_pct=round(sum(values) / len(values), 4) if values else None,
                median_return_pct=round(statistics.median(values), 4) if values else None,
                win_rate_pct=round(sum(1 for value in values if value > 0) / len(values) * 100, 2)
                if values
                else None,
            )
        )
    return summaries


def summarize_current_distribution(rows: list[CurrentAmountRankRow]) -> dict[str, Any]:
    by_lookback: dict[int, Counter[str]] = defaultdict(Counter)
    by_improvement: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_lookback[row.lookback_days][row.rank_bucket] += 1
        by_improvement[row.lookback_days][row.improvement_bucket] += 1
    return {
        "anchor_rank_buckets": {str(key): dict(value) for key, value in by_lookback.items()},
        "rank_improvement_buckets": {str(key): dict(value) for key, value in by_improvement.items()},
    }


def rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "missing"
    if rank <= 100:
        return "001-100"
    if rank <= 300:
        return "101-300"
    if rank <= 1000:
        return "301-1000"
    return "1001+"


def improvement_bucket(change: int | None) -> str:
    if change is None:
        return "missing"
    if change <= 0:
        return "no_improvement"
    if change <= 300:
        return "001-300"
    if change <= 1000:
        return "301-1000"
    return "1001+"


def write_dataclass_csv(path: Path, rows: list[Any], row_type: type[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row_type.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _rank_improvement(anchor_rank: int | None, current_rank: int | None) -> int | None:
    if anchor_rank is None or current_rank is None:
        return None
    return anchor_rank - current_rank


def _return_pct(start: float | None, end: float | None) -> float | None:
    if start is None or end is None or start <= 0:
        return None
    return round((end / start - 1.0) * 100.0, 4)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _float_or_none(value: object) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_ints(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("expected at least one integer")
    return items


if __name__ == "__main__":
    main()
