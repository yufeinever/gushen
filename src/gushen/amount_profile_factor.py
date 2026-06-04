from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from gushen.amount_rank_factor import (
    DEFAULT_CACHE_DIR,
    DEFAULT_FORWARD_HORIZONS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POOL_FILE,
    DEFAULT_TRADE_DATE,
    AmountRankPoint,
    PoolStock,
    _float_or_none,
    _parse_ints,
    _return_pct,
    build_amount_ranks,
    build_simulation_eval_dates,
    load_pool,
    load_reference_trading_dates,
    offset_trade_date,
    write_dataclass_csv,
)


DEFAULT_PROFILE_OUTPUT_DIR = DEFAULT_OUTPUT_DIR.parent / "amount_profile_factor"


@dataclass(frozen=True)
class AmountProfile:
    ts_code: str
    name: str
    trade_date: str
    amount_rank: int | None
    amount_ratio_20: float | None
    amount_ma20_ma60: float | None
    ret20_pct: float | None
    ret60_pct: float | None
    ma20_gap_pct: float | None
    ma60_gap_pct: float | None
    state: str


@dataclass(frozen=True)
class CurrentProfileRow:
    pool_rank: int
    ts_code: str
    name: str
    trade_date: str
    amount_rank: int | None
    amount_ratio_20: float | None
    amount_ma20_ma60: float | None
    ret20_pct: float | None
    ret60_pct: float | None
    ma20_gap_pct: float | None
    ma60_gap_pct: float | None
    state: str


@dataclass(frozen=True)
class ProfileSimulationRow:
    eval_date: str
    ts_code: str
    name: str
    amount_rank: int
    state: str
    factor: str
    bucket: str
    forward_days: int
    forward_date: str
    forward_return_pct: float


@dataclass(frozen=True)
class ProfileBucketSummary:
    factor: str
    forward_days: int
    bucket: str
    count: int
    avg_return_pct: float
    median_return_pct: float
    win_rate_pct: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research amount-profile filters for current and rolling Top100 pools."
    )
    parser.add_argument("--pool-file", default=str(DEFAULT_POOL_FILE))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_PROFILE_OUTPUT_DIR))
    parser.add_argument("--trade-date", default=DEFAULT_TRADE_DATE)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--forward-horizons", default=",".join(str(x) for x in DEFAULT_FORWARD_HORIZONS))
    parser.add_argument("--simulation-start", default="2024-06-03")
    parser.add_argument("--simulation-step", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_amount_profile_factor_research(
        pool_file=Path(args.pool_file),
        cache_dir=Path(args.cache_dir),
        output_dir=Path(args.output_dir),
        trade_date=args.trade_date,
        limit=args.limit,
        top_n=args.top_n,
        forward_horizons=_parse_ints(args.forward_horizons),
        simulation_start=args.simulation_start,
        simulation_step=args.simulation_step,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def run_amount_profile_factor_research(
    pool_file: Path,
    cache_dir: Path,
    output_dir: Path,
    trade_date: str,
    limit: int,
    top_n: int,
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
        lookbacks=[120],
        forward_horizons=forward_horizons,
        step=simulation_step,
    )
    target_dates = build_profile_target_dates(trading_dates, [trade_date, *eval_dates], forward_horizons)
    points_by_date, points_by_stock, loaded_files = load_profile_points(cache_dir, target_dates)
    ranks_by_date = build_amount_ranks(points_by_date)
    current_rows = build_current_profile_rows(pool, trade_date, trading_dates, points_by_stock, ranks_by_date)
    simulation_rows = run_profile_simulation(
        eval_dates=eval_dates,
        top_n=top_n,
        forward_horizons=forward_horizons,
        trading_dates=trading_dates,
        points_by_stock=points_by_stock,
        ranks_by_date=ranks_by_date,
    )
    summaries = summarize_profile_simulation(simulation_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"amount_profile_factor_{trade_date}"
    current_path = output_dir / f"{stem}_current_top{limit}.csv"
    summary_path = output_dir / f"{stem}_summary.csv"
    rows_path = output_dir / f"{stem}_simulation_rows.csv"
    report_path = output_dir / f"{stem}.json"
    write_dataclass_csv(current_path, current_rows, CurrentProfileRow)
    write_dataclass_csv(summary_path, summaries, ProfileBucketSummary)
    write_dataclass_csv(rows_path, simulation_rows, ProfileSimulationRow)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_sufficiency": {
            "status": "research_only",
            "note": (
                "Uses cached qfq daily bars and amount-profile factors only. It excludes news, "
                "tradability replay, limit rules, execution costs and index benchmark alignment."
            ),
            "pool_size": len(pool),
            "loaded_history_files": loaded_files,
            "trade_date": trade_date,
            "simulation_eval_dates": len(eval_dates),
            "simulation_rows": len(simulation_rows),
        },
        "current_distribution": summarize_current_profiles(current_rows),
        "simulation_summary": [asdict(item) for item in summaries],
        "outputs": {
            "current_top_rows": str(current_path),
            "simulation_summary": str(summary_path),
            "simulation_rows": str(rows_path),
            "report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_profile_target_dates(
    trading_dates: list[str], eval_dates: Iterable[str], forward_horizons: Iterable[int]
) -> set[str]:
    targets: set[str] = set()
    for eval_date in eval_dates:
        if eval_date not in trading_dates:
            continue
        index = trading_dates.index(eval_date)
        for step in range(max(0, index - 120), min(len(trading_dates), index + max(forward_horizons) + 1)):
            targets.add(trading_dates[step])
    return targets


def load_profile_points(
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
                close = _float_or_none(row.get("close"))
                amount = _float_or_none(row.get("amount"))
                if close is None or amount is None or close <= 0 or amount < 0:
                    continue
                point = AmountRankPoint(
                    ts_code=ts_code,
                    name=str(row.get("name") or ts_code),
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


def build_current_profile_rows(
    pool: list[PoolStock],
    trade_date: str,
    trading_dates: list[str],
    points_by_stock: dict[str, dict[str, AmountRankPoint]],
    ranks_by_date: dict[str, dict[str, int]],
) -> list[CurrentProfileRow]:
    rows: list[CurrentProfileRow] = []
    for stock in pool:
        profile = calculate_profile(stock.ts_code, stock.name, trade_date, trading_dates, points_by_stock, ranks_by_date)
        rows.append(
            CurrentProfileRow(
                pool_rank=stock.rank,
                ts_code=stock.ts_code,
                name=stock.name,
                trade_date=trade_date,
                amount_rank=profile.amount_rank if profile else None,
                amount_ratio_20=profile.amount_ratio_20 if profile else None,
                amount_ma20_ma60=profile.amount_ma20_ma60 if profile else None,
                ret20_pct=profile.ret20_pct if profile else None,
                ret60_pct=profile.ret60_pct if profile else None,
                ma20_gap_pct=profile.ma20_gap_pct if profile else None,
                ma60_gap_pct=profile.ma60_gap_pct if profile else None,
                state=profile.state if profile else "missing",
            )
        )
    return rows


def run_profile_simulation(
    eval_dates: Iterable[str],
    top_n: int,
    forward_horizons: Iterable[int],
    trading_dates: list[str],
    points_by_stock: dict[str, dict[str, AmountRankPoint]],
    ranks_by_date: dict[str, dict[str, int]],
) -> list[ProfileSimulationRow]:
    rows: list[ProfileSimulationRow] = []
    for eval_date in eval_dates:
        top_items = sorted(ranks_by_date.get(eval_date, {}).items(), key=lambda item: item[1])[:top_n]
        for ts_code, rank in top_items:
            name = points_by_stock.get(ts_code, {}).get(eval_date).name
            profile = calculate_profile(ts_code, name, eval_date, trading_dates, points_by_stock, ranks_by_date)
            if profile is None:
                continue
            for forward_days in forward_horizons:
                forward_date = offset_trade_date(trading_dates, eval_date, forward_days)
                forward_point = points_by_stock.get(ts_code, {}).get(forward_date or "")
                current_point = points_by_stock.get(ts_code, {}).get(eval_date)
                forward_return = _return_pct(
                    current_point.close if current_point else None,
                    forward_point.close if forward_point else None,
                )
                if forward_date is None or forward_return is None:
                    continue
                for factor, bucket in profile_buckets(profile).items():
                    rows.append(
                        ProfileSimulationRow(
                            eval_date=eval_date,
                            ts_code=ts_code,
                            name=profile.name,
                            amount_rank=rank,
                            state=profile.state,
                            factor=factor,
                            bucket=bucket,
                            forward_days=forward_days,
                            forward_date=forward_date,
                            forward_return_pct=round(forward_return, 4),
                        )
                    )
    return rows


def calculate_profile(
    ts_code: str,
    name: str,
    trade_date: str,
    trading_dates: list[str],
    points_by_stock: dict[str, dict[str, AmountRankPoint]],
    ranks_by_date: dict[str, dict[str, int]],
) -> AmountProfile | None:
    rows = points_by_stock.get(ts_code, {})
    current = rows.get(trade_date)
    if current is None or trade_date not in trading_dates:
        return None
    index = trading_dates.index(trade_date)
    amount_ma20 = _mean_positive(rows.get(date).amount if rows.get(date) else None for date in trading_dates[max(0, index - 20) : index])
    amount_ma60 = _mean_positive(rows.get(date).amount if rows.get(date) else None for date in trading_dates[max(0, index - 60) : index])
    close_ma20 = _mean_positive(rows.get(date).close if rows.get(date) else None for date in trading_dates[max(0, index - 20) : index + 1])
    close_ma60 = _mean_positive(rows.get(date).close if rows.get(date) else None for date in trading_dates[max(0, index - 60) : index + 1])
    date20 = offset_trade_date(trading_dates, trade_date, -20)
    date60 = offset_trade_date(trading_dates, trade_date, -60)
    ret20 = _return_pct(rows.get(date20).close if date20 and rows.get(date20) else None, current.close)
    ret60 = _return_pct(rows.get(date60).close if date60 and rows.get(date60) else None, current.close)
    amount_ratio_20 = _ratio(current.amount, amount_ma20)
    amount_ma20_ma60 = _ratio(amount_ma20, amount_ma60)
    ma20_gap = _return_pct(close_ma20, current.close)
    ma60_gap = _return_pct(close_ma60, current.close)
    return AmountProfile(
        ts_code=ts_code,
        name=name,
        trade_date=trade_date,
        amount_rank=ranks_by_date.get(trade_date, {}).get(ts_code),
        amount_ratio_20=amount_ratio_20,
        amount_ma20_ma60=amount_ma20_ma60,
        ret20_pct=ret20,
        ret60_pct=ret60,
        ma20_gap_pct=ma20_gap,
        ma60_gap_pct=ma60_gap,
        state=profile_state(amount_ratio_20, amount_ma20_ma60, ret20, ret60, ma20_gap),
    )


def profile_state(
    amount_ratio_20: float | None,
    amount_ma20_ma60: float | None,
    ret20_pct: float | None,
    ret60_pct: float | None,
    ma20_gap_pct: float | None,
) -> str:
    if None in (amount_ratio_20, amount_ma20_ma60, ret20_pct, ret60_pct, ma20_gap_pct):
        return "missing"
    if ret60_pct > 20 and amount_ratio_20 >= 4:
        return "late_climax_volume"
    if (
        ret60_pct > 0
        and ret20_pct > 0
        and 1 <= amount_ratio_20 < 3
        and amount_ma20_ma60 >= 1
        and ma20_gap_pct > 0
    ):
        return "steady_price_volume_confirm"
    if ret60_pct > 0 and amount_ratio_20 < 1 and ma20_gap_pct > 0:
        return "quiet_trend"
    if ret60_pct <= 0 and amount_ratio_20 >= 2:
        return "weak_high_volume"
    if ret60_pct > 0 and ret20_pct < 0 and amount_ratio_20 >= 1:
        return "pullback_with_liquidity"
    return "mixed"


def profile_buckets(profile: AmountProfile) -> dict[str, str]:
    return {
        "state": profile.state,
        "amount_ratio_20": amount_ratio_bucket(profile.amount_ratio_20),
        "amount_ma20_ma60": amount_trend_bucket(profile.amount_ma20_ma60),
        "ret20": return_bucket(profile.ret20_pct),
        "ret60": return_bucket(profile.ret60_pct),
        "ma20_gap": gap_bucket(profile.ma20_gap_pct),
        "ma60_gap": gap_bucket(profile.ma60_gap_pct),
    }


def summarize_profile_simulation(rows: list[ProfileSimulationRow]) -> list[ProfileBucketSummary]:
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row.factor, row.forward_days, row.bucket)].append(row.forward_return_pct)
    summaries: list[ProfileBucketSummary] = []
    for key, values in sorted(grouped.items()):
        factor, forward_days, bucket = key
        summaries.append(
            ProfileBucketSummary(
                factor=factor,
                forward_days=forward_days,
                bucket=bucket,
                count=len(values),
                avg_return_pct=round(sum(values) / len(values), 4),
                median_return_pct=round(_median(values), 4),
                win_rate_pct=round(sum(1 for value in values if value > 0) / len(values) * 100, 2),
            )
        )
    return summaries


def summarize_current_profiles(rows: list[CurrentProfileRow]) -> dict[str, dict[str, int]]:
    return {
        "state": dict(Counter(row.state for row in rows)),
        "amount_ratio_20": dict(Counter(amount_ratio_bucket(row.amount_ratio_20) for row in rows)),
        "amount_ma20_ma60": dict(Counter(amount_trend_bucket(row.amount_ma20_ma60) for row in rows)),
        "ret20": dict(Counter(return_bucket(row.ret20_pct) for row in rows)),
        "ret60": dict(Counter(return_bucket(row.ret60_pct) for row in rows)),
        "ma20_gap": dict(Counter(gap_bucket(row.ma20_gap_pct) for row in rows)),
        "ma60_gap": dict(Counter(gap_bucket(row.ma60_gap_pct) for row in rows)),
    }


def amount_ratio_bucket(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < 1:
        return "<1"
    if value < 2:
        return "1-2"
    if value < 4:
        return "2-4"
    return ">=4"


def amount_trend_bucket(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < 0.8:
        return "<0.8"
    if value < 1.2:
        return "0.8-1.2"
    return ">=1.2"


def return_bucket(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < 0:
        return "<0"
    if value < 20:
        return "0-20"
    if value < 50:
        return "20-50"
    return ">=50"


def gap_bucket(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < 0:
        return "<0"
    if value < 10:
        return "0-10"
    if value < 30:
        return "10-30"
    return ">=30"


def _mean_positive(values: Iterable[float | None]) -> float | None:
    items = [value for value in values if value is not None and value > 0]
    if not items:
        return None
    return sum(items) / len(items)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


if __name__ == "__main__":
    main()
