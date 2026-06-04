from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from gushen.amount_profile_factor import (
    calculate_profile,
    load_profile_points,
)
from gushen.amount_rank_factor import (
    DEFAULT_CACHE_DIR,
    DEFAULT_POOL_FILE,
    DEFAULT_TRADE_DATE,
    AmountRankPoint,
    _parse_ints,
    _return_pct,
    build_amount_ranks,
    load_reference_trading_dates,
    offset_trade_date,
)

DEFAULT_OUTPUT_DIR = Path("reports/generated/sliding_amount_strategy")
DEFAULT_INDEX_PATH = Path("data/raw/indexes/sse_composite_2021-01-01_2026-06-02.csv")


@dataclass(frozen=True)
class StrategyRule:
    rule_id: str
    description: str


@dataclass(frozen=True)
class CandidateRow:
    eval_date: str
    ts_code: str
    name: str
    amount_rank: int
    anchor_rank_100: int | None
    anchor_rank_200: int | None
    amount_ratio_20: float | None
    amount_ma20_ma60: float | None
    ret20_pct: float | None
    ret60_pct: float | None
    ma20_gap_pct: float | None
    ma60_gap_pct: float | None
    state: str
    forward_date: str
    forward_return_pct: float


@dataclass(frozen=True)
class RuleScore:
    rebalance_date: str
    rule_id: str
    train_count: int
    train_avg_return_pct: float | None
    validation_count: int
    validation_avg_return_pct: float | None
    validation_win_rate_pct: float | None
    selected: bool


@dataclass(frozen=True)
class PortfolioPeriod:
    rebalance_date: str
    exit_date: str
    selected_rule: str
    selected_count: int
    strategy_return_pct: float | None
    top100_equal_return_pct: float | None
    index_return_pct: float | None
    excess_vs_top100_pct: float | None
    excess_vs_index_pct: float | None
    selected_symbols: str


@dataclass(frozen=True)
class SlidingStrategyReport:
    trade_date: str
    status: str
    note: str
    train_days: int
    validation_days: int
    hold_days: int
    rebalance_step: int
    periods: int
    cumulative_strategy_return_pct: float | None
    cumulative_top100_return_pct: float | None
    cumulative_index_return_pct: float | None
    excess_vs_top100_pct: float | None
    excess_vs_index_pct: float | None
    selected_rule_counts: dict[str, int]
    outputs: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward amount-profile Top100 selection research.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--pool-file", default=str(DEFAULT_POOL_FILE))
    parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--trade-date", default=DEFAULT_TRADE_DATE)
    parser.add_argument("--simulation-start", default="2024-06-03")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--train-days", type=int, default=240)
    parser.add_argument("--validation-days", type=int, default=120)
    parser.add_argument("--hold-days", type=int, default=20)
    parser.add_argument("--rebalance-step", type=int, default=20)
    parser.add_argument("--min-picks", type=int, default=5)
    parser.add_argument("--max-picks", type=int, default=20)
    parser.add_argument("--min-validation-count", type=int, default=20)
    parser.add_argument("--lookbacks", default="100,200")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_sliding_amount_strategy(
        cache_dir=Path(args.cache_dir),
        index_path=Path(args.index_path),
        output_dir=Path(args.output_dir),
        trade_date=args.trade_date,
        simulation_start=args.simulation_start,
        top_n=args.top_n,
        train_days=args.train_days,
        validation_days=args.validation_days,
        hold_days=args.hold_days,
        rebalance_step=args.rebalance_step,
        min_picks=args.min_picks,
        max_picks=args.max_picks,
        min_validation_count=args.min_validation_count,
        lookbacks=_parse_ints(args.lookbacks),
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


def run_sliding_amount_strategy(
    cache_dir: Path,
    index_path: Path,
    output_dir: Path,
    trade_date: str,
    simulation_start: str,
    top_n: int,
    train_days: int,
    validation_days: int,
    hold_days: int,
    rebalance_step: int,
    min_picks: int,
    max_picks: int,
    min_validation_count: int,
    lookbacks: list[int],
) -> SlidingStrategyReport:
    trading_dates = load_reference_trading_dates(cache_dir, trade_date)
    rebalance_dates = build_rebalance_dates(
        trading_dates, simulation_start, trade_date, train_days, validation_days, hold_days, rebalance_step
    )
    target_dates = build_target_dates(trading_dates, rebalance_dates, train_days, validation_days, hold_days, lookbacks)
    points_by_date, points_by_stock, _ = load_profile_points(cache_dir, target_dates)
    ranks_by_date = build_amount_ranks(points_by_date)
    candidate_rows = build_candidate_rows(
        rebalance_dates=all_candidate_eval_dates(trading_dates, rebalance_dates, train_days, validation_days),
        top_n=top_n,
        hold_days=hold_days,
        lookbacks=lookbacks,
        trading_dates=trading_dates,
        points_by_stock=points_by_stock,
        ranks_by_date=ranks_by_date,
    )
    rows_by_date: dict[str, list[CandidateRow]] = defaultdict(list)
    for row in candidate_rows:
        rows_by_date[row.eval_date].append(row)
    rules = strategy_rules()
    scores: list[RuleScore] = []
    periods: list[PortfolioPeriod] = []
    index_data = load_index_data(index_path)
    for rebalance_date in rebalance_dates:
        train_dates, validation_dates = split_window_dates(
            trading_dates, rebalance_date, train_days, validation_days
        )
        selected_rule, rule_scores = choose_rule(
            rules,
            rows_by_date,
            train_dates,
            validation_dates,
            min_validation_count,
        )
        scores.extend(rule_scores)
        period = run_portfolio_period(
            rebalance_date=rebalance_date,
            hold_days=hold_days,
            rows_by_date=rows_by_date,
            selected_rule=selected_rule,
            rules=rules,
            min_picks=min_picks,
            max_picks=max_picks,
            trading_dates=trading_dates,
            index_data=index_data,
        )
        periods.append(period)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sliding_amount_strategy_{trade_date}"
    periods_path = output_dir / f"{stem}_periods.csv"
    scores_path = output_dir / f"{stem}_rule_scores.csv"
    report_path = output_dir / f"{stem}.json"
    write_dataclass_csv(periods_path, periods, PortfolioPeriod)
    write_dataclass_csv(scores_path, scores, RuleScore)
    strategy_total = compound_return([p.strategy_return_pct for p in periods])
    top100_total = compound_return([p.top100_equal_return_pct for p in periods])
    index_total = compound_return([p.index_return_pct for p in periods])
    report = SlidingStrategyReport(
        trade_date=trade_date,
        status="research_only",
        note=(
            "Walk-forward research: each rebalance uses only prior train/validation windows. "
            "No tradability replay, announcement filters, limit rules or execution costs are included."
        ),
        train_days=train_days,
        validation_days=validation_days,
        hold_days=hold_days,
        rebalance_step=rebalance_step,
        periods=len(periods),
        cumulative_strategy_return_pct=strategy_total,
        cumulative_top100_return_pct=top100_total,
        cumulative_index_return_pct=index_total,
        excess_vs_top100_pct=diff_or_none(strategy_total, top100_total),
        excess_vs_index_pct=diff_or_none(strategy_total, index_total),
        selected_rule_counts=dict(count_items(p.selected_rule for p in periods)),
        outputs={
            "periods": str(periods_path),
            "rule_scores": str(scores_path),
            "report": str(report_path),
        },
    )
    report_path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_rebalance_dates(
    trading_dates: list[str],
    simulation_start: str,
    trade_date: str,
    train_days: int,
    validation_days: int,
    hold_days: int,
    step: int,
) -> list[str]:
    start_index = max(train_days + validation_days, first_index_at_or_after(trading_dates, simulation_start))
    end_index = min(len(trading_dates) - hold_days - 1, last_index_at_or_before(trading_dates, trade_date))
    return [trading_dates[index] for index in range(start_index, end_index + 1, max(1, step))]


def all_candidate_eval_dates(
    trading_dates: list[str], rebalance_dates: list[str], train_days: int, validation_days: int
) -> list[str]:
    indexes: set[int] = set()
    for rebalance_date in rebalance_dates:
        index = trading_dates.index(rebalance_date)
        for item in range(index - train_days - validation_days, index + 1):
            if item >= 0:
                indexes.add(item)
    return [trading_dates[index] for index in sorted(indexes)]


def build_target_dates(
    trading_dates: list[str],
    rebalance_dates: list[str],
    train_days: int,
    validation_days: int,
    hold_days: int,
    lookbacks: list[int],
) -> set[str]:
    targets: set[str] = set()
    max_back = max([120, *lookbacks])
    for date in all_candidate_eval_dates(trading_dates, rebalance_dates, train_days, validation_days):
        index = trading_dates.index(date)
        for item in range(max(0, index - max_back), min(len(trading_dates), index + hold_days + 1)):
            targets.add(trading_dates[item])
    return targets


def build_candidate_rows(
    rebalance_dates: Iterable[str],
    top_n: int,
    hold_days: int,
    lookbacks: list[int],
    trading_dates: list[str],
    points_by_stock: dict[str, dict[str, AmountRankPoint]],
    ranks_by_date: dict[str, dict[str, int]],
) -> list[CandidateRow]:
    rows: list[CandidateRow] = []
    for eval_date in rebalance_dates:
        forward_date = offset_trade_date(trading_dates, eval_date, hold_days)
        if forward_date is None:
            continue
        top_items = sorted(ranks_by_date.get(eval_date, {}).items(), key=lambda item: item[1])[:top_n]
        for ts_code, rank in top_items:
            current_point = points_by_stock.get(ts_code, {}).get(eval_date)
            forward_point = points_by_stock.get(ts_code, {}).get(forward_date)
            forward_return = _return_pct(
                current_point.close if current_point else None,
                forward_point.close if forward_point else None,
            )
            if current_point is None or forward_return is None:
                continue
            profile = calculate_profile(ts_code, current_point.name, eval_date, trading_dates, points_by_stock, ranks_by_date)
            if profile is None:
                continue
            anchor_ranks = {
                lookback: ranks_by_date.get(offset_trade_date(trading_dates, eval_date, -lookback) or "", {}).get(ts_code)
                for lookback in lookbacks
            }
            rows.append(
                CandidateRow(
                    eval_date=eval_date,
                    ts_code=ts_code,
                    name=current_point.name,
                    amount_rank=rank,
                    anchor_rank_100=anchor_ranks.get(100),
                    anchor_rank_200=anchor_ranks.get(200),
                    amount_ratio_20=profile.amount_ratio_20,
                    amount_ma20_ma60=profile.amount_ma20_ma60,
                    ret20_pct=profile.ret20_pct,
                    ret60_pct=profile.ret60_pct,
                    ma20_gap_pct=profile.ma20_gap_pct,
                    ma60_gap_pct=profile.ma60_gap_pct,
                    state=profile.state,
                    forward_date=forward_date,
                    forward_return_pct=forward_return,
                )
            )
    return rows


def strategy_rules() -> list[StrategyRule]:
    return [
        StrategyRule("rank_persistent_not_hot", "anchor rank <=300, ret60 <50, MA60 gap <30"),
        StrategyRule("rank100_persistent_not_hot", "anchor100 <=300, ret60 <50, MA60 gap <30"),
        StrategyRule("rank200_persistent_not_hot", "anchor200 <=300, ret60 <50, MA60 gap <30"),
        StrategyRule("quiet_or_pullback", "quiet trend or pullback with liquidity, not overheated"),
        StrategyRule("moderate_volume_confirm", "amount ratio 1-4, MA20 gap 0-30, ret60 0-50"),
        StrategyRule("anti_overheat", "exclude ret60>=50, MA20 gap>=30, MA60 gap>=30, amount ratio>=4"),
        StrategyRule("top_rank_quality", "current amount rank <=50 and anti-overheat"),
    ]


def rule_filter(rule_id: str, row: CandidateRow) -> bool:
    anti_hot = not (
        _gte(row.ret60_pct, 50) or _gte(row.ma20_gap_pct, 30) or _gte(row.ma60_gap_pct, 30) or _gte(row.amount_ratio_20, 4)
    )
    persistent_100 = row.anchor_rank_100 is not None and row.anchor_rank_100 <= 300
    persistent_200 = row.anchor_rank_200 is not None and row.anchor_rank_200 <= 300
    if rule_id == "rank_persistent_not_hot":
        return anti_hot and (persistent_100 or persistent_200)
    if rule_id == "rank100_persistent_not_hot":
        return anti_hot and persistent_100
    if rule_id == "rank200_persistent_not_hot":
        return anti_hot and persistent_200
    if rule_id == "quiet_or_pullback":
        return anti_hot and row.state in {"quiet_trend", "pullback_with_liquidity"}
    if rule_id == "moderate_volume_confirm":
        return (
            row.amount_ratio_20 is not None
            and 1 <= row.amount_ratio_20 < 4
            and row.ma20_gap_pct is not None
            and 0 <= row.ma20_gap_pct < 30
            and row.ret60_pct is not None
            and 0 <= row.ret60_pct < 50
        )
    if rule_id == "anti_overheat":
        return anti_hot
    if rule_id == "top_rank_quality":
        return anti_hot and row.amount_rank <= 50
    raise ValueError(f"unknown rule: {rule_id}")


def choose_rule(
    rules: list[StrategyRule],
    rows_by_date: dict[str, list[CandidateRow]],
    train_dates: list[str],
    validation_dates: list[str],
    min_validation_count: int,
) -> tuple[StrategyRule, list[RuleScore]]:
    scores: list[RuleScore] = []
    scored_rules: list[tuple[float, float, int, StrategyRule]] = []
    for rule in rules:
        train_returns = selected_returns(rule, rows_by_date, train_dates)
        validation_returns = selected_returns(rule, rows_by_date, validation_dates)
        validation_avg = average(validation_returns)
        validation_win = win_rate(validation_returns)
        train_avg = average(train_returns)
        enough = len(validation_returns) >= min_validation_count
        score = validation_avg if validation_avg is not None and enough else -9999.0
        scores.append(
            RuleScore(
                rebalance_date=validation_dates[-1] if validation_dates else "",
                rule_id=rule.rule_id,
                train_count=len(train_returns),
                train_avg_return_pct=train_avg,
                validation_count=len(validation_returns),
                validation_avg_return_pct=validation_avg,
                validation_win_rate_pct=validation_win,
                selected=False,
            )
        )
        scored_rules.append((score, validation_win or -1.0, len(validation_returns), rule))
    selected = max(scored_rules, key=lambda item: item[:3])[3]
    scores = [score if score.rule_id != selected.rule_id else score.__class__(**(asdict(score) | {"selected": True})) for score in scores]
    return selected, scores


def selected_returns(rule: StrategyRule, rows_by_date: dict[str, list[CandidateRow]], dates: Iterable[str]) -> list[float]:
    returns: list[float] = []
    for date in dates:
        selected = [row for row in rows_by_date.get(date, []) if rule_filter(rule.rule_id, row)]
        if selected:
            returns.append(sum(row.forward_return_pct for row in selected) / len(selected))
    return returns


def run_portfolio_period(
    rebalance_date: str,
    hold_days: int,
    rows_by_date: dict[str, list[CandidateRow]],
    selected_rule: StrategyRule,
    rules: list[StrategyRule],
    min_picks: int,
    max_picks: int,
    trading_dates: list[str],
    index_data: pd.DataFrame,
) -> PortfolioPeriod:
    candidates = rows_by_date.get(rebalance_date, [])
    selected = [row for row in candidates if rule_filter(selected_rule.rule_id, row)]
    fallback_rule = next(rule for rule in rules if rule.rule_id == "anti_overheat")
    if len(selected) < min_picks:
        selected = [row for row in candidates if rule_filter(fallback_rule.rule_id, row)]
        selected_rule = fallback_rule
    selected = sorted(selected, key=selection_sort_key)[:max_picks]
    exit_date = offset_trade_date(trading_dates, rebalance_date, hold_days) or ""
    strategy_return = average([row.forward_return_pct for row in selected])
    top100_return = average([row.forward_return_pct for row in candidates])
    index_return = index_hold_return(index_data, rebalance_date, exit_date)
    return PortfolioPeriod(
        rebalance_date=rebalance_date,
        exit_date=exit_date,
        selected_rule=selected_rule.rule_id,
        selected_count=len(selected),
        strategy_return_pct=strategy_return,
        top100_equal_return_pct=top100_return,
        index_return_pct=index_return,
        excess_vs_top100_pct=diff_or_none(strategy_return, top100_return),
        excess_vs_index_pct=diff_or_none(strategy_return, index_return),
        selected_symbols=";".join(row.ts_code for row in selected),
    )


def split_window_dates(
    trading_dates: list[str], rebalance_date: str, train_days: int, validation_days: int
) -> tuple[list[str], list[str]]:
    index = trading_dates.index(rebalance_date)
    train_start = max(0, index - train_days - validation_days)
    validation_start = max(0, index - validation_days)
    return trading_dates[train_start:validation_start], trading_dates[validation_start:index]


def selection_sort_key(row: CandidateRow) -> tuple[int, int, float]:
    anchor = min([rank for rank in [row.anchor_rank_100, row.anchor_rank_200] if rank is not None] or [999999])
    hot_penalty = 1 if _gte(row.ret60_pct, 50) or _gte(row.ma60_gap_pct, 30) else 0
    return (hot_penalty, anchor, row.amount_rank)


def load_index_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame.set_index("trade_date").sort_index()


def index_hold_return(index_data: pd.DataFrame, entry_date: str, exit_date: str) -> float | None:
    if not entry_date or not exit_date:
        return None
    entry_rows = index_data.loc[index_data.index >= pd.Timestamp(entry_date)]
    exit_rows = index_data.loc[index_data.index >= pd.Timestamp(exit_date)]
    if entry_rows.empty or exit_rows.empty:
        return None
    entry = float(entry_rows.iloc[0]["close"])
    exit_ = float(exit_rows.iloc[0]["close"])
    return round((exit_ / entry - 1) * 100, 4)


def write_dataclass_csv(path: Path, rows: list[Any], row_type: type[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row_type.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def average(values: Iterable[float | None]) -> float | None:
    items = [value for value in values if value is not None]
    if not items:
        return None
    return round(sum(items) / len(items), 4)


def win_rate(values: Iterable[float | None]) -> float | None:
    items = [value for value in values if value is not None]
    if not items:
        return None
    return round(sum(1 for value in items if value > 0) / len(items) * 100, 2)


def compound_return(values: Iterable[float | None]) -> float | None:
    total = 1.0
    count = 0
    for value in values:
        if value is None:
            continue
        total *= 1 + value / 100
        count += 1
    if count == 0:
        return None
    return round((total - 1) * 100, 4)


def diff_or_none(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(left - right, 4)


def count_items(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def first_index_at_or_after(dates: list[str], target: str) -> int:
    for index, date in enumerate(dates):
        if date >= target:
            return index
    return len(dates) - 1


def last_index_at_or_before(dates: list[str], target: str) -> int:
    for index in range(len(dates) - 1, -1, -1):
        if dates[index] <= target:
            return index
    return 0


def _gte(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


if __name__ == "__main__":
    main()
