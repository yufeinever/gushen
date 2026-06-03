from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from gushen.data import DailyBar, fetch_daily_bars
from gushen.mature_backtest import (
    build_aligned_index_hold_benchmark,
    build_anchor_window_low_hold_benchmark,
    load_ohlcv,
)
from gushen.trade_calendar import latest_research_trade_date

DEFAULT_STOCK_POOL_DOC = Path("docs/GUIDED_FACTOR_BACKTEST_STOCKS.md")
DEFAULT_OUTPUT_ROOT = Path("reports/generated/guided_factor_backtests")
DEFAULT_CACHE_DIR = Path("data/local/guided_factor_backtests")
DEFAULT_INDEX_PATH = Path("data/raw/indexes/sse_composite_2021-01-01_2026-06-02.csv")
DEFAULT_POOL_LIMIT = 100
FACTOR_WINDOWS = (5, 10, 20, 60)
FORWARD_HORIZON = 10
SEARCH_HOLD_DAYS = (3, 5, 10, 20)
SEARCH_QUANTILES = (0.6, 0.7, 0.8)


@dataclass(frozen=True)
class GuidedStock:
    group: str
    rank: int
    name: str
    code: str
    ts_code: str


@dataclass(frozen=True)
class DataSufficiency:
    status: str
    bars: int
    first_date: str | None
    last_date: str | None
    missing_cells: int
    bad_ohlc_rows: int
    note: str


@dataclass(frozen=True)
class FactorScore:
    factor: str
    direction: int
    n: int
    correlation: float
    high_bucket_return: float
    low_bucket_return: float
    spread_return: float
    score: float


@dataclass(frozen=True)
class StrategyCandidate:
    strategy_id: str
    factors: str
    quantile: float
    min_confirmations: int
    hold_days: int
    train_return_pct: float | None
    validation_return_pct: float | None
    validation_win_rate_pct: float | None
    validation_max_drawdown_pct: float | None
    validation_trade_count: int
    score: float


@dataclass(frozen=True)
class GuidedTrade:
    signal_date: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    net_return: float
    hold_days: int
    signal_score: int
    signal_factors: str


@dataclass(frozen=True)
class GuidedBacktestResult:
    code: str
    ts_code: str
    name: str
    group: str
    rank: int
    status: str
    data_sufficiency: DataSufficiency
    selected_factors: list[FactorScore]
    best_strategy: StrategyCandidate | None
    trades: list[GuidedTrade]
    strategy_return_pct: float | None
    win_rate_pct: float | None
    max_drawdown_pct: float | None
    anchor_window_low_hold_return_pct: float | None
    index_hold_return_pct: float | None
    excess_vs_index_pct: float | None
    anchor_window_low_excess_vs_index_pct: float | None
    output_dir: str
    note: str


def parse_guided_stock_pool(path: Path = DEFAULT_STOCK_POOL_DOC) -> list[GuidedStock]:
    current_group = ""
    stocks: list[GuidedStock] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## Image"):
            current_group = line.removeprefix("## ").strip()
            continue
        if not line.startswith("|") or "---" in line or "Rank" in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 3 or not parts[0].isdigit():
            continue
        rank = int(parts[0])
        name = parts[1]
        code = parts[2]
        stocks.append(GuidedStock(current_group, rank, name, code, to_ts_code(code)))
    return stocks


def parse_external_stock_pool(
    path: Path,
    limit: int = DEFAULT_POOL_LIMIT,
    group: str = "external top amount",
) -> list[GuidedStock]:
    frame = _read_pool_frame(path)
    columns = {str(column).strip(): column for column in frame.columns}
    code_col = _pick_required_column(columns, ["code", "代码", "证券代码"])
    name_col = _pick_required_column(columns, ["name", "名称", "证券名称"])
    rank_col = _pick_optional_column(columns, ["rank", "amount_rank", "序", "排名"])
    if rank_col is not None:
        frame = frame.sort_values(rank_col, kind="stable")
    frame = frame.head(limit).copy()
    stocks: list[GuidedStock] = []
    seen: set[str] = set()
    for index, row in enumerate(frame.to_dict("records"), start=1):
        raw_code = str(row.get(code_col) or "").strip()
        code = normalize_stock_code(raw_code)
        if not code or code in seen:
            continue
        seen.add(code)
        rank_value = row.get(rank_col) if rank_col is not None else index
        try:
            rank = int(float(rank_value))
        except (TypeError, ValueError):
            rank = index
        stocks.append(
            GuidedStock(
                group=group,
                rank=rank,
                name=str(row.get(name_col) or "").strip(),
                code=code,
                ts_code=to_ts_code(code),
            )
        )
    return stocks


def _read_pool_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported pool file type: {path.suffix}")


def _pick_required_column(columns: dict[str, Any], names: list[str]) -> Any:
    column = _pick_optional_column(columns, names)
    if column is None:
        raise ValueError(f"missing required stock pool column, expected one of: {', '.join(names)}")
    return column


def _pick_optional_column(columns: dict[str, Any], names: list[str]) -> Any | None:
    lowered = {key.lower(): value for key, value in columns.items()}
    for name in names:
        if name in columns:
            return columns[name]
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def normalize_stock_code(code: str) -> str:
    raw = code.strip().upper()
    if "." in raw:
        raw = raw.split(".")[0]
    if raw.startswith(("SH", "SZ", "BJ")):
        raw = raw[2:]
    digits = "".join(character for character in raw if character.isdigit())
    return digits.zfill(6) if digits else ""


def to_ts_code(code: str) -> str:
    raw = code.split(".")[0]
    if raw.startswith(("6", "9")):
        return f"{raw}.SH"
    if raw.startswith(("0", "3")):
        return f"{raw}.SZ"
    if raw.startswith(("4", "8")):
        return f"{raw}.BJ"
    return code


def load_or_fetch_history(
    stock: GuidedStock,
    start_date: str,
    end_date: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    adjust: str = "qfq",
) -> list[DailyBar]:
    cache_path = cache_dir / "daily_bars" / adjust / f"{stock.ts_code}_{start_date}_{end_date}.csv"
    if cache_path.exists():
        rows = read_daily_bars(cache_path)
        if rows:
            return rows
        cache_path.unlink(missing_ok=True)
    rows = fetch_daily_bars(stock.ts_code, stock.name, start_date, end_date, adjust=adjust)
    rows = sorted(rows, key=lambda item: item.trade_date)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_daily_bars(cache_path, rows)
    return rows


def read_daily_bars(path: Path) -> list[DailyBar]:
    with path.open(newline="", encoding="utf-8") as file:
        return [DailyBar(**coerce_daily_bar(row)) for row in csv.DictReader(file)]


def write_daily_bars(path: Path, rows: list[DailyBar]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        if rows:
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))


def coerce_daily_bar(row: dict[str, str]) -> dict[str, Any]:
    values: dict[str, Any] = {"trade_date": row["trade_date"], "code": row["code"], "name": row["name"]}
    for key in ["open", "close", "high", "low", "volume", "amount", "amplitude", "pct_change", "turnover"]:
        values[key] = float(row[key])
    return values


def bars_to_frame(rows: list[DailyBar]) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(row) for row in rows])
    if frame.empty:
        return frame
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame.sort_values("trade_date").reset_index(drop=True)


def assess_sufficiency(frame: pd.DataFrame, min_bars: int = 420) -> DataSufficiency:
    if frame.empty:
        return DataSufficiency("fail", 0, None, None, 0, 0, "no daily bars")
    numeric = ["open", "high", "low", "close", "volume", "amount"]
    missing_cells = int(frame[numeric].isna().sum().sum())
    bad_ohlc = int(
        (
            (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
            | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
            | (frame["close"] <= 0)
            | (frame["open"] <= 0)
        ).sum()
    )
    status = "pass" if len(frame) >= min_bars and missing_cells == 0 and bad_ohlc == 0 else "fail"
    note = "sufficient for research backtest" if status == "pass" else "insufficient daily bar quality"
    return DataSufficiency(
        status=status,
        bars=int(len(frame)),
        first_date=frame.iloc[0]["trade_date"].date().isoformat(),
        last_date=frame.iloc[-1]["trade_date"].date().isoformat(),
        missing_cells=missing_cells,
        bad_ohlc_rows=bad_ohlc,
        note=note,
    )


def slice_recent_window(frame: pd.DataFrame, end_date: str, years: int = 2) -> pd.DataFrame:
    anchor = pd.Timestamp(end_date) - pd.DateOffset(years=years)
    recent = frame[frame["trade_date"] >= anchor].copy()
    return recent.reset_index(drop=True)


def build_factor_frame(frame: pd.DataFrame, horizon: int = FORWARD_HORIZON) -> pd.DataFrame:
    data = frame.copy()
    close = data["close"]
    amount = data["amount"]
    volume = data["volume"]
    day_return = close.pct_change()
    data["turnover"] = data["turnover"].fillna(0)
    data["overnight_gap"] = data["open"] / close.shift(1) - 1
    data["intraday_return"] = data["close"] / data["open"] - 1
    data["intraday_range"] = (data["high"] - data["low"]) / data["close"]
    data["close_position"] = (data["close"] - data["low"]) / (data["high"] - data["low"]).replace(0, math.nan)
    data["upper_shadow"] = (data["high"] - data[["open", "close"]].max(axis=1)) / data["close"]
    data["lower_shadow"] = (data[["open", "close"]].min(axis=1) - data["low"]) / data["close"]
    data["real_body"] = (data["close"] - data["open"]).abs() / data["close"]
    for window in FACTOR_WINDOWS:
        data[f"return_{window}"] = close.pct_change(window)
        data[f"ma_gap_{window}"] = close / close.rolling(window).mean() - 1
        data[f"amount_ratio_{window}"] = amount / amount.rolling(window).mean() - 1
        data[f"volume_ratio_{window}"] = volume / volume.rolling(window).mean() - 1
        data[f"turnover_ratio_{window}"] = data["turnover"] / data["turnover"].rolling(window).mean() - 1
        data[f"volatility_{window}"] = day_return.rolling(window).std()
        data[f"low_position_{window}"] = close / close.rolling(window).min() - 1
        data[f"high_position_{window}"] = close / close.rolling(window).max() - 1
        data[f"positive_day_ratio_{window}"] = (day_return > 0).rolling(window).mean()
        data[f"close_position_mean_{window}"] = data["close_position"].rolling(window).mean()
        data[f"amount_zscore_{window}"] = (amount - amount.rolling(window).mean()) / amount.rolling(window).std()
    data["return_acceleration_5_20"] = data["return_5"] - data["return_20"]
    data["trend_strength_20_60"] = data["ma_gap_20"] - data["ma_gap_60"]
    data["volume_price_confirm_20"] = data["return_20"] * data["volume_ratio_20"]
    data["amount_price_confirm_20"] = data["return_20"] * data["amount_ratio_20"]
    data["volatility_compression_20_60"] = data["volatility_20"] / data["volatility_60"] - 1
    data["turnover_price_confirm_20"] = data["return_20"] * data["turnover_ratio_20"]
    data["forward_return"] = close.shift(-horizon) / close.shift(-1) - 1
    return data


def score_factors(factors: pd.DataFrame, train_end_index: int, max_factors: int = 5) -> list[FactorScore]:
    excluded = {
        "forward_return",
        "trade_date",
        "code",
        "name",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    }
    candidates = [column for column in factors.columns if column not in excluded and factors[column].dtype.kind in "if"]
    train = factors.iloc[:train_end_index].copy()
    scores: list[FactorScore] = []
    for factor in candidates:
        sample = train[[factor, "forward_return"]].replace([math.inf, -math.inf], math.nan).dropna()
        if len(sample) < 120 or sample[factor].nunique() < 5:
            continue
        corr = float(sample[factor].rank().corr(sample["forward_return"].rank()))
        if math.isnan(corr):
            continue
        quantiles = pd.qcut(sample[factor], 5, labels=False, duplicates="drop")
        if quantiles.nunique() < 3:
            continue
        bucket_returns = sample.groupby(quantiles)["forward_return"].mean()
        low_return = float(bucket_returns.iloc[0])
        high_return = float(bucket_returns.iloc[-1])
        direction = 1 if corr >= 0 else -1
        spread = (high_return - low_return) * direction
        score = abs(corr) + max(spread, 0) * 2
        scores.append(
            FactorScore(
                factor=factor,
                direction=direction,
                n=int(len(sample)),
                correlation=round(corr, 6),
                high_bucket_return=round(high_return, 6),
                low_bucket_return=round(low_return, 6),
                spread_return=round(spread, 6),
                score=round(score, 6),
            )
        )
    return sorted(scores, key=lambda item: item.score, reverse=True)[:max_factors]


def run_factor_guided_backtest(
    factors: pd.DataFrame,
    selected: list[FactorScore],
    train_end_index: int = 0,
    hold_days: int = FORWARD_HORIZON,
    commission: float = 0.0008,
    min_confirmations: int = 2,
    quantile: float = 0.75,
    start_index: int | None = None,
    end_index: int | None = None,
) -> list[GuidedTrade]:
    if not selected:
        return []
    thresholds = build_signal_thresholds(factors.iloc[:train_end_index], selected, quantile)
    trades: list[GuidedTrade] = []
    index = max(start_index if start_index is not None else train_end_index, 1)
    last_index = min(end_index if end_index is not None else len(factors) - 1, len(factors) - 1)
    while index < last_index - hold_days:
        row = factors.iloc[index]
        passed: list[str] = []
        for score in selected:
            value = row[score.factor]
            threshold = thresholds.get(score.factor)
            if pd.isna(value) or threshold is None:
                continue
            if (score.direction > 0 and value >= threshold) or (score.direction < 0 and value <= threshold):
                passed.append(score.factor)
        if len(passed) < min_confirmations:
            index += 1
            continue
        entry_index = index + 1
        exit_index = min(entry_index + hold_days, last_index)
        entry_price = float(factors.iloc[entry_index]["open"])
        exit_price = float(factors.iloc[exit_index]["close"])
        net_return = exit_price * (1 - commission) / (entry_price * (1 + commission)) - 1
        trades.append(
            GuidedTrade(
                signal_date=factors.iloc[index]["trade_date"].date().isoformat(),
                entry_date=factors.iloc[entry_index]["trade_date"].date().isoformat(),
                exit_date=factors.iloc[exit_index]["trade_date"].date().isoformat(),
                entry_price=round(entry_price, 4),
                exit_price=round(exit_price, 4),
                net_return=round(net_return, 6),
                hold_days=exit_index - entry_index + 1,
                signal_score=len(passed),
                signal_factors="|".join(passed),
            )
        )
        index = exit_index + 1
    return trades


def build_signal_thresholds(
    train: pd.DataFrame, selected: list[FactorScore], quantile: float = 0.75
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for score in selected:
        series = train[score.factor].replace([math.inf, -math.inf], math.nan).dropna()
        if series.empty:
            continue
        threshold_quantile = quantile if score.direction > 0 else 1 - quantile
        thresholds[score.factor] = float(series.quantile(threshold_quantile))
    return thresholds


def summarize_returns(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    equity = 1.0
    curve = []
    for value in values:
        equity *= 1 + value
        curve.append(equity)
    peak = curve[0]
    max_drawdown = 0.0
    for value in curve:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, value / peak - 1)
    win_rate = sum(1 for value in values if value > 0) / len(values) * 100
    return (equity - 1) * 100, win_rate, max_drawdown * 100


def calculate_excess_return(
    strategy_return: float | None,
    anchor_low_return: float | None,
    index_return: float | None,
) -> tuple[float | None, float | None]:
    strategy_excess = None
    anchor_low_excess = None
    if strategy_return is not None and index_return is not None:
        strategy_excess = strategy_return - index_return
    if anchor_low_return is not None and index_return is not None:
        anchor_low_excess = anchor_low_return - index_return
    return strategy_excess, anchor_low_excess


def search_strategy_library(
    factors: pd.DataFrame,
    selected: list[FactorScore],
    train_end_index: int,
    test_start_index: int | None = None,
) -> tuple[StrategyCandidate | None, list[StrategyCandidate], list[GuidedTrade]]:
    if not selected:
        return None, [], []
    if test_start_index is None:
        _, test_start_index = build_strategy_search_splits(len(factors))
    validation_start = train_end_index
    validation_end = test_start_index - 1
    test_start = test_start_index
    test_end = len(factors) - 1
    if validation_end - validation_start < 20 or test_end - test_start < 20:
        return None, [], []
    library: list[StrategyCandidate] = []
    selected_sets = build_factor_sets(selected[:5])
    calibration_start = max(60, min(120, train_end_index // 2))
    for factor_set in selected_sets:
        for hold_days in SEARCH_HOLD_DAYS:
            for quantile in SEARCH_QUANTILES:
                for min_confirmations in sorted({1, min(2, len(factor_set)), len(factor_set)}):
                    if min_confirmations > len(factor_set):
                        continue
                    train_trades = run_factor_guided_backtest(
                        factors,
                        factor_set,
                        train_end_index=train_end_index,
                        hold_days=hold_days,
                        min_confirmations=min_confirmations,
                        quantile=quantile,
                        start_index=calibration_start,
                        end_index=train_end_index - 1,
                    )
                    validation_trades = run_factor_guided_backtest(
                        factors,
                        factor_set,
                        train_end_index=train_end_index,
                        hold_days=hold_days,
                        min_confirmations=min_confirmations,
                        quantile=quantile,
                        start_index=validation_start,
                        end_index=validation_end,
                    )
                    train_return, _, _ = summarize_returns([trade.net_return for trade in train_trades])
                    val_return, val_win, val_dd = summarize_returns([trade.net_return for trade in validation_trades])
                    score = strategy_candidate_score(
                        train_return,
                        val_return,
                        val_dd,
                        len(train_trades),
                        len(validation_trades),
                    )
                    library.append(
                        StrategyCandidate(
                            strategy_id=strategy_id(factor_set, quantile, min_confirmations, hold_days),
                            factors="|".join(item.factor for item in factor_set),
                            quantile=quantile,
                            min_confirmations=min_confirmations,
                            hold_days=hold_days,
                            train_return_pct=None if train_return is None else round(train_return, 4),
                            validation_return_pct=None if val_return is None else round(val_return, 4),
                            validation_win_rate_pct=None if val_win is None else round(val_win, 4),
                            validation_max_drawdown_pct=None if val_dd is None else round(val_dd, 4),
                            validation_trade_count=len(validation_trades),
                            score=round(score, 6),
                        )
                    )
    library = sorted(library, key=lambda item: item.score, reverse=True)
    best = library[0] if library else None
    best_trades: list[GuidedTrade] = []
    if best:
        best_factors = [item for item in selected if item.factor in best.factors.split("|")]
        best_trades = run_factor_guided_backtest(
            factors,
            best_factors,
            train_end_index=train_end_index,
            hold_days=best.hold_days,
            min_confirmations=best.min_confirmations,
            quantile=best.quantile,
            start_index=test_start,
            end_index=test_end,
        )
    return best, library[:25], best_trades


def build_factor_sets(selected: list[FactorScore]) -> list[list[FactorScore]]:
    sets = [[item] for item in selected]
    for index, first in enumerate(selected):
        for second in selected[index + 1 :]:
            sets.append([first, second])
    if len(selected) >= 3:
        sets.append(selected[:3])
    return sets


def build_strategy_search_splits(length: int) -> tuple[int, int]:
    if length < 240:
        train_end = max(120, int(length * 0.55))
        test_start = max(train_end + 30, int(length * 0.78))
    else:
        train_end = min(max(int(length * 0.55), 150), length - 100)
        test_start = min(max(int(length * 0.78), train_end + 50), length - 40)
    train_end = min(max(train_end, 120), max(length - 80, 120))
    test_start = min(max(test_start, train_end + 30), max(length - 20, train_end + 30))
    return train_end, test_start


def strategy_candidate_score(
    train_return: float | None,
    validation_return: float | None,
    validation_drawdown: float | None,
    train_count: int,
    validation_count: int,
) -> float:
    if validation_return is None or validation_count < 2:
        return -1_000_000.0
    train_penalty = 0.0 if train_return is not None and train_return > -20 and train_count >= 2 else 50.0
    drawdown_penalty = abs(validation_drawdown or 0.0) * 0.35
    return validation_return - drawdown_penalty - train_penalty + min(validation_count, 8) * 0.5


def strategy_id(factors: list[FactorScore], quantile: float, min_confirmations: int, hold_days: int) -> str:
    factor_part = "+".join(item.factor for item in factors)
    return f"{factor_part}:q{quantile:.1f}:c{min_confirmations}:h{hold_days}"


def build_ohlcv_for_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.rename(
        columns={"trade_date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    data = data.set_index("Date").astype(float)
    return data


def run_one_stock(
    stock: GuidedStock,
    start_date: str,
    end_date: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    index_data: pd.DataFrame | None = None,
) -> GuidedBacktestResult:
    output_dir = output_root / stock.ts_code
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_or_fetch_history(stock, start_date, end_date)
    frame = bars_to_frame(rows)
    sufficiency = assess_sufficiency(frame)
    selected: list[FactorScore] = []
    best_strategy: StrategyCandidate | None = None
    strategy_library: list[StrategyCandidate] = []
    trades: list[GuidedTrade] = []
    strategy_return = win_rate = max_drawdown = None
    anchor_low_return = index_return = strategy_excess = anchor_low_excess = None
    status = "completed"
    note = "research-only per-stock factor backtest"
    if sufficiency.status != "pass":
        status = "insufficient_data"
        note = sufficiency.note
    else:
        recent_frame = slice_recent_window(frame, end_date=end_date, years=2)
        recent_sufficiency = assess_sufficiency(recent_frame, min_bars=240)
        if recent_sufficiency.status != "pass":
            status = "insufficient_recent_data"
            note = "insufficient two-year window for factor screening"
            factor_frame = build_factor_frame(recent_frame) if not recent_frame.empty else pd.DataFrame()
            anchor_low = build_anchor_window_low_hold_benchmark(build_ohlcv_for_benchmark(frame), end_date=end_date)
            index_benchmark = None
            selected = []
            best_strategy = None
            strategy_library = []
            trades = []
        else:
            factor_frame = build_factor_frame(recent_frame)
            train_end_index, test_start_index = build_strategy_search_splits(len(factor_frame))
            selected = score_factors(factor_frame, train_end_index)
            best_strategy, strategy_library, trades = search_strategy_library(
                factor_frame, selected, train_end_index, test_start_index
            )
            strategy_return, win_rate, max_drawdown = summarize_returns([trade.net_return for trade in trades])
            anchor_low = build_anchor_window_low_hold_benchmark(build_ohlcv_for_benchmark(frame), end_date=end_date)
        anchor_low_return = anchor_low.get("return_pct")
        index_benchmark = None
        if index_data is not None:
            index_benchmark = build_aligned_index_hold_benchmark(
                index_data,
                entry_date=anchor_low.get("entry_date"),
                exit_date=anchor_low.get("exit_date"),
                name="SSE Composite",
            )
            index_return = index_benchmark.get("return_pct")
        strategy_excess, anchor_low_excess = calculate_excess_return(
            strategy_return, anchor_low_return, index_return
        )
        if status == "completed" and (not selected or not trades):
            status = "no_factor_trade"
            note = "factor screening completed, but no test-window trades fired"
        elif status == "completed" and (anchor_low_return is None or index_return is None):
            status = "baseline_incomplete"
            note = "factor backtest completed, but anchor-window-low/index baseline is missing"
        write_stock_artifacts(output_dir, factor_frame, selected, strategy_library, trades, anchor_low, index_benchmark)
    result = GuidedBacktestResult(
        code=stock.code,
        ts_code=stock.ts_code,
        name=stock.name,
        group=stock.group,
        rank=stock.rank,
        status=status,
        data_sufficiency=sufficiency,
        selected_factors=selected,
        best_strategy=best_strategy,
        trades=trades,
        strategy_return_pct=None if strategy_return is None else round(strategy_return, 4),
        win_rate_pct=None if win_rate is None else round(win_rate, 4),
        max_drawdown_pct=None if max_drawdown is None else round(max_drawdown, 4),
        anchor_window_low_hold_return_pct=None if anchor_low_return is None else round(float(anchor_low_return), 4),
        index_hold_return_pct=None if index_return is None else round(float(index_return), 4),
        excess_vs_index_pct=None if strategy_excess is None else round(float(strategy_excess), 4),
        anchor_window_low_excess_vs_index_pct=(
            None if anchor_low_excess is None else round(float(anchor_low_excess), 4)
        ),
        output_dir=str(output_dir),
        note=note,
    )
    (output_dir / "guided_summary.json").write_text(json.dumps(to_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def write_stock_artifacts(
    output_dir: Path,
    factors: pd.DataFrame,
    selected: list[FactorScore],
    strategy_library: list[StrategyCandidate],
    trades: list[GuidedTrade],
    anchor_low: dict[str, Any],
    index_benchmark: dict[str, Any] | None,
) -> None:
    factors.to_csv(output_dir / "factor_frame.csv", index=False)
    write_dataclass_csv(output_dir / "selected_factors.csv", selected)
    write_dataclass_csv(output_dir / "strategy_library.csv", strategy_library)
    write_dataclass_csv(output_dir / "trades.csv", trades)
    (output_dir / "baselines.json").write_text(
        json.dumps({"anchor_window_low_hold": anchor_low, "aligned_index_hold": index_benchmark}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_dataclass_csv(path: Path, rows: list[Any]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    return value


def write_batch_outputs(results: list[GuidedBacktestResult], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    write_dataclass_csv(output_root / "guided_batch_summary.csv", results)
    (output_root / "guided_batch_summary.json").write_text(
        json.dumps([to_jsonable(result) for result in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_index_data(path: Path, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame | None:
    if not path.exists() and start_date and end_date:
        fetch_sse_composite_index(path, start_date, end_date)
    if not path.exists():
        return None
    return load_ohlcv(path)


def fetch_sse_composite_index(path: Path, start_date: str, end_date: str) -> None:
    import akshare as ak

    frame = ak.stock_zh_index_daily_em(symbol="sh000001")
    frame = frame.rename(
        columns={
            "date": "trade_date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
    )
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame[(frame["trade_date"] >= start_date) & (frame["trade_date"] <= end_date)]
    frame = frame.sort_values("trade_date")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[["trade_date", "open", "high", "low", "close", "volume"]].to_csv(path, index=False)


def run_guided_factor_backtests(
    limit: int = 3,
    group: str = "Image 3 - top amount leaders",
    end_date: str | None = None,
    years: int = 5,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    index_path: Path = DEFAULT_INDEX_PATH,
    pool_path: Path | None = None,
) -> list[GuidedBacktestResult]:
    end_date = end_date or latest_research_trade_date()
    start_date = (date.fromisoformat(end_date) - timedelta(days=365 * years + 30)).isoformat()
    if pool_path is not None:
        stocks = parse_external_stock_pool(pool_path, limit=limit, group=group)
    else:
        stocks = [stock for stock in parse_guided_stock_pool() if stock.group == group]
        stocks = stocks[:limit]
    index_data = load_index_data(index_path, start_date, end_date)
    results = [run_one_stock(stock, start_date, end_date, output_root, index_data) for stock in stocks]
    write_batch_outputs(results, output_root)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run guided per-stock factor screening and backtests.")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--group", default="Image 3 - top amount leaders")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH))
    parser.add_argument("--pool-file", default=None, help="Optional CSV/XLSX stock pool with code/name/rank columns.")
    args = parser.parse_args(argv)
    results = run_guided_factor_backtests(
        limit=args.limit,
        group=args.group,
        end_date=args.end_date,
        years=args.years,
        output_root=Path(args.output_root),
        index_path=Path(args.index_path),
        pool_path=Path(args.pool_file) if args.pool_file else None,
    )
    print(json.dumps([to_jsonable(result) for result in results], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
