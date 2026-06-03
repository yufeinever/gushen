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
    build_ipo_window_low_hold_benchmark,
    load_ohlcv,
)
from gushen.trade_calendar import latest_research_trade_date

DEFAULT_STOCK_POOL_DOC = Path("docs/GUIDED_FACTOR_BACKTEST_STOCKS.md")
DEFAULT_OUTPUT_ROOT = Path("reports/generated/guided_factor_backtests")
DEFAULT_CACHE_DIR = Path("data/local/guided_factor_backtests")
DEFAULT_INDEX_PATH = Path("data/raw/indexes/sse_composite_2021-01-01_2026-06-02.csv")
FACTOR_WINDOWS = (5, 10, 20, 60)
FORWARD_HORIZON = 10


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
    trades: list[GuidedTrade]
    strategy_return_pct: float | None
    win_rate_pct: float | None
    max_drawdown_pct: float | None
    ipo_window_low_hold_return_pct: float | None
    index_hold_return_pct: float | None
    excess_vs_index_pct: float | None
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


def build_factor_frame(frame: pd.DataFrame, horizon: int = FORWARD_HORIZON) -> pd.DataFrame:
    data = frame.copy()
    close = data["close"]
    amount = data["amount"]
    volume = data["volume"]
    for window in FACTOR_WINDOWS:
        data[f"return_{window}"] = close.pct_change(window)
        data[f"ma_gap_{window}"] = close / close.rolling(window).mean() - 1
        data[f"amount_ratio_{window}"] = amount / amount.rolling(window).mean() - 1
        data[f"volume_ratio_{window}"] = volume / volume.rolling(window).mean() - 1
        data[f"volatility_{window}"] = close.pct_change().rolling(window).std()
        data[f"low_position_{window}"] = close / close.rolling(window).min() - 1
        data[f"high_position_{window}"] = close / close.rolling(window).max() - 1
    data["intraday_range"] = (data["high"] - data["low"]) / data["close"]
    data["close_position"] = (data["close"] - data["low"]) / (data["high"] - data["low"]).replace(0, math.nan)
    data["turnover"] = data["turnover"].fillna(0)
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
    train_end_index: int,
    hold_days: int = FORWARD_HORIZON,
    commission: float = 0.0008,
    min_confirmations: int = 2,
) -> list[GuidedTrade]:
    if not selected:
        return []
    thresholds = build_signal_thresholds(factors.iloc[:train_end_index], selected)
    trades: list[GuidedTrade] = []
    index = max(train_end_index, 1)
    while index < len(factors) - hold_days - 1:
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
        exit_index = min(entry_index + hold_days, len(factors) - 1)
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


def build_signal_thresholds(train: pd.DataFrame, selected: list[FactorScore]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for score in selected:
        series = train[score.factor].replace([math.inf, -math.inf], math.nan).dropna()
        if series.empty:
            continue
        thresholds[score.factor] = float(series.quantile(0.75 if score.direction > 0 else 0.25))
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
    trades: list[GuidedTrade] = []
    strategy_return = win_rate = max_drawdown = None
    ipo_low_return = index_return = excess = None
    status = "completed"
    note = "research-only per-stock factor backtest"
    if sufficiency.status != "pass":
        status = "insufficient_data"
        note = sufficiency.note
    else:
        factor_frame = build_factor_frame(frame)
        train_end_index = max(int(len(factor_frame) * 0.65), 260)
        selected = score_factors(factor_frame, train_end_index)
        trades = run_factor_guided_backtest(factor_frame, selected, train_end_index)
        strategy_return, win_rate, max_drawdown = summarize_returns([trade.net_return for trade in trades])
        ohlcv = build_ohlcv_for_benchmark(frame)
        ipo_low = build_ipo_window_low_hold_benchmark(ohlcv)
        ipo_low_return = ipo_low.get("return_pct")
        index_benchmark = None
        if index_data is not None:
            index_benchmark = build_aligned_index_hold_benchmark(
                index_data,
                entry_date=ipo_low.get("entry_date"),
                exit_date=ipo_low.get("exit_date"),
                name="SSE Composite",
            )
            index_return = index_benchmark.get("return_pct")
        if ipo_low_return is not None and index_return is not None:
            excess = ipo_low_return - index_return
        if not selected or not trades:
            status = "no_factor_trade"
            note = "factor screening completed, but no test-window trades fired"
        elif ipo_low_return is None or index_return is None:
            status = "baseline_incomplete"
            note = "factor backtest completed, but IPO-window-low/index baseline is missing"
        write_stock_artifacts(output_dir, factor_frame, selected, trades, ipo_low, index_benchmark)
    result = GuidedBacktestResult(
        code=stock.code,
        ts_code=stock.ts_code,
        name=stock.name,
        group=stock.group,
        rank=stock.rank,
        status=status,
        data_sufficiency=sufficiency,
        selected_factors=selected,
        trades=trades,
        strategy_return_pct=None if strategy_return is None else round(strategy_return, 4),
        win_rate_pct=None if win_rate is None else round(win_rate, 4),
        max_drawdown_pct=None if max_drawdown is None else round(max_drawdown, 4),
        ipo_window_low_hold_return_pct=None if ipo_low_return is None else round(float(ipo_low_return), 4),
        index_hold_return_pct=None if index_return is None else round(float(index_return), 4),
        excess_vs_index_pct=None if excess is None else round(float(excess), 4),
        output_dir=str(output_dir),
        note=note,
    )
    (output_dir / "guided_summary.json").write_text(json.dumps(to_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def write_stock_artifacts(
    output_dir: Path,
    factors: pd.DataFrame,
    selected: list[FactorScore],
    trades: list[GuidedTrade],
    ipo_low: dict[str, Any],
    index_benchmark: dict[str, Any] | None,
) -> None:
    factors.to_csv(output_dir / "factor_frame.csv", index=False)
    write_dataclass_csv(output_dir / "selected_factors.csv", selected)
    write_dataclass_csv(output_dir / "trades.csv", trades)
    (output_dir / "baselines.json").write_text(
        json.dumps({"ipo_window_low_hold": ipo_low, "aligned_index_hold": index_benchmark}, ensure_ascii=False, indent=2),
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
) -> list[GuidedBacktestResult]:
    end_date = end_date or latest_research_trade_date()
    start_date = (date.fromisoformat(end_date) - timedelta(days=365 * years + 30)).isoformat()
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
    args = parser.parse_args(argv)
    results = run_guided_factor_backtests(
        limit=args.limit,
        group=args.group,
        end_date=args.end_date,
        years=args.years,
        output_root=Path(args.output_root),
        index_path=Path(args.index_path),
    )
    print(json.dumps([to_jsonable(result) for result in results], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
