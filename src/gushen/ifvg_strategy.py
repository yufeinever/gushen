from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from gushen.guided_factor_backtest import build_factor_frame

DEFAULT_CACHE_DIR = Path("data/local/guided_factor_backtests/daily_bars/qfq")
DEFAULT_OUTPUT_DIR = Path("reports/generated/ifvg_backtests")
BACKTEST_CASH = 100_000.0
DEFAULT_COMMISSION = 0.0008


@dataclass(frozen=True)
class IfvgZone:
    zone_id: str
    direction: str
    fvg_date: str
    inverted_date: str
    lower: float
    upper: float
    invalidation: float


@dataclass(frozen=True)
class IfvgSignal:
    signal_date: str
    direction: str
    zone_id: str
    lower: float
    upper: float
    close: float
    bias: str
    confirmation: str


@dataclass(frozen=True)
class IfvgTrade:
    ts_code: str
    name: str
    direction: str
    signal_date: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    net_return: float
    r_multiple: float
    bars_held: int
    exit_reason: str
    zone_id: str


@dataclass(frozen=True)
class IfvgStockResult:
    ts_code: str
    name: str
    bars: int
    first_date: str | None
    last_date: str | None
    signals: int
    trades: int
    win_rate_pct: float | None
    avg_return_pct: float | None
    total_return_pct: float | None
    buy_hold_return_pct: float | None
    excess_vs_buy_hold_pct: float | None
    max_drawdown_pct: float | None
    profit_factor: float | None
    avg_r_multiple: float | None
    status: str
    note: str


@dataclass(frozen=True)
class IfvgBatchResult:
    output_dir: str
    files_scanned: int
    stocks_tested: int
    stocks_with_trades: int
    total_trades: int
    win_rate_pct: float | None
    avg_return_pct: float | None
    total_return_pct: float | None
    buy_hold_return_pct: float | None
    excess_vs_buy_hold_pct: float | None
    stocks_outperform_buy_hold: int
    trade_compound_return_pct: float | None
    profit_factor: float | None
    result_path: str
    trades_path: str


def load_daily_bar_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"trade_date", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"daily bar file missing columns: {missing}")
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    for column in ["open", "high", "low", "close", "volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if "code" not in data.columns:
        data["code"] = infer_ts_code_from_path(path)
    if "name" not in data.columns:
        data["name"] = ""
    return data.sort_values("trade_date").reset_index(drop=True)


def infer_ts_code_from_path(path: Path) -> str:
    stem = path.stem
    return stem.split("_")[0] if "_" in stem else stem


def detect_ifvg_signals(
    frame: pd.DataFrame,
    htf_window: int = 60,
    htf_slope_window: int = 5,
    fvg_lookback: int = 80,
    min_gap_pct: float = 0.002,
    confirm_window: int = 5,
    directions: tuple[str, ...] = ("bullish",),
) -> list[IfvgSignal]:
    data = prepare_frame(frame)
    if len(data) < max(htf_window + htf_slope_window + 5, 80):
        return []

    close = data["close"]
    htf_ma = close.rolling(htf_window).mean()
    htf_slope = htf_ma.diff(htf_slope_window)
    active_zones: list[IfvgZone] = []
    used_zones: set[str] = set()
    signals: list[IfvgSignal] = []

    for index in range(2, len(data)):
        row = data.iloc[index]
        prev2 = data.iloc[index - 2]
        date_value = iso_date(row["trade_date"])

        # Bullish FVG: candle 3 low stays above candle 1 high.
        bullish_lower = float(prev2["high"])
        bullish_upper = float(row["low"])
        if bullish_upper > bullish_lower:
            gap_pct = (bullish_upper - bullish_lower) / max(float(row["close"]), 1e-9)
            if gap_pct >= min_gap_pct:
                zone_id = f"{date_value}:bullish:{index}"
                active_zones.append(
                    IfvgZone(
                        zone_id=zone_id,
                        direction="bearish",
                        fvg_date=date_value,
                        inverted_date="",
                        lower=round(bullish_lower, 6),
                        upper=round(bullish_upper, 6),
                        invalidation=round(bullish_upper, 6),
                    )
                )

        # Bearish FVG: candle 3 high stays below candle 1 low.
        bearish_lower = float(row["high"])
        bearish_upper = float(prev2["low"])
        if bearish_upper > bearish_lower:
            gap_pct = (bearish_upper - bearish_lower) / max(float(row["close"]), 1e-9)
            if gap_pct >= min_gap_pct:
                zone_id = f"{date_value}:bearish:{index}"
                active_zones.append(
                    IfvgZone(
                        zone_id=zone_id,
                        direction="bullish",
                        fvg_date=date_value,
                        inverted_date="",
                        lower=round(bearish_lower, 6),
                        upper=round(bearish_upper, 6),
                        invalidation=round(bearish_lower, 6),
                    )
                )

        active_zones = [
            zone
            for zone in active_zones
            if index - int(zone.zone_id.rsplit(":", 1)[-1]) <= fvg_lookback
            and zone.zone_id not in used_zones
            and zone.direction in directions
        ]

        for zone in list(active_zones):
            if zone.direction not in directions:
                continue
            zone_index = int(zone.zone_id.rsplit(":", 1)[-1])
            if index <= zone_index + 1:
                continue
            if not zone.inverted_date:
                inverted = (
                    zone.direction == "bullish"
                    and float(row["close"]) > zone.upper
                    or zone.direction == "bearish"
                    and float(row["close"]) < zone.lower
                )
                if inverted:
                    active_zones.remove(zone)
                    active_zones.append(
                        IfvgZone(
                            zone_id=zone.zone_id,
                            direction=zone.direction,
                            fvg_date=zone.fvg_date,
                            inverted_date=date_value,
                            lower=zone.lower,
                            upper=zone.upper,
                            invalidation=zone.invalidation,
                        )
                    )
                continue

            if not htf_bias_matches(data, htf_ma, htf_slope, index, zone.direction):
                continue
            if not zone_retested_and_respected(row, zone):
                continue
            if not displacement_confirmed(data, index, zone.direction, confirm_window):
                continue
            signals.append(
                IfvgSignal(
                    signal_date=date_value,
                    direction=zone.direction,
                    zone_id=zone.zone_id,
                    lower=zone.lower,
                    upper=zone.upper,
                    close=round(float(row["close"]), 6),
                    bias=zone.direction,
                    confirmation=f"{confirm_window}bar_structure_break",
                )
            )
            used_zones.add(zone.zone_id)
            active_zones.remove(zone)
            break
    return signals


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    for column in ["open", "high", "low", "close", "volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=["open", "high", "low", "close"]).sort_values("trade_date").reset_index(drop=True)


def htf_bias_matches(
    data: pd.DataFrame,
    htf_ma: pd.Series,
    htf_slope: pd.Series,
    index: int,
    direction: str,
) -> bool:
    ma_value = htf_ma.iloc[index]
    slope_value = htf_slope.iloc[index]
    close_value = float(data.iloc[index]["close"])
    if pd.isna(ma_value) or pd.isna(slope_value):
        return False
    if direction == "bullish":
        return close_value >= float(ma_value) and float(slope_value) >= 0
    return close_value <= float(ma_value) and float(slope_value) <= 0


def zone_retested_and_respected(row: pd.Series, zone: IfvgZone) -> bool:
    low = float(row["low"])
    high = float(row["high"])
    close = float(row["close"])
    open_price = float(row["open"])
    if zone.direction == "bullish":
        touched = low <= zone.upper and high >= zone.lower
        respected = close >= zone.lower and close >= open_price
    else:
        touched = high >= zone.lower and low <= zone.upper
        respected = close <= zone.upper and close <= open_price
    return touched and respected


def displacement_confirmed(data: pd.DataFrame, index: int, direction: str, confirm_window: int) -> bool:
    if index - confirm_window < 1:
        return False
    previous = data.iloc[index - confirm_window : index]
    close = float(data.iloc[index]["close"])
    candle_range = float(data.iloc[index]["high"] - data.iloc[index]["low"])
    avg_range = float((previous["high"] - previous["low"]).mean())
    if avg_range <= 0 or candle_range < avg_range * 0.8:
        return False
    if direction == "bullish":
        return close > float(previous["high"].max())
    return close < float(previous["low"].min())


def run_ifvg_backtest(
    frame: pd.DataFrame,
    ts_code: str,
    name: str = "",
    risk_reward: float = 1.5,
    max_hold_bars: int = 10,
    commission: float = DEFAULT_COMMISSION,
    **signal_kwargs: Any,
) -> tuple[IfvgStockResult, list[IfvgTrade], list[IfvgSignal]]:
    data = prepare_frame(frame)
    signals = detect_ifvg_signals(data, **signal_kwargs)
    if data.empty:
        return (
            IfvgStockResult(
                ts_code,
                name,
                0,
                None,
                None,
                0,
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "no_data",
                "",
            ),
            [],
            signals,
        )
    index_by_date = {iso_date(row.trade_date): position for position, row in data.iterrows()}
    trades: list[IfvgTrade] = []
    next_allowed_index = 0
    for signal in signals:
        signal_index = index_by_date.get(signal.signal_date)
        if signal_index is None or signal_index < next_allowed_index:
            continue
        entry_index = signal_index + 1
        if entry_index >= len(data):
            continue
        entry = data.iloc[entry_index]
        entry_price = float(entry["open"])
        if signal.direction == "bullish":
            stop_loss = min(signal.lower, float(data.iloc[signal_index]["low"]))
            risk = entry_price - stop_loss
            take_profit = entry_price + risk * risk_reward
        else:
            stop_loss = max(signal.upper, float(data.iloc[signal_index]["high"]))
            risk = stop_loss - entry_price
            take_profit = entry_price - risk * risk_reward
        if risk <= 0:
            continue
        exit_index, exit_price, exit_reason = resolve_exit(
            data,
            entry_index,
            signal.direction,
            stop_loss,
            take_profit,
            max_hold_bars,
        )
        net_return = trade_return(entry_price, exit_price, signal.direction, commission)
        trades.append(
            IfvgTrade(
                ts_code=ts_code,
                name=name,
                direction=signal.direction,
                signal_date=signal.signal_date,
                entry_date=iso_date(entry["trade_date"]),
                exit_date=iso_date(data.iloc[exit_index]["trade_date"]),
                entry_price=round(entry_price, 4),
                exit_price=round(exit_price, 4),
                stop_loss=round(stop_loss, 4),
                take_profit=round(take_profit, 4),
                net_return=round(net_return, 6),
                r_multiple=round(net_return / (risk / entry_price), 4),
                bars_held=exit_index - entry_index + 1,
                exit_reason=exit_reason,
                zone_id=signal.zone_id,
            )
        )
        next_allowed_index = exit_index + 1
    result = summarize_stock(ts_code, name, data, signals, trades)
    return result, trades, signals


def resolve_exit(
    data: pd.DataFrame,
    entry_index: int,
    direction: str,
    stop_loss: float,
    take_profit: float,
    max_hold_bars: int,
) -> tuple[int, float, str]:
    last_index = min(entry_index + max_hold_bars - 1, len(data) - 1)
    for index in range(entry_index, last_index + 1):
        row = data.iloc[index]
        high = float(row["high"])
        low = float(row["low"])
        if direction == "bullish":
            stop_hit = low <= stop_loss
            target_hit = high >= take_profit
        else:
            stop_hit = high >= stop_loss
            target_hit = low <= take_profit
        if stop_hit and target_hit:
            return index, stop_loss, "stop_loss_first_assumption"
        if stop_hit:
            return index, stop_loss, "stop_loss"
        if target_hit:
            return index, take_profit, "take_profit"
    return last_index, float(data.iloc[last_index]["close"]), "max_hold"


def trade_return(entry_price: float, exit_price: float, direction: str, commission: float) -> float:
    if direction == "bullish":
        gross = exit_price / entry_price - 1
    else:
        gross = entry_price / exit_price - 1
    return gross - commission * 2


def summarize_stock(
    ts_code: str,
    name: str,
    data: pd.DataFrame,
    signals: list[IfvgSignal],
    trades: list[IfvgTrade],
) -> IfvgStockResult:
    returns = [trade.net_return for trade in trades]
    equity_curve = build_equity_curve(returns)
    win_rate = percentage(sum(1 for value in returns if value > 0) / len(returns)) if returns else None
    total_return = percentage(equity_curve[-1] - 1) if equity_curve else None
    buy_hold_return = buy_hold_return_pct(data)
    excess_return = (
        total_return - buy_hold_return if total_return is not None and buy_hold_return is not None else None
    )
    losses = abs(sum(value for value in returns if value < 0))
    gains = sum(value for value in returns if value > 0)
    profit_factor = gains / losses if losses > 0 else (None if gains == 0 else math.inf)
    status = "tested" if trades else "no_trades"
    note = "research-only daily IFVG approximation"
    return IfvgStockResult(
        ts_code=ts_code,
        name=name,
        bars=int(len(data)),
        first_date=iso_date(data.iloc[0]["trade_date"]) if not data.empty else None,
        last_date=iso_date(data.iloc[-1]["trade_date"]) if not data.empty else None,
        signals=len(signals),
        trades=len(trades),
        win_rate_pct=round(win_rate, 2) if win_rate is not None else None,
        avg_return_pct=round(percentage(sum(returns) / len(returns)), 2) if returns else None,
        total_return_pct=round(total_return, 2) if total_return is not None else None,
        buy_hold_return_pct=round(buy_hold_return, 2) if buy_hold_return is not None else None,
        excess_vs_buy_hold_pct=round(excess_return, 2) if excess_return is not None else None,
        max_drawdown_pct=round(max_drawdown_pct(equity_curve), 2) if equity_curve else None,
        profit_factor=round(profit_factor, 4) if profit_factor is not None and math.isfinite(profit_factor) else profit_factor,
        avg_r_multiple=round(sum(trade.r_multiple for trade in trades) / len(trades), 4) if trades else None,
        status=status,
        note=note,
    )


def buy_hold_return_pct(data: pd.DataFrame, commission: float = DEFAULT_COMMISSION) -> float | None:
    if data.empty:
        return None
    entry_price = float(data.iloc[0]["open"])
    exit_price = float(data.iloc[-1]["close"])
    if entry_price <= 0:
        return None
    return percentage(exit_price * (1 - commission) / (entry_price * (1 + commission)) - 1)


def build_equity_curve(returns: list[float]) -> list[float]:
    equity = 1.0
    curve: list[float] = []
    for value in returns:
        equity *= 1 + value
        curve.append(equity)
    return curve


def max_drawdown_pct(equity_curve: list[float]) -> float:
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            max_dd = min(max_dd, value / peak - 1)
    return percentage(max_dd)


def percentage(value: float) -> float:
    return value * 100


def iso_date(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def run_ifvg_batch(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = 100,
    min_bars: int = 420,
    start_date: str | None = None,
    end_date: str | None = None,
    selection_date: str | None = None,
    selection_by: str = "code",
    selection_offset: int = 0,
    pretrade_filter: str | None = None,
    **kwargs: Any,
) -> IfvgBatchResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = select_cache_paths(cache_dir, selection_date=selection_date, selection_by=selection_by)
    if selection_offset < 0:
        raise ValueError("selection_offset must be non-negative")
    if selection_offset:
        paths = paths[selection_offset:]
    if limit is not None:
        paths = paths[:limit]
    selected_paths = paths
    if pretrade_filter:
        if not selection_date:
            raise ValueError("selection_date is required when pretrade_filter is set")
        selected_paths = [
            path for path in paths if frame_passes_pretrade_filter(load_daily_bar_csv(path), selection_date, pretrade_filter)
        ]
    results: list[IfvgStockResult] = []
    all_trades: list[IfvgTrade] = []
    files_scanned = len(paths)
    for path in selected_paths:
        frame = load_daily_bar_csv(path)
        frame = slice_date_window(frame, start_date, end_date)
        if len(frame) < min_bars:
            continue
        ts_code = str(frame.iloc[0].get("code") or infer_ts_code_from_path(path))
        name = str(frame.iloc[0].get("name") or "")
        result, trades, _ = run_ifvg_backtest(frame, ts_code=ts_code, name=name, **kwargs)
        results.append(result)
        all_trades.extend(trades)

    result_path = output_dir / "ifvg_stock_summary.csv"
    trades_path = output_dir / "ifvg_trades.csv"
    batch_path = output_dir / "ifvg_batch_summary.json"
    write_dataclass_csv(result_path, results, IfvgStockResult)
    write_dataclass_csv(trades_path, all_trades, IfvgTrade)
    batch = summarize_batch(output_dir, files_scanned, results, all_trades, result_path, trades_path)
    batch_path.write_text(json.dumps(asdict(batch), ensure_ascii=False, indent=2), encoding="utf-8")
    return batch


def summarize_batch(
    output_dir: Path,
    files_scanned: int,
    results: list[IfvgStockResult],
    trades: list[IfvgTrade],
    result_path: Path,
    trades_path: Path,
) -> IfvgBatchResult:
    returns = [trade.net_return for trade in trades]
    wins = sum(1 for value in returns if value > 0)
    gains = sum(value for value in returns if value > 0)
    losses = abs(sum(value for value in returns if value < 0))
    profit_factor = gains / losses if losses > 0 else (None if gains == 0 else math.inf)
    equity_curve = build_equity_curve(returns)
    stock_returns = [
        result.total_return_pct for result in results if result.total_return_pct is not None and result.trades > 0
    ]
    buy_hold_returns = [result.buy_hold_return_pct for result in results if result.buy_hold_return_pct is not None]
    excess_returns = [
        result.excess_vs_buy_hold_pct
        for result in results
        if result.excess_vs_buy_hold_pct is not None and result.trades > 0
    ]
    return IfvgBatchResult(
        output_dir=str(output_dir),
        files_scanned=files_scanned,
        stocks_tested=len(results),
        stocks_with_trades=sum(1 for result in results if result.trades > 0),
        total_trades=len(trades),
        win_rate_pct=round(percentage(wins / len(returns)), 2) if returns else None,
        avg_return_pct=round(percentage(sum(returns) / len(returns)), 2) if returns else None,
        total_return_pct=round(sum(stock_returns) / len(stock_returns), 2) if stock_returns else None,
        buy_hold_return_pct=round(sum(buy_hold_returns) / len(buy_hold_returns), 2) if buy_hold_returns else None,
        excess_vs_buy_hold_pct=round(sum(excess_returns) / len(excess_returns), 2) if excess_returns else None,
        stocks_outperform_buy_hold=sum(
            1 for result in results if result.excess_vs_buy_hold_pct is not None and result.excess_vs_buy_hold_pct > 0
        ),
        trade_compound_return_pct=round(percentage(equity_curve[-1] - 1), 2) if equity_curve else None,
        profit_factor=round(profit_factor, 4) if profit_factor is not None and math.isfinite(profit_factor) else profit_factor,
        result_path=str(result_path),
        trades_path=str(trades_path),
    )


def latest_cache_paths(cache_dir: Path) -> list[Path]:
    latest_by_code: dict[str, Path] = {}
    for path in sorted(cache_dir.glob("*.csv")):
        ts_code = infer_ts_code_from_path(path)
        current = latest_by_code.get(ts_code)
        if current is None or cache_end_date(path) > cache_end_date(current):
            latest_by_code[ts_code] = path
    return sorted(latest_by_code.values(), key=lambda item: infer_ts_code_from_path(item))


def select_cache_paths(cache_dir: Path, selection_date: str | None, selection_by: str) -> list[Path]:
    paths = latest_cache_paths(cache_dir)
    if selection_by == "code":
        return paths
    if selection_by != "amount":
        raise ValueError("selection_by must be one of: code, amount")
    if not selection_date:
        raise ValueError("selection_date is required when selection_by=amount")
    ranked: list[tuple[float, Path]] = []
    for path in paths:
        frame = pd.read_csv(path, usecols=["trade_date", "amount"])
        hit = frame[frame["trade_date"].astype(str).eq(selection_date)]
        if hit.empty:
            continue
        amount = pd.to_numeric(hit.iloc[-1]["amount"], errors="coerce")
        if pd.isna(amount):
            continue
        ranked.append((float(amount), path))
    return [path for _, path in sorted(ranked, key=lambda item: item[0], reverse=True)]


def frame_passes_pretrade_filter(frame: pd.DataFrame, selection_date: str, expression: str) -> bool:
    factor, operator, raw_value = parse_pretrade_filter(expression)
    factor_frame = build_factor_frame(frame)
    row = factor_frame[factor_frame["trade_date"].eq(pd.Timestamp(selection_date))]
    if row.empty or factor not in row.columns:
        return False
    value = pd.to_numeric(row.iloc[-1][factor], errors="coerce")
    if pd.isna(value):
        return False
    threshold = float(raw_value)
    if operator == "<":
        return float(value) < threshold
    if operator == "<=":
        return float(value) <= threshold
    if operator == ">":
        return float(value) > threshold
    if operator == ">=":
        return float(value) >= threshold
    if operator == "==":
        return float(value) == threshold
    raise ValueError(f"unsupported pretrade filter operator: {operator}")


def parse_pretrade_filter(expression: str) -> tuple[str, str, str]:
    for operator in ("<=", ">=", "==", "<", ">"):
        if operator in expression:
            left, right = expression.split(operator, 1)
            factor = left.strip()
            threshold = right.strip()
            if not factor or not threshold:
                break
            return factor, operator, threshold
    raise ValueError("pretrade filter must look like: factor<=number")


def cache_end_date(path: Path) -> str:
    parts = path.stem.split("_")
    return parts[-1] if parts else ""


def slice_date_window(frame: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    data = frame.copy()
    if start_date:
        data = data[data["trade_date"] >= pd.Timestamp(start_date)]
    if end_date:
        data = data[data["trade_date"] <= pd.Timestamp(end_date)]
    return data.reset_index(drop=True)


def write_dataclass_csv(path: Path, rows: list[Any], row_type: type[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row_type.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run research-only daily IFVG approximation backtests.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--risk-reward", type=float, default=1.5)
    parser.add_argument("--max-hold-bars", type=int, default=10)
    parser.add_argument("--htf-window", type=int, default=60)
    parser.add_argument("--fvg-lookback", type=int, default=80)
    parser.add_argument("--min-gap-pct", type=float, default=0.002)
    parser.add_argument("--confirm-window", type=int, default=5)
    parser.add_argument("--start-date", default="2024-06-05")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--selection-date", default=None)
    parser.add_argument("--selection-by", default="code", choices=["code", "amount"])
    parser.add_argument("--selection-offset", type=int, default=0)
    parser.add_argument("--pretrade-filter", default=None)
    parser.add_argument("--directions", default="bullish", help="Comma-separated: bullish,bearish")
    args = parser.parse_args(argv)
    directions = tuple(item.strip() for item in args.directions.split(",") if item.strip())
    result = run_ifvg_batch(
        cache_dir=Path(args.cache_dir),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        start_date=args.start_date,
        end_date=args.end_date,
        selection_date=args.selection_date,
        selection_by=args.selection_by,
        selection_offset=args.selection_offset,
        pretrade_filter=args.pretrade_filter,
        risk_reward=args.risk_reward,
        max_hold_bars=args.max_hold_bars,
        htf_window=args.htf_window,
        fvg_lookback=args.fvg_lookback,
        min_gap_pct=args.min_gap_pct,
        confirm_window=args.confirm_window,
        directions=directions,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
