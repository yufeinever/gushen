from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd

from gushen.trade_calendar import latest_research_trade_date


DEFAULT_CACHE_DIR = Path("data/local/guided_factor_backtests/daily_bars/qfq")
DEFAULT_OUTPUT_DIR = Path("reports/generated/top20_ma5_pullback")
DEFAULT_START_DATE = "2024-06-03"


@dataclass(frozen=True)
class StrategyConfig:
    start_date: str
    end_date: str
    top_n: int = 20
    wait_days: int = 5
    ma_window: int = 5
    max_hold_days: int = 1
    initial_cash: float = 100_000.0
    position_pct: float = 0.20
    max_positions: int = 1
    commission_rate: float = 0.0008
    slippage_rate: float = 0.0005
    limit_up_pct: float = 0.098
    source_note: str = (
        "daily-bar proxy for a video strategy; no intraday 10:00/14:30 execution, "
        "no real limit-order queue, no suspension replay"
    )


@dataclass(frozen=True)
class CandidateRow:
    signal_date: str
    code: str
    name: str
    amount_rank: int
    close: float
    ma5: float
    amount: float


@dataclass(frozen=True)
class TradeRow:
    signal_date: str
    entry_date: str
    exit_date: str
    code: str
    name: str
    amount_rank: int
    entry_price: float
    exit_price: float
    gross_return_pct: float
    net_return_pct: float
    exit_reason: str
    wait_days: int
    entry_ma5: float
    next_day_high: float
    next_day_close: float
    note: str


@dataclass(frozen=True)
class PortfolioLedgerRow:
    signal_date: str
    entry_date: str
    exit_date: str
    code: str
    name: str
    status: str
    shares: int
    entry_price: float
    exit_price: float
    cash_invested: float
    cash_returned: float
    pnl: float
    net_return_pct: float
    cash_after: float
    equity_after: float
    reason: str


@dataclass(frozen=True)
class StrategySummary:
    start_date: str
    end_date: str
    top_n: int
    candidates: int
    trades: int
    filled_trades: int
    skipped_trades: int
    win_rate_pct: float | None
    avg_net_return_pct: float | None
    median_net_return_pct: float | None
    total_compound_return_pct: float | None
    final_equity: float
    portfolio_return_pct: float
    max_drawdown_pct: float
    source_note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest a daily-bar proxy for Top20 amount + MA5 pullback ultra-short strategy."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--wait-days", type=int, default=5)
    parser.add_argument("--max-positions", type=int, default=1)
    parser.add_argument("--position-pct", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end_date = args.end_date or infer_latest_cache_date(args.cache_dir) or latest_research_trade_date()
    config = StrategyConfig(
        start_date=args.start_date,
        end_date=end_date,
        top_n=args.top_n,
        wait_days=args.wait_days,
        max_positions=args.max_positions,
        position_pct=args.position_pct,
    )
    result = run_top20_ma5_pullback_backtest(args.cache_dir, args.output_dir, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_top20_ma5_pullback_backtest(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: StrategyConfig | None = None,
) -> dict[str, Any]:
    config = config or StrategyConfig(
        start_date=DEFAULT_START_DATE,
        end_date=infer_latest_cache_date(cache_dir) or latest_research_trade_date(),
    )
    frames = load_daily_frames(cache_dir, config)
    if not frames:
        raise RuntimeError(f"no daily bar CSV files found in {cache_dir}")
    market = build_market_frame(frames, config)
    candidates = select_candidates(market, config)
    trades = build_trades(frames, candidates, config)
    ledger, summary = simulate_portfolio(trades, len(candidates), config)
    write_outputs(output_dir, config, candidates, trades, ledger, summary)
    return {
        "summary": asdict(summary),
        "output_dir": str(output_dir),
        "candidates_path": str(output_dir / "top20_ma5_candidates.csv"),
        "trades_path": str(output_dir / "top20_ma5_trades.csv"),
        "portfolio_path": str(output_dir / "top20_ma5_portfolio.csv"),
        "summary_path": str(output_dir / "top20_ma5_summary.json"),
    }


def infer_latest_cache_date(cache_dir: Path) -> str | None:
    latest: str | None = None
    for path in cache_dir.glob("*.csv"):
        parts = path.stem.split("_")
        if len(parts) >= 3 and len(parts[-1]) == 10 and (latest is None or parts[-1] > latest):
            latest = parts[-1]
    return latest


def load_daily_frames(cache_dir: Path, config: StrategyConfig) -> dict[str, pd.DataFrame]:
    paths = choose_latest_paths_by_code(cache_dir)
    frames: dict[str, pd.DataFrame] = {}
    warmup_start = (date.fromisoformat(config.start_date) - timedelta(days=30)).isoformat()
    for code, path in paths.items():
        frame = pd.read_csv(path)
        required = {"trade_date", "open", "high", "low", "close", "amount"}
        if missing := required - set(frame.columns):
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        data = frame.copy()
        data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.strftime("%Y-%m-%d")
        for column in ["open", "high", "low", "close", "amount"]:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        if "code" not in data.columns:
            data["code"] = code
        if "name" not in data.columns:
            data["name"] = ""
        data["code"] = data["code"].map(normalize_ts_code)
        data["name"] = data["name"].fillna("").astype(str)
        data = data[(data["trade_date"] >= warmup_start) & (data["trade_date"] <= config.end_date)]
        data = data.dropna(subset=["open", "high", "low", "close", "amount"])
        data = data.sort_values("trade_date").reset_index(drop=True)
        if len(data) >= config.ma_window + config.wait_days + config.max_hold_days + 1:
            data["ma5"] = data["close"].rolling(config.ma_window).mean()
            frames[code] = data
    return frames


def choose_latest_paths_by_code(cache_dir: Path) -> dict[str, Path]:
    paths: dict[str, tuple[str, Path]] = {}
    for path in cache_dir.glob("*.csv"):
        parts = path.stem.split("_")
        if len(parts) < 3:
            continue
        code = normalize_ts_code(parts[0])
        end_date = parts[-1]
        previous = paths.get(code)
        if previous is None or end_date > previous[0]:
            paths[code] = (end_date, path)
    return {code: path for code, (_, path) in paths.items()}


def build_market_frame(frames: dict[str, pd.DataFrame], config: StrategyConfig) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for code, frame in frames.items():
        subset = frame[
            (frame["trade_date"] >= config.start_date)
            & (frame["trade_date"] <= config.end_date)
            & frame["ma5"].notna()
        ][["trade_date", "code", "name", "close", "high", "low", "amount", "ma5"]].copy()
        subset["code"] = code
        rows.append(subset)
    if not rows:
        return pd.DataFrame()
    market = pd.concat(rows, ignore_index=True)
    market = market[~market["code"].map(is_excluded_board)]
    market = market[~market["name"].map(is_st_name)]
    market = market[market["amount"] > 0]
    market["amount_rank"] = market.groupby("trade_date")["amount"].rank(method="first", ascending=False)
    return market


def select_candidates(market: pd.DataFrame, config: StrategyConfig) -> list[CandidateRow]:
    if market.empty:
        return []
    selected = market[(market["amount_rank"] <= config.top_n) & (market["close"] > market["ma5"])].copy()
    selected = selected.sort_values(["trade_date", "amount_rank", "code"])
    return [
        CandidateRow(
            signal_date=str(row["trade_date"]),
            code=str(row["code"]),
            name=str(row.get("name") or ""),
            amount_rank=int(row["amount_rank"]),
            close=round_float(row["close"]),
            ma5=round_float(row["ma5"]),
            amount=round_float(row["amount"]),
        )
        for row in selected.to_dict("records")
    ]


def build_trades(
    frames: dict[str, pd.DataFrame],
    candidates: list[CandidateRow],
    config: StrategyConfig,
) -> list[TradeRow]:
    trades: list[TradeRow] = []
    for candidate in candidates:
        frame = frames.get(candidate.code)
        if frame is None:
            continue
        matches = frame.index[frame["trade_date"] == candidate.signal_date].tolist()
        if not matches:
            continue
        signal_index = matches[0]
        entry = find_pullback_entry(frame, signal_index, config)
        if entry is None:
            continue
        entry_index, entry_price, entry_ma5 = entry
        exit_index = min(entry_index + config.max_hold_days, len(frame) - 1)
        if exit_index <= entry_index:
            continue
        exit_row = frame.iloc[exit_index]
        exit_price, exit_reason = daily_proxy_exit_price(float(frame.iloc[entry_index]["close"]), exit_row, config)
        gross_return = exit_price / entry_price - 1.0
        net_return = net_trade_return(entry_price, exit_price, config)
        trades.append(
            TradeRow(
                signal_date=candidate.signal_date,
                entry_date=str(frame.iloc[entry_index]["trade_date"]),
                exit_date=str(exit_row["trade_date"]),
                code=candidate.code,
                name=candidate.name,
                amount_rank=candidate.amount_rank,
                entry_price=round_float(entry_price),
                exit_price=round_float(exit_price),
                gross_return_pct=round_float(gross_return * 100),
                net_return_pct=round_float(net_return * 100),
                exit_reason=exit_reason,
                wait_days=entry_index - signal_index,
                entry_ma5=round_float(entry_ma5),
                next_day_high=round_float(exit_row["high"]),
                next_day_close=round_float(exit_row["close"]),
                note=config.source_note,
            )
        )
    return trades


def find_pullback_entry(
    frame: pd.DataFrame,
    signal_index: int,
    config: StrategyConfig,
) -> tuple[int, float, float] | None:
    for index in range(signal_index + 1, min(signal_index + config.wait_days + 1, len(frame) - 1)):
        row = frame.iloc[index]
        ma5 = float(row["ma5"])
        if not math.isfinite(ma5):
            continue
        if float(row["low"]) <= ma5 <= float(row["high"]) and float(row["close"]) >= ma5:
            return index, ma5 * (1.0 + config.slippage_rate), ma5
    return None


def daily_proxy_exit_price(previous_close: float, exit_row: pd.Series, config: StrategyConfig) -> tuple[float, str]:
    high = float(exit_row["high"])
    close = float(exit_row["close"])
    limit_price = previous_close * (1.0 + config.limit_up_pct)
    if high >= limit_price and close >= limit_price * 0.995:
        return close * (1.0 - config.slippage_rate), "daily_proxy_limit_up_hold_to_close"
    proxy_price = max(float(exit_row["open"]), min(high, (high + close) / 2.0))
    return proxy_price * (1.0 - config.slippage_rate), "daily_proxy_next_day_strength_exit"


def simulate_portfolio(
    trades: list[TradeRow],
    candidate_count: int,
    config: StrategyConfig,
) -> tuple[list[PortfolioLedgerRow], StrategySummary]:
    cash = config.initial_cash
    open_until: list[str] = []
    ledger: list[PortfolioLedgerRow] = []
    equity_curve = [cash]
    filled_returns: list[float] = []
    skipped = 0
    for trade in sorted(trades, key=lambda item: (item.entry_date, item.amount_rank, item.code)):
        open_until = [exit_date for exit_date in open_until if exit_date >= trade.entry_date]
        if len(open_until) >= config.max_positions:
            skipped += 1
            ledger.append(make_skipped_ledger(trade, cash, "max_positions"))
            continue
        budget = cash * config.position_pct
        shares = int(budget // (trade.entry_price * 100)) * 100
        if shares <= 0:
            skipped += 1
            ledger.append(make_skipped_ledger(trade, cash, "insufficient_cash"))
            continue
        invested = shares * trade.entry_price
        returned = shares * trade.exit_price * (1.0 - config.commission_rate)
        pnl = returned - invested
        cash += pnl
        open_until.append(trade.exit_date)
        filled_returns.append(trade.net_return_pct)
        equity_curve.append(cash)
        ledger.append(
            PortfolioLedgerRow(
                trade.signal_date,
                trade.entry_date,
                trade.exit_date,
                trade.code,
                trade.name,
                "filled",
                shares,
                trade.entry_price,
                trade.exit_price,
                round_float(invested),
                round_float(returned),
                round_float(pnl),
                trade.net_return_pct,
                round_float(cash),
                round_float(cash),
                "filled",
            )
        )
    summary = StrategySummary(
        start_date=config.start_date,
        end_date=config.end_date,
        top_n=config.top_n,
        candidates=candidate_count,
        trades=len(trades),
        filled_trades=len(filled_returns),
        skipped_trades=skipped,
        win_rate_pct=round_float(sum(1 for item in filled_returns if item > 0) / len(filled_returns) * 100)
        if filled_returns
        else None,
        avg_net_return_pct=round_float(mean(filled_returns)) if filled_returns else None,
        median_net_return_pct=round_float(median(filled_returns)) if filled_returns else None,
        total_compound_return_pct=round_float(compound_pct(filled_returns)) if filled_returns else None,
        final_equity=round_float(cash),
        portfolio_return_pct=round_float((cash / config.initial_cash - 1.0) * 100),
        max_drawdown_pct=round_float(max_drawdown_pct(equity_curve)),
        source_note=config.source_note,
    )
    return ledger, summary


def make_skipped_ledger(trade: TradeRow, cash: float, reason: str) -> PortfolioLedgerRow:
    return PortfolioLedgerRow(
        trade.signal_date,
        trade.entry_date,
        trade.exit_date,
        trade.code,
        trade.name,
        "skipped",
        0,
        trade.entry_price,
        trade.exit_price,
        0.0,
        0.0,
        0.0,
        trade.net_return_pct,
        round_float(cash),
        round_float(cash),
        reason,
    )


def write_outputs(
    output_dir: Path,
    config: StrategyConfig,
    candidates: list[CandidateRow],
    trades: list[TradeRow],
    ledger: list[PortfolioLedgerRow],
    summary: StrategySummary,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_dataclass_csv(output_dir / "top20_ma5_candidates.csv", candidates)
    write_dataclass_csv(output_dir / "top20_ma5_trades.csv", trades)
    write_dataclass_csv(output_dir / "top20_ma5_portfolio.csv", ledger)
    payload = {"config": asdict(config), "summary": asdict(summary)}
    (output_dir / "top20_ma5_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_dataclass_csv(path: Path, rows: list[Any]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    payload = [asdict(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(payload[0]))
        writer.writeheader()
        writer.writerows(payload)


def normalize_ts_code(value: str) -> str:
    code = str(value).strip().upper()
    if "." in code:
        return code
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    if code.startswith(("0", "2", "3")):
        return f"{code}.SZ"
    if code.startswith(("4", "8")):
        return f"{code}.BJ"
    return code


def is_excluded_board(code: str) -> bool:
    raw = code.split(".")[0]
    return (
        code.endswith(".BJ")
        or raw.startswith(("300", "301"))
        or raw.startswith(("688", "689"))
        or raw.startswith(("4", "8"))
    )


def is_st_name(name: str) -> bool:
    normalized = str(name).upper()
    return "ST" in normalized or "退" in normalized


def net_trade_return(entry_price: float, exit_price: float, config: StrategyConfig) -> float:
    buy_cost = entry_price * (1.0 + config.commission_rate)
    sell_value = exit_price * (1.0 - config.commission_rate)
    return sell_value / buy_cost - 1.0


def compound_pct(returns_pct: list[float]) -> float:
    value = 1.0
    for item in returns_pct:
        value *= 1.0 + item / 100.0
    return (value - 1.0) * 100.0


def max_drawdown_pct(equity_curve: list[float]) -> float:
    peak = equity_curve[0] if equity_curve else 0.0
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            max_dd = min(max_dd, value / peak - 1.0)
    return max_dd * 100.0


def round_float(value: Any, digits: int = 6) -> float:
    number = float(value)
    if not math.isfinite(number):
        return 0.0
    return round(number, digits)


if __name__ == "__main__":
    main()
