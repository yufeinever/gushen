from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

from rich.console import Console
from rich.table import Table

from gushen.data import DailyBar, fetch_daily_bars
from gushen.trade_calendar import latest_research_trade_date


DEFAULT_UNIVERSE = {
    "000066.SZ": "China Greatwall",
    "002208.SZ": "Hefei Urban Construction",
    "688126.SH": "National Silicon Industry",
    "300058.SZ": "BlueFocus",
    "002185.SZ": "Huatian Technology",
}


@dataclass(frozen=True)
class BacktestConfig:
    start_date: str
    end_date: str
    adjust: str = "qfq"
    max_hold_days: int = 5
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0005
    slippage_rate: float = 0.0005
    source_note: str = "research-grade qfq daily bars; no limit/suspension replay yet"


@dataclass(frozen=True)
class TradeRow:
    code: str
    name: str
    signal_date: str
    entry_date: str
    exit_date: str
    hold_days: int
    entry_price: float
    exit_price: float
    gross_return: float
    net_return: float
    exit_reason: str
    signal_close: float
    ma5: float
    ma10: float
    ma20: float
    amount_ratio_5d: float
    signal_note: str


@dataclass(frozen=True)
class SummaryRow:
    code: str
    name: str
    bars_count: int
    first_date: str
    last_date: str
    signal_count: int
    trade_count: int
    win_rate: float | None
    avg_net_return: float | None
    median_net_return: float | None
    total_net_return: float
    max_drawdown: float
    avg_hold_days: float | None
    source_status: str
    note: str


def run_fixed5_t1_technical_backtest(
    end_date: str | None = None,
    years: int = 5,
    adjust: str = "qfq",
) -> tuple[list[TradeRow], list[SummaryRow]]:
    console = Console()
    end_date = end_date or latest_research_trade_date()
    start_date = (date.fromisoformat(end_date) - timedelta(days=365 * years + 30)).isoformat()
    config = BacktestConfig(start_date=start_date, end_date=end_date, adjust=adjust)
    histories = {
        code: load_or_fetch_backtest_history(code, name, config)
        for code, name in DEFAULT_UNIVERSE.items()
    }
    trades: list[TradeRow] = []
    summaries: list[SummaryRow] = []
    for code, name in DEFAULT_UNIVERSE.items():
        rows = histories.get(code, [])
        stock_trades, signal_count = backtest_stock(code, name, rows, config)
        trades.extend(stock_trades)
        summaries.append(summarize_stock(code, name, rows, stock_trades, signal_count))
    summaries.append(summarize_overall(trades, summaries))
    write_backtest_outputs(config, trades, summaries)
    print_backtest_summary(console, config, summaries)
    return trades, summaries


def load_or_fetch_backtest_history(
    code: str,
    name: str,
    config: BacktestConfig,
    cache_dir: Path = Path("data/local/backtests/fixed5_technical"),
) -> list[DailyBar]:
    cache_path = cache_dir / config.adjust / f"{code}_{config.start_date}_{config.end_date}.csv"
    if cache_path.exists():
        rows = _read_bars(cache_path)
        if rows:
            return rows
        cache_path.unlink(missing_ok=True)
    rows = fetch_daily_bars(
        code=code,
        name=name,
        start_date=config.start_date,
        end_date=config.end_date,
        adjust=config.adjust,
    )
    rows = sorted(rows, key=lambda item: item.trade_date)
    if not rows:
        raise RuntimeError(f"No daily bars fetched for {code} from {config.start_date} to {config.end_date}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _write_bars(cache_path, rows)
    return rows


def backtest_stock(
    code: str,
    name: str,
    bars: list[DailyBar],
    config: BacktestConfig,
) -> tuple[list[TradeRow], int]:
    rows = sorted([bar for bar in bars if bar.close > 0], key=lambda item: item.trade_date)
    trades: list[TradeRow] = []
    signal_count = 0
    index = 60
    while index < len(rows) - config.max_hold_days - 2:
        signal = build_signal(rows, index)
        if not signal:
            index += 1
            continue
        signal_count += 1
        entry_index = index + 1
        exit_index, exit_price, reason = simulate_exit(rows, entry_index, config)
        entry_price = rows[entry_index].open
        gross_return = exit_price / entry_price - 1 if entry_price else 0.0
        net_return = _net_return(entry_price, exit_price, config)
        trades.append(
            TradeRow(
                code=code,
                name=name,
                signal_date=rows[index].trade_date,
                entry_date=rows[entry_index].trade_date,
                exit_date=rows[exit_index].trade_date,
                hold_days=exit_index - entry_index + 1,
                entry_price=round(entry_price, 4),
                exit_price=round(exit_price, 4),
                gross_return=round(gross_return, 6),
                net_return=round(net_return, 6),
                exit_reason=reason,
                signal_close=round(rows[index].close, 4),
                ma5=round(signal["ma5"], 4),
                ma10=round(signal["ma10"], 4),
                ma20=round(signal["ma20"], 4),
                amount_ratio_5d=round(signal["amount_ratio_5d"], 4),
                signal_note=signal["note"],
            )
        )
        index = exit_index + 1
    return trades, signal_count


def build_signal(bars: list[DailyBar], index: int) -> dict[str, float | str] | None:
    if index < 60:
        return None
    bar = bars[index]
    closes = [item.close for item in bars[: index + 1]]
    amounts = [item.amount for item in bars[: index + 1]]
    ma5 = _ma(closes, 5)
    ma10 = _ma(closes, 10)
    ma20 = _ma(closes, 20)
    ma60 = _ma(closes, 60)
    if None in {ma5, ma10, ma20, ma60}:
        return None
    amount_ratio = _amount_ratio(amounts, 5)
    day_range = bar.high - bar.low
    upper_shadow = (bar.high - bar.close) / day_range if day_range > 0 else 0.0
    ma5_gap = bar.close / ma5 - 1 if ma5 else 0.0
    ma20_gap = bar.close / ma20 - 1 if ma20 else 0.0
    checks = [
        bar.close < 50,
        bar.close > ma20,
        ma5 >= ma10 >= ma20,
        ma20 >= ma60 * 0.97,
        0 <= ma5_gap <= 0.08,
        0 <= ma20_gap <= 0.18,
        1.2 <= amount_ratio <= 3.0,
        0 < bar.pct_change < 0.085,
        upper_shadow < 0.45,
    ]
    if not all(checks):
        return None
    return {
        "ma5": ma5,
        "ma10": ma10,
        "ma20": ma20,
        "amount_ratio_5d": amount_ratio,
        "note": "close<50; MA5>=MA10>=MA20; volume expansion; no long upper shadow",
    }


def simulate_exit(
    bars: list[DailyBar],
    entry_index: int,
    config: BacktestConfig,
) -> tuple[int, float, str]:
    entry = bars[entry_index]
    stop_price = entry.open * (1 - config.stop_loss_pct)
    target_price = entry.open * (1 + config.take_profit_pct)
    last_index = min(len(bars) - 1, entry_index + config.max_hold_days - 1)
    for index in range(entry_index + 1, last_index + 1):
        bar = bars[index]
        if bar.open <= stop_price:
            return index, bar.open, "stop_gap_open"
        if bar.low <= stop_price:
            return index, stop_price, "stop_loss"
        if bar.high >= target_price:
            return index, target_price, "take_profit"
        ma20 = _ma([item.close for item in bars[: index + 1]], 20)
        if ma20 is not None and bar.close < ma20:
            next_index = min(index + 1, len(bars) - 1)
            return next_index, bars[next_index].open, "ma20_break_next_open"
    return last_index, bars[last_index].close, "time_exit"


def summarize_stock(
    code: str,
    name: str,
    bars: list[DailyBar],
    trades: list[TradeRow],
    signal_count: int,
) -> SummaryRow:
    net_returns = [trade.net_return for trade in trades]
    return SummaryRow(
        code=code,
        name=name,
        bars_count=len(bars),
        first_date=bars[0].trade_date if bars else "",
        last_date=bars[-1].trade_date if bars else "",
        signal_count=signal_count,
        trade_count=len(trades),
        win_rate=_win_rate(net_returns),
        avg_net_return=_mean_or_none(net_returns),
        median_net_return=_median_or_none(net_returns),
        total_net_return=round(_compound_return(net_returns), 6),
        max_drawdown=round(_max_drawdown(net_returns), 6),
        avg_hold_days=_mean_or_none([trade.hold_days for trade in trades]),
        source_status="ok" if len(bars) >= 250 else "insufficient_history",
        note="fixed-universe T+1 technical timing backtest; research only",
    )


def summarize_overall(trades: list[TradeRow], summaries: list[SummaryRow]) -> SummaryRow:
    net_returns = [trade.net_return for trade in sorted(trades, key=lambda item: item.entry_date)]
    bars_count = min((row.bars_count for row in summaries), default=0)
    first_date = min((row.first_date for row in summaries if row.first_date), default="")
    last_date = max((row.last_date for row in summaries if row.last_date), default="")
    return SummaryRow(
        code="__ALL__",
        name="Fixed five-stock universe",
        bars_count=bars_count,
        first_date=first_date,
        last_date=last_date,
        signal_count=sum(row.signal_count for row in summaries),
        trade_count=len(trades),
        win_rate=_win_rate(net_returns),
        avg_net_return=_mean_or_none(net_returns),
        median_net_return=_median_or_none(net_returns),
        total_net_return=round(_compound_return(net_returns), 6),
        max_drawdown=round(_max_drawdown(net_returns), 6),
        avg_hold_days=_mean_or_none([trade.hold_days for trade in trades]),
        source_status="ok" if summaries and all(row.source_status == "ok" for row in summaries) else "partial",
        note="pooled trade statistics; not a portfolio equity curve",
    )


def write_backtest_outputs(
    config: BacktestConfig,
    trades: list[TradeRow],
    summaries: list[SummaryRow],
) -> None:
    output_dir = Path("reports/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"fixed5_t1_technical_backtest_{config.end_date}"
    _write_dataclass_csv(output_dir / f"{prefix}_trades.csv", trades)
    _write_dataclass_csv(output_dir / f"{prefix}_summary.csv", summaries)
    payload = {
        "config": asdict(config),
        "data_sufficiency": (
            "research_only: fixed five-stock universe, qfq daily bars, no historical Top100 replay, "
            "no complete limit/suspension execution constraints"
        ),
        "summary": [asdict(row) for row in summaries],
        "trades": [asdict(row) for row in trades],
    }
    (output_dir / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_backtest_summary(
    console: Console,
    config: BacktestConfig,
    summaries: list[SummaryRow],
) -> None:
    table = Table(title=f"Fixed 5 T+1 technical backtest {config.start_date} to {config.end_date}")
    table.add_column("Code")
    table.add_column("Trades", justify="right")
    table.add_column("Win", justify="right")
    table.add_column("Avg Net", justify="right")
    table.add_column("Total Net", justify="right")
    table.add_column("Max DD", justify="right")
    for row in summaries:
        table.add_row(
            row.code,
            str(row.trade_count),
            _pct(row.win_rate),
            _pct(row.avg_net_return),
            f"{row.total_net_return:.2%}",
            f"{row.max_drawdown:.2%}",
        )
    console.print("Data sufficiency: research_only; this is not an execution-grade backtest.")
    console.print(table)
    console.print(f"Reports: reports/generated/fixed5_t1_technical_backtest_{config.end_date}_*.csv")


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


def _write_dataclass_csv(path: Path, rows: list) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _ma(values: list[float], periods: int) -> float | None:
    if len(values) < periods:
        return None
    return mean(values[-periods:])


def _amount_ratio(values: list[float], periods: int) -> float:
    if len(values) < periods + 1:
        return 1.0
    avg = mean(values[-periods - 1 : -1])
    return values[-1] / avg if avg else 1.0


def _net_return(entry_price: float, exit_price: float, config: BacktestConfig) -> float:
    buy_cost = config.commission_rate + config.slippage_rate
    sell_cost = config.commission_rate + config.stamp_tax_rate + config.slippage_rate
    cash_out = entry_price * (1 + buy_cost)
    cash_in = exit_price * (1 - sell_cost)
    return cash_in / cash_out - 1 if cash_out else 0.0


def _win_rate(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(value > 0 for value in values) / len(values)


def _mean_or_none(values: list[float | int]) -> float | None:
    if not values:
        return None
    return round(float(mean(values)), 6)


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[mid], 6)
    return round((ordered[mid - 1] + ordered[mid]) / 2, 6)


def _compound_return(values: list[float]) -> float:
    equity = 1.0
    for value in values:
        equity *= 1 + value
    return equity - 1


def _max_drawdown(values: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for value in values:
        equity *= 1 + value
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1)
    return max_dd


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def main() -> None:
    run_fixed5_t1_technical_backtest(end_date="2026-05-20")


if __name__ == "__main__":
    main()
