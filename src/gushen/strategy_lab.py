from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

from rich.console import Console
from rich.table import Table

from gushen.data import DailyBar
from gushen.fixed_universe_backtest import (
    BacktestConfig,
    DEFAULT_UNIVERSE,
    BenchmarkSummary,
    PortfolioSummary,
    PortfolioTradeRow,
    TradeRow,
    load_or_fetch_backtest_history,
    simulate_buy_hold_benchmark,
    simulate_portfolio,
)
from gushen.trade_calendar import latest_research_trade_date


@dataclass(frozen=True)
class StrategySpec:
    name: str
    description: str
    min_bars: int
    max_hold_days: int
    stop_loss_pct: float
    take_profit_pct: float


@dataclass(frozen=True)
class StrategyComparisonRow:
    strategy: str
    final_equity: float
    total_return: float
    max_drawdown: float
    trade_count: int
    skipped_count: int
    win_rate: float | None
    avg_pnl: float | None
    benchmark_return: float
    excess_return: float
    score: float
    note: str


@dataclass(frozen=True)
class StockPickRow:
    code: str
    name: str
    best_strategy: str
    best_trade_count: int
    best_win_rate: float | None
    best_avg_net_return: float | None
    best_total_net_return: float
    buy_hold_return: float
    excess_vs_buy_hold: float
    max_drawdown: float
    action: str
    note: str


STRATEGIES = [
    StrategySpec(
        name="ma20_trend_hold",
        description="Hold trend while close stays above MA20; enter when MA5>MA10>MA20 and volume expands.",
        min_bars=60,
        max_hold_days=20,
        stop_loss_pct=0.08,
        take_profit_pct=0.25,
    ),
    StrategySpec(
        name="ma20_ma60_trend",
        description="Medium-term trend following: close above MA20 and MA20 above MA60.",
        min_bars=80,
        max_hold_days=30,
        stop_loss_pct=0.10,
        take_profit_pct=0.35,
    ),
    StrategySpec(
        name="ma5_ma20_cross",
        description="Moving-average crossover: MA5 crosses above MA20 with positive close.",
        min_bars=60,
        max_hold_days=15,
        stop_loss_pct=0.07,
        take_profit_pct=0.20,
    ),
    StrategySpec(
        name="donchian_20_breakout",
        description="20-day price breakout with volume confirmation.",
        min_bars=80,
        max_hold_days=20,
        stop_loss_pct=0.08,
        take_profit_pct=0.25,
    ),
    StrategySpec(
        name="momentum_20_60",
        description="20-day momentum positive and stronger than 60-day baseline.",
        min_bars=80,
        max_hold_days=25,
        stop_loss_pct=0.09,
        take_profit_pct=0.30,
    ),
]


def run_strategy_lab(
    end_date: str | None = None,
    years: int = 5,
    adjust: str = "qfq",
) -> tuple[list[StrategyComparisonRow], list[StockPickRow]]:
    console = Console()
    end_date = end_date or latest_research_trade_date()
    start_date = (date.fromisoformat(end_date) - timedelta(days=365 * years + 30)).isoformat()
    base_config = BacktestConfig(start_date=start_date, end_date=end_date, adjust=adjust)
    histories = {
        code: load_or_fetch_backtest_history(code, name, base_config)
        for code, name in DEFAULT_UNIVERSE.items()
    }
    buy_hold_rows, benchmark = simulate_buy_hold_benchmark(histories, base_config)
    comparisons: list[StrategyComparisonRow] = []
    strategy_trades: dict[str, list[TradeRow]] = {}
    strategy_portfolios: dict[str, list[PortfolioTradeRow]] = {}
    for spec in STRATEGIES:
        config = _config_for_strategy(base_config, spec)
        trades = build_strategy_trades(histories, spec, config)
        portfolio_trades, portfolio_summary = simulate_portfolio(trades, config)
        comparisons.append(_comparison_row(spec, portfolio_summary, benchmark))
        strategy_trades[spec.name] = trades
        strategy_portfolios[spec.name] = portfolio_trades
    picks = build_stock_picks(strategy_trades, buy_hold_rows)
    write_strategy_lab_outputs(
        base_config,
        benchmark,
        comparisons,
        picks,
        strategy_trades,
        strategy_portfolios,
    )
    print_strategy_lab(console, base_config, benchmark, comparisons, picks)
    return comparisons, picks


def build_strategy_trades(
    histories: dict[str, list[DailyBar]],
    spec: StrategySpec,
    config: BacktestConfig,
) -> list[TradeRow]:
    trades: list[TradeRow] = []
    for code, name in DEFAULT_UNIVERSE.items():
        bars = sorted([bar for bar in histories.get(code, []) if bar.close > 0], key=lambda item: item.trade_date)
        trades.extend(_backtest_strategy_stock(code, name, bars, spec, config))
    return sorted(trades, key=lambda item: (item.entry_date, item.code))


def build_stock_picks(
    strategy_trades: dict[str, list[TradeRow]],
    buy_hold_rows: list,
) -> list[StockPickRow]:
    buy_hold = {row.code: row.net_return for row in buy_hold_rows}
    rows: list[StockPickRow] = []
    for code, name in DEFAULT_UNIVERSE.items():
        best_name = ""
        best_trades: list[TradeRow] = []
        best_score = -999.0
        for strategy, trades in strategy_trades.items():
            stock_trades = [trade for trade in trades if trade.code == code]
            if not stock_trades:
                continue
            score = _compound([trade.net_return for trade in stock_trades]) - abs(
                _max_drawdown_from_returns([trade.net_return for trade in stock_trades])
            )
            if score > best_score:
                best_name = strategy
                best_trades = stock_trades
                best_score = score
        if not best_trades:
            rows.append(
                StockPickRow(
                    code=code,
                    name=name,
                    best_strategy="none",
                    best_trade_count=0,
                    best_win_rate=None,
                    best_avg_net_return=None,
                    best_total_net_return=0.0,
                    buy_hold_return=round(buy_hold.get(code, 0.0), 6),
                    excess_vs_buy_hold=round(-buy_hold.get(code, 0.0), 6),
                    max_drawdown=0.0,
                    action="research_only",
                    note="no strategy trades",
                )
            )
            continue
        returns = [trade.net_return for trade in best_trades]
        total = _compound(returns)
        baseline = buy_hold.get(code, 0.0)
        excess = total - baseline
        rows.append(
            StockPickRow(
                code=code,
                name=name,
                best_strategy=best_name,
                best_trade_count=len(best_trades),
                best_win_rate=_win_rate(returns),
                best_avg_net_return=_mean_or_none(returns),
                best_total_net_return=round(total, 6),
                buy_hold_return=round(baseline, 6),
                excess_vs_buy_hold=round(excess, 6),
                max_drawdown=round(_max_drawdown_from_returns(returns), 6),
                action=_pick_action(total, baseline, len(best_trades)),
                note="strategy-ranked research candidate; not live trading advice",
            )
        )
    return sorted(rows, key=lambda item: (item.action, item.excess_vs_buy_hold), reverse=True)


def _backtest_strategy_stock(
    code: str,
    name: str,
    bars: list[DailyBar],
    spec: StrategySpec,
    config: BacktestConfig,
) -> list[TradeRow]:
    trades: list[TradeRow] = []
    index = spec.min_bars
    while index < len(bars) - spec.max_hold_days - 2:
        signal = _strategy_signal(bars, index, spec)
        if signal is None:
            index += 1
            continue
        entry_index = index + 1
        exit_index, exit_price, reason = _strategy_exit(bars, entry_index, spec, config)
        entry_price = bars[entry_index].open
        trades.append(
            TradeRow(
                code=code,
                name=name,
                signal_date=bars[index].trade_date,
                entry_date=bars[entry_index].trade_date,
                exit_date=bars[exit_index].trade_date,
                hold_days=exit_index - entry_index + 1,
                entry_price=round(entry_price, 4),
                exit_price=round(exit_price, 4),
                gross_return=round(exit_price / entry_price - 1 if entry_price else 0.0, 6),
                net_return=round(_net_return(entry_price, exit_price, config), 6),
                exit_reason=reason,
                signal_close=round(bars[index].close, 4),
                ma5=round(signal["ma5"], 4),
                ma10=round(signal["ma10"], 4),
                ma20=round(signal["ma20"], 4),
                amount_ratio_5d=round(signal["amount_ratio_5d"], 4),
                signal_note=spec.name,
            )
        )
        index = exit_index + 1
    return trades


def _strategy_signal(
    bars: list[DailyBar],
    index: int,
    spec: StrategySpec,
) -> dict[str, float] | None:
    closes = [bar.close for bar in bars[: index + 1]]
    amounts = [bar.amount for bar in bars[: index + 1]]
    bar = bars[index]
    previous = bars[index - 1]
    ma5 = _ma(closes, 5)
    ma10 = _ma(closes, 10)
    ma20 = _ma(closes, 20)
    ma60 = _ma(closes, 60)
    if None in {ma5, ma10, ma20, ma60}:
        return None
    amount_ratio = _amount_ratio(amounts, 5)
    common = bar.close < 50 and bar.pct_change < 0.095
    if not common:
        return None
    ok = False
    if spec.name == "ma20_trend_hold":
        ok = bar.close > ma20 and ma5 > ma10 > ma20 and 1.05 <= amount_ratio <= 3.5
    elif spec.name == "ma20_ma60_trend":
        ok = bar.close > ma20 > ma60 and ma20 / ma60 - 1 <= 0.35 and amount_ratio >= 0.8
    elif spec.name == "ma5_ma20_cross":
        prev_closes = [item.close for item in bars[:index]]
        prev_ma5 = _ma(prev_closes, 5)
        prev_ma20 = _ma(prev_closes, 20)
        ok = bool(prev_ma5 and prev_ma20 and prev_ma5 <= prev_ma20 and ma5 > ma20 and bar.close > previous.close)
    elif spec.name == "donchian_20_breakout":
        previous_high = max(item.high for item in bars[index - 20 : index])
        ok = bar.close > previous_high and amount_ratio >= 1.2
    elif spec.name == "momentum_20_60":
        ret20 = bar.close / bars[index - 20].close - 1
        ret60 = bar.close / bars[index - 60].close - 1
        ok = 0.03 <= ret20 <= 0.45 and ret20 > ret60 / 3 and bar.close > ma20
    if not ok:
        return None
    return {"ma5": ma5, "ma10": ma10, "ma20": ma20, "amount_ratio_5d": amount_ratio}


def _strategy_exit(
    bars: list[DailyBar],
    entry_index: int,
    spec: StrategySpec,
    config: BacktestConfig,
) -> tuple[int, float, str]:
    entry = bars[entry_index]
    stop_price = entry.open * (1 - spec.stop_loss_pct)
    target_price = entry.open * (1 + spec.take_profit_pct)
    last_index = min(len(bars) - 1, entry_index + spec.max_hold_days - 1)
    for index in range(entry_index + 1, last_index + 1):
        bar = bars[index]
        if bar.open <= stop_price:
            return index, bar.open, "stop_gap_open"
        if bar.low <= stop_price:
            return index, stop_price, "stop_loss"
        if bar.high >= target_price:
            return index, target_price, "take_profit"
        closes = [item.close for item in bars[: index + 1]]
        ma20 = _ma(closes, 20)
        ma60 = _ma(closes, 60)
        if ma20 and bar.close < ma20:
            next_index = min(index + 1, len(bars) - 1)
            return next_index, bars[next_index].open, "ma20_break_next_open"
        if spec.name in {"ma20_ma60_trend", "momentum_20_60"} and ma60 and bar.close < ma60:
            next_index = min(index + 1, len(bars) - 1)
            return next_index, bars[next_index].open, "ma60_break_next_open"
    return last_index, bars[last_index].close, "time_exit"


def write_strategy_lab_outputs(
    config: BacktestConfig,
    benchmark: BenchmarkSummary,
    comparisons: list[StrategyComparisonRow],
    picks: list[StockPickRow],
    strategy_trades: dict[str, list[TradeRow]],
    strategy_portfolios: dict[str, list[PortfolioTradeRow]],
) -> None:
    output_dir = Path("reports/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"strategy_lab_{config.end_date}"
    _write_dataclass_csv(output_dir / f"{prefix}_comparison.csv", comparisons)
    _write_dataclass_csv(output_dir / f"{prefix}_stock_picks.csv", picks)
    for strategy, trades in strategy_trades.items():
        _write_dataclass_csv(output_dir / f"{prefix}_{strategy}_trades.csv", trades)
        _write_dataclass_csv(output_dir / f"{prefix}_{strategy}_portfolio.csv", strategy_portfolios[strategy])
    payload = {
        "config": asdict(config),
        "benchmark": asdict(benchmark),
        "strategies": [asdict(item) for item in STRATEGIES],
        "comparison": [asdict(item) for item in comparisons],
        "stock_picks": [asdict(item) for item in picks],
        "data_sufficiency": (
            "research_only: fixed five-stock universe, qfq daily bars, no historical Top100 replay, "
            "no complete limit/suspension execution constraints"
        ),
    }
    (output_dir / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_strategy_lab(
    console: Console,
    config: BacktestConfig,
    benchmark: BenchmarkSummary,
    comparisons: list[StrategyComparisonRow],
    picks: list[StockPickRow],
) -> None:
    table = Table(title=f"Strategy lab {config.start_date} to {config.end_date}")
    table.add_column("Strategy")
    table.add_column("Return", justify="right")
    table.add_column("BuyHold Excess", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Trades", justify="right")
    for row in sorted(comparisons, key=lambda item: item.score, reverse=True):
        table.add_row(
            row.strategy,
            f"{row.total_return:.2%}",
            f"{row.excess_return:.2%}",
            f"{row.max_drawdown:.2%}",
            str(row.trade_count),
        )
    console.print("Data sufficiency: research_only; compare against buy-hold before selecting stocks.")
    console.print(f"Buy-hold baseline: {benchmark.total_return:.2%}, max_dd={benchmark.max_drawdown:.2%}")
    console.print(table)
    pick_table = Table(title="Strategy-ranked stock candidates")
    pick_table.add_column("Code")
    pick_table.add_column("Best Strategy")
    pick_table.add_column("Strategy Return", justify="right")
    pick_table.add_column("BuyHold", justify="right")
    pick_table.add_column("Action")
    for row in picks:
        pick_table.add_row(
            row.code,
            row.best_strategy,
            f"{row.best_total_net_return:.2%}",
            f"{row.buy_hold_return:.2%}",
            row.action,
        )
    console.print(pick_table)
    console.print(f"Reports: reports/generated/strategy_lab_{config.end_date}_*.csv")


def _comparison_row(
    spec: StrategySpec,
    portfolio: PortfolioSummary,
    benchmark: BenchmarkSummary,
) -> StrategyComparisonRow:
    excess = portfolio.total_return - benchmark.total_return
    score = portfolio.total_return - abs(portfolio.max_drawdown) * 0.7 + excess * 0.3
    return StrategyComparisonRow(
        strategy=spec.name,
        final_equity=portfolio.final_equity,
        total_return=portfolio.total_return,
        max_drawdown=portfolio.max_drawdown,
        trade_count=portfolio.trade_count,
        skipped_count=portfolio.skipped_count,
        win_rate=portfolio.win_rate,
        avg_pnl=portfolio.avg_pnl,
        benchmark_return=benchmark.total_return,
        excess_return=round(excess, 6),
        score=round(score, 6),
        note=spec.description,
    )


def _config_for_strategy(base: BacktestConfig, spec: StrategySpec) -> BacktestConfig:
    return BacktestConfig(
        start_date=base.start_date,
        end_date=base.end_date,
        adjust=base.adjust,
        initial_cash=base.initial_cash,
        position_pct=base.position_pct,
        max_positions=base.max_positions,
        max_hold_days=spec.max_hold_days,
        stop_loss_pct=spec.stop_loss_pct,
        take_profit_pct=spec.take_profit_pct,
        commission_rate=base.commission_rate,
        stamp_tax_rate=base.stamp_tax_rate,
        slippage_rate=base.slippage_rate,
        source_note=base.source_note,
    )


def _pick_action(total: float, baseline: float, trade_count: int) -> str:
    if trade_count < 5:
        return "insufficient_samples"
    if total > baseline and total > 0:
        return "strategy_watch"
    if total > 0:
        return "buy_hold_preferred"
    return "avoid_strategy"


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


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(mean(values)), 6)


def _compound(values: list[float]) -> float:
    equity = 1.0
    for value in values:
        equity *= 1 + value
    return equity - 1


def _max_drawdown_from_returns(values: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for value in values:
        equity *= 1 + value
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1)
    return max_dd


def _write_dataclass_csv(path: Path, rows: list) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    run_strategy_lab(end_date="2026-05-20")


if __name__ == "__main__":
    main()
