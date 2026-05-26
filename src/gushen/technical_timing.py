from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from rich.console import Console
from rich.table import Table

from gushen.data import DailyBar
from gushen.deep_analysis import load_or_fetch_histories
from gushen.research import load_or_fetch_daily_snapshot
from gushen.trade_calendar import latest_research_trade_date


@dataclass(frozen=True)
class TechnicalTimingRow:
    trade_date: str
    code: str
    name: str
    amount_rank: int
    close: float
    bars_count: int
    data_sufficiency: str
    trend_state: str
    kline_state: str
    volume_state: str
    ma5: float | None
    ma10: float | None
    ma20: float | None
    ma60: float | None
    ma5_gap: float | None
    ma20_gap: float | None
    support_price: float | None
    resistance_price: float | None
    observation_action: str
    entry_watch: str
    exit_watch: str
    risk_note: str
    missing_data: str


def build_top100_technical_timing(
    trade_date: str | None = None,
    max_close: float = 50.0,
) -> list[TechnicalTimingRow]:
    console = Console()
    trade_date = trade_date or latest_research_trade_date()
    snapshot = load_or_fetch_daily_snapshot(trade_date)
    top100 = sorted(snapshot, key=lambda item: item.amount, reverse=True)[:100]
    amount_rank = {bar.code: index + 1 for index, bar in enumerate(top100)}
    candidates = [bar for bar in top100 if 0 < bar.close < max_close]
    histories = load_or_fetch_histories(candidates, trade_date)
    rows = [
        analyze_technical_timing(
            bar=bar,
            history=histories.get(bar.code, []),
            trade_date=trade_date,
            amount_rank=amount_rank[bar.code],
        )
        for bar in candidates
    ]
    rows = sorted(rows, key=lambda item: (item.observation_action, item.amount_rank))
    write_technical_timing(trade_date, rows)
    print_technical_timing(console, trade_date, rows)
    return rows


def analyze_technical_timing(
    bar: DailyBar,
    history: list[DailyBar],
    trade_date: str,
    amount_rank: int,
) -> TechnicalTimingRow:
    bars = _prepare_bars(bar, history, trade_date)
    closes = [item.close for item in bars]
    lows = [item.low for item in bars]
    highs = [item.high for item in bars]
    amounts = [item.amount for item in bars]
    latest = bars[-1]

    ma5 = _ma(closes, 5)
    ma10 = _ma(closes, 10)
    ma20 = _ma(closes, 20)
    ma60 = _ma(closes, 60)
    ma5_gap = _gap(latest.close, ma5)
    ma20_gap = _gap(latest.close, ma20)
    amount_ratio_5d = _amount_ratio(amounts, 5)

    sufficiency, missing = _data_sufficiency(bars)
    trend_state = _trend_state(latest.close, ma5, ma10, ma20)
    kline_state = _kline_state(bars, ma5, ma20, amount_ratio_5d)
    volume_state = _volume_state(amount_ratio_5d)
    support = _support_price(latest.close, lows, [ma5, ma10, ma20])
    resistance = _resistance_price(latest.close, highs)
    action = _observation_action(sufficiency, trend_state, kline_state, ma5_gap, ma20_gap)
    entry = _entry_watch(action, trend_state, kline_state)
    exit_watch = _exit_watch(support, ma20)
    risk_note = _risk_note(kline_state, volume_state, ma5_gap, ma20_gap)

    return TechnicalTimingRow(
        trade_date=trade_date,
        code=bar.code,
        name=bar.name,
        amount_rank=amount_rank,
        close=latest.close,
        bars_count=len(bars),
        data_sufficiency=sufficiency,
        trend_state=trend_state,
        kline_state=kline_state,
        volume_state=volume_state,
        ma5=_round_or_none(ma5),
        ma10=_round_or_none(ma10),
        ma20=_round_or_none(ma20),
        ma60=_round_or_none(ma60),
        ma5_gap=_round_or_none(ma5_gap),
        ma20_gap=_round_or_none(ma20_gap),
        support_price=_round_or_none(support),
        resistance_price=_round_or_none(resistance),
        observation_action=action,
        entry_watch=entry,
        exit_watch=exit_watch,
        risk_note=risk_note,
        missing_data="; ".join(missing) if missing else "none",
    )


def write_technical_timing(trade_date: str, rows: list[TechnicalTimingRow]) -> None:
    output = Path(f"reports/generated/top100_technical_timing_{trade_date}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(TechnicalTimingRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def print_technical_timing(
    console: Console,
    trade_date: str,
    rows: list[TechnicalTimingRow],
) -> None:
    table = Table(title=f"{trade_date} Top100 close<50 technical timing")
    table.add_column("Rank", justify="right")
    table.add_column("Code")
    table.add_column("Name")
    table.add_column("Close", justify="right")
    table.add_column("Sufficiency")
    table.add_column("Trend")
    table.add_column("K-line")
    table.add_column("Action")
    table.add_column("Entry Watch")
    for row in rows[:20]:
        table.add_row(
            str(row.amount_rank),
            row.code,
            row.name,
            f"{row.close:.2f}",
            row.data_sufficiency,
            row.trend_state,
            row.kline_state,
            row.observation_action,
            row.entry_watch,
        )
    console.print("Data sufficiency is evaluated before any timing language.")
    console.print("All entry/exit fields are paper-observation rules, not live buy/sell advice.")
    console.print(table)
    console.print(f"Rows: {len(rows)}")
    console.print("Report: reports/generated/top100_technical_timing_" + trade_date + ".csv")


def _prepare_bars(bar: DailyBar, history: list[DailyBar], trade_date: str) -> list[DailyBar]:
    rows = sorted(
        [item for item in history if item.trade_date <= trade_date and item.close > 0],
        key=lambda item: item.trade_date,
    )
    if not rows or rows[-1].trade_date != trade_date:
        rows.append(bar)
    return rows


def _data_sufficiency(bars: list[DailyBar]) -> tuple[str, list[str]]:
    missing = []
    if len(bars) < 20:
        missing.append("need at least 20 daily bars for MA20/support analysis")
        return "insufficient", missing
    if len(bars) < 60:
        missing.append("need 60 daily bars for MA60 trend confirmation")
        return "partial", missing
    return "sufficient_for_paper_observation", missing


def _trend_state(close: float, ma5: float | None, ma10: float | None, ma20: float | None) -> str:
    if ma5 is None or ma10 is None or ma20 is None:
        return "insufficient_ma"
    if ma5 > ma10 > ma20 and close >= ma5:
        return "strong_uptrend"
    if ma5 >= ma10 >= ma20 and close >= ma20:
        return "uptrend_pullback"
    if close < ma20:
        return "below_ma20"
    return "mixed"


def _kline_state(
    bars: list[DailyBar],
    ma5: float | None,
    ma20: float | None,
    amount_ratio_5d: float,
) -> str:
    if len(bars) < 2:
        return "insufficient_kline"
    latest = bars[-1]
    previous = bars[-2]
    previous_highs = [item.high for item in bars[-21:-1]]
    previous_20d_high = max(previous_highs) if len(previous_highs) >= 20 else None
    day_range = latest.high - latest.low
    upper_shadow = (latest.high - latest.close) / day_range if day_range > 0 else 0.0
    if previous_20d_high is not None and latest.close > previous_20d_high and amount_ratio_5d >= 1.2:
        return "volume_breakout"
    if upper_shadow >= 0.45 and latest.close < latest.high:
        return "long_upper_shadow"
    if ma5 is not None and previous.close < ma5 <= latest.close:
        return "reclaim_ma5"
    if ma20 is not None and latest.pct_change < 0 and latest.close >= ma20:
        return "pullback_above_ma20"
    return "normal"


def _volume_state(amount_ratio_5d: float) -> str:
    if amount_ratio_5d >= 1.5:
        return "volume_expansion"
    if amount_ratio_5d < 0.8:
        return "volume_contraction"
    return "volume_normal"


def _observation_action(
    sufficiency: str,
    trend_state: str,
    kline_state: str,
    ma5_gap: float | None,
    ma20_gap: float | None,
) -> str:
    if sufficiency == "insufficient":
        return "data_insufficient"
    if trend_state == "below_ma20":
        return "research_only"
    if kline_state == "long_upper_shadow":
        return "observe_pullback"
    if ma20_gap is not None and ma20_gap > 0.18:
        return "observe_pullback"
    if trend_state in {"strong_uptrend", "uptrend_pullback"} and ma5_gap is not None and ma5_gap <= 0.08:
        return "paper_watch"
    return "research_only"


def _entry_watch(action: str, trend_state: str, kline_state: str) -> str:
    if action == "data_insufficient":
        return "No timing call; collect more daily bars first."
    if kline_state == "volume_breakout":
        return "Paper observe only if next session holds above breakout close without limit-up chasing."
    if trend_state == "strong_uptrend":
        return "Paper observe pullback holding MA5, or reclaiming intraday VWAP after early shakeout."
    if trend_state == "uptrend_pullback":
        return "Paper observe rebound near MA10/MA20 with volume returning above recent average."
    return "Wait for price to reclaim MA20 and MA5 to turn up before paper observation."


def _exit_watch(support: float | None, ma20: float | None) -> str:
    triggers = []
    if support is not None:
        triggers.append(f"breaks support {support:.2f}")
    if ma20 is not None:
        triggers.append(f"closes below MA20 {ma20:.2f}")
    if not triggers:
        return "No exit timing call; support/MA data insufficient."
    return "Paper invalidation if " + " or ".join(triggers) + "; force review after 3-5 sessions."


def _risk_note(
    kline_state: str,
    volume_state: str,
    ma5_gap: float | None,
    ma20_gap: float | None,
) -> str:
    risks = []
    if kline_state == "long_upper_shadow":
        risks.append("upper shadow suggests intraday selling pressure")
    if volume_state == "volume_contraction":
        risks.append("volume contraction weakens confirmation")
    if ma5_gap is not None and ma5_gap > 0.10:
        risks.append("price is extended above MA5")
    if ma20_gap is not None and ma20_gap > 0.18:
        risks.append("price is extended above MA20")
    return "; ".join(risks) if risks else "none"


def _support_price(close: float, lows: list[float], averages: list[float | None]) -> float | None:
    candidates = [value for value in averages if value is not None and 0 < value <= close]
    if lows:
        candidates.append(min(lows[-10:]))
    below = [value for value in candidates if value > 0 and value <= close]
    return max(below) if below else None


def _resistance_price(close: float, highs: list[float]) -> float | None:
    if len(highs) < 2:
        return None
    previous_high = max(highs[-21:-1]) if len(highs) >= 21 else max(highs[:-1])
    return previous_high if previous_high > close else None


def _ma(values: list[float], periods: int) -> float | None:
    if len(values) < periods:
        return None
    return mean(values[-periods:])


def _gap(close: float, average: float | None) -> float | None:
    if average in {None, 0}:
        return None
    return close / average - 1


def _amount_ratio(values: list[float], periods: int) -> float:
    if len(values) < periods + 1:
        return 1.0
    avg = mean(values[-periods - 1 : -1])
    return values[-1] / avg if avg else 1.0


def _round_or_none(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def main() -> None:
    build_top100_technical_timing()


if __name__ == "__main__":
    main()
