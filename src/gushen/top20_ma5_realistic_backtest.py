from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from gushen.top20_ma5_intraday_backtest import (
    DEFAULT_INTRADAY_CACHE_DIR,
    bars_for_date,
    fetch_sina_5m,
    intraday_cache_folder,
    read_intraday_csv,
    write_intraday_by_date,
)
from gushen.top20_ma5_pullback_strategy import (
    DEFAULT_CACHE_DIR,
    StrategyConfig,
    build_market_frame,
    choose_latest_paths_by_code,
    is_excluded_board,
    is_st_name,
    load_daily_frames,
    normalize_ts_code,
    round_float,
)

DEFAULT_OUTPUT_DIR = Path("reports/generated/top20_ma5_realistic_20260601_20260701")
COMMISSION_RATE = 0.00025
PER_POSITION = 10_000.0
PRICE_CAP = 110.0
OBSERVATION_DAYS = 3
PULLBACK_DROP_PCT = 0.8


@dataclass(frozen=True)
class RealisticTrade:
    execution_date: str
    exit_date: str
    code: str
    name: str
    best_rank: int
    source: str
    source_ranks: str
    boundary_price: float
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    exit_reason: str
    shares: int
    invested: float
    gross_pnl: float
    net_pnl: float
    gross_return_pct: float
    net_return_pct: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realistic Top20 MA5 morning execution backtest.")
    parser.add_argument("--daily-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--intraday-cache-dir", type=Path, default=DEFAULT_INTRADAY_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default="2026-06-01")
    parser.add_argument("--end-date", default="2026-07-01")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--per-position", type=float, default=PER_POSITION)
    parser.add_argument("--price-cap", type=float, default=PRICE_CAP)
    parser.add_argument("--commission-rate", type=float, default=COMMISSION_RATE)
    parser.add_argument("--sina-datalen", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_backtest(
        daily_cache_dir=args.daily_cache_dir,
        intraday_cache_dir=args.intraday_cache_dir,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=args.top_n,
        per_position=args.per_position,
        price_cap=args.price_cap,
        commission_rate=args.commission_rate,
        sina_datalen=args.sina_datalen,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_backtest(
    daily_cache_dir: Path,
    intraday_cache_dir: Path,
    output_dir: Path,
    start_date: str,
    end_date: str,
    top_n: int,
    per_position: float,
    price_cap: float,
    commission_rate: float,
    sina_datalen: int,
) -> dict[str, Any]:
    config = StrategyConfig(
        start_date=(date.fromisoformat(start_date) - timedelta(days=45)).isoformat(),
        end_date=end_date,
        top_n=top_n,
        wait_days=OBSERVATION_DAYS,
        max_positions=999999,
        position_pct=1.0,
        commission_rate=commission_rate,
        slippage_rate=0.0,
        source_note="Top20 MA5 realistic morning execution; 5-minute intraday proxy.",
    )
    frames = load_daily_frames(daily_cache_dir, config)
    if not frames:
        raise RuntimeError(f"no daily data under {daily_cache_dir}")
    market = build_market_frame(frames, config)
    candidates = select_signal_candidates(market, top_n)
    trade_dates = sorted({str(item) for item in market["trade_date"].unique()})
    execution_dates = [item for item in trade_dates if start_date <= item <= end_date]
    trades: list[RealisticTrade] = []
    diagnostics: dict[str, int] = {
        "execution_days": len(execution_dates),
        "observation_candidates": 0,
        "skipped_price_cap": 0,
        "skipped_no_lot": 0,
        "missing_entry_intraday": 0,
        "no_entry_touch": 0,
        "missing_exit_intraday": 0,
        "no_exit": 0,
    }
    for execution_date in execution_dates:
        pool = build_observation_pool(candidates, frames, trade_dates, execution_date, OBSERVATION_DAYS)
        diagnostics["observation_candidates"] += len(pool)
        for item in pool:
            boundary = dynamic_boundary_for_date(frames[item["code"]], execution_date)
            if boundary is None:
                continue
            if boundary > price_cap:
                diagnostics["skipped_price_cap"] += 1
                continue
            shares = recommend_lot(boundary, per_position)
            if shares <= 0:
                diagnostics["skipped_no_lot"] += 1
                continue
            entry_bars = load_or_fetch_intraday_for_date(
                item["code"], item["name"], execution_date, intraday_cache_dir, sina_datalen
            )
            if entry_bars.empty:
                diagnostics["missing_entry_intraday"] += 1
                continue
            entry = morning_entry(entry_bars, boundary)
            if entry is None:
                diagnostics["no_entry_touch"] += 1
                continue
            exit_date = next_trade_date(trade_dates, execution_date)
            if exit_date is None:
                exit_date = next_weekday(execution_date)
            exit_bars = load_or_fetch_intraday_for_date(
                item["code"], item["name"], exit_date, intraday_cache_dir, sina_datalen
            )
            if exit_bars.empty:
                diagnostics["missing_exit_intraday"] += 1
                continue
            exit_result = morning_pullback_exit(exit_bars, PULLBACK_DROP_PCT)
            if exit_result is None:
                diagnostics["no_exit"] += 1
                continue
            entry_price = float(entry["price"])
            exit_price = float(exit_result["price"])
            invested = shares * entry_price
            gross_pnl = shares * (exit_price - entry_price)
            sell_value = shares * exit_price
            net_pnl = gross_pnl - invested * commission_rate - sell_value * commission_rate
            trades.append(
                RealisticTrade(
                    execution_date=execution_date,
                    exit_date=exit_date,
                    code=item["code"],
                    name=item["name"],
                    best_rank=item["best_rank"],
                    source=item["source"],
                    source_ranks=item["source_ranks"],
                    boundary_price=round_float(boundary, 4),
                    entry_time=str(entry["time"]),
                    entry_price=round_float(entry_price, 4),
                    exit_time=str(exit_result["time"]),
                    exit_price=round_float(exit_price, 4),
                    exit_reason=str(exit_result["reason"]),
                    shares=shares,
                    invested=round_float(invested, 2),
                    gross_pnl=round_float(gross_pnl, 2),
                    net_pnl=round_float(net_pnl, 2),
                    gross_return_pct=round_float(gross_pnl / invested * 100 if invested else 0, 4),
                    net_return_pct=round_float(net_pnl / invested * 100 if invested else 0, 4),
                )
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_df = pd.DataFrame([asdict(item) for item in trades])
    trades_path = output_dir / "trades.csv"
    trades_df.to_csv(trades_path, index=False)
    daily_df = daily_summary(trades_df)
    daily_path = output_dir / "daily_summary.csv"
    daily_df.to_csv(daily_path, index=False)
    capital_df, capital_summary = capital_flow_summary(trades_df, commission_rate)
    capital_path = output_dir / "capital_flow.csv"
    capital_df.to_csv(capital_path, index=False)
    summary = make_summary(
        start_date,
        end_date,
        top_n,
        trades_df,
        diagnostics,
        per_position,
        price_cap,
        commission_rate,
        capital_summary,
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path = output_dir / "report.html"
    html_path.write_text(render_html(summary, daily_df, trades_df), encoding="utf-8")
    return {
        "summary": summary,
        "output_dir": str(output_dir),
        "html_path": str(html_path),
        "trades_path": str(trades_path),
        "daily_summary_path": str(daily_path),
        "capital_flow_path": str(capital_path),
    }


def select_signal_candidates(market: pd.DataFrame, top_n: int) -> pd.DataFrame:
    selected = market[(market["amount_rank"] <= top_n) & (market["close"] > market["ma5"])].copy()
    selected = selected.sort_values(["trade_date", "amount_rank", "code"])
    return selected[["trade_date", "code", "name", "amount_rank", "close", "ma5", "amount"]]


def build_observation_pool(
    candidates: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    trade_dates: list[str],
    execution_date: str,
    observation_days: int,
) -> list[dict[str, Any]]:
    previous_dates = previous_trade_dates(trade_dates, execution_date, observation_days)
    if not previous_dates:
        return []
    selected = candidates[candidates["trade_date"].astype(str).isin(previous_dates)].copy()
    rows: list[dict[str, Any]] = []
    for code, group in selected.groupby("code"):
        code = str(code)
        frame = frames.get(code)
        if frame is None:
            continue
        valid = []
        for record in group.sort_values(["trade_date", "amount_rank"]).to_dict("records"):
            signal_date = str(record["trade_date"])
            if signal_survives(frame, signal_date, execution_date):
                valid.append(record)
        if not valid:
            continue
        best_rank = min(int(item["amount_rank"]) for item in valid)
        rows.append(
            {
                "code": code,
                "name": str(valid[0].get("name") or ""),
                "best_rank": best_rank,
                "source": "/".join(f"第{previous_dates.index(str(item['trade_date'])) + 1}" for item in valid),
                "source_ranks": ",".join(str(int(item["amount_rank"])) for item in valid),
                "signal_dates": ",".join(str(item["trade_date"]) for item in valid),
            }
        )
    return sorted(rows, key=lambda item: (item["best_rank"], item["code"]))


def previous_trade_dates(trade_dates: list[str], execution_date: str, count: int) -> list[str]:
    previous = [item for item in trade_dates if item < execution_date]
    return list(reversed(previous[-count:]))


def signal_survives(frame: pd.DataFrame, signal_date: str, execution_date: str) -> bool:
    data = frame.copy()
    if "ma5" not in data.columns:
        data["ma5"] = pd.to_numeric(data["close"], errors="coerce").rolling(5).mean()
    observation = data[(data["trade_date"] > signal_date) & (data["trade_date"] < execution_date)].copy()
    if observation.empty:
        return True
    return not bool((pd.to_numeric(observation["close"], errors="coerce") < pd.to_numeric(observation["ma5"], errors="coerce")).any())


def dynamic_boundary_for_date(frame: pd.DataFrame, execution_date: str) -> float | None:
    previous = frame[frame["trade_date"] < execution_date].tail(4)
    closes = pd.to_numeric(previous["close"], errors="coerce").dropna()
    if len(closes) != 4:
        return None
    value = float(closes.mean())
    return value if math.isfinite(value) else None


def recommend_lot(price: float, per_position: float) -> int:
    if price <= 0 or not math.isfinite(price):
        return 0
    return int(per_position // (price * 100)) * 100


def load_or_fetch_intraday_for_date(
    code: str,
    name: str,
    trade_date: str,
    cache_dir: Path,
    sina_datalen: int,
) -> pd.DataFrame:
    folder = intraday_cache_folder(cache_dir, code, name)
    path = folder / f"{trade_date}.csv"
    if path.exists():
        return read_intraday_csv(path)
    if folder.exists():
        matches = sorted(folder.glob("*.csv"))
        if matches:
            frame = pd.concat([read_intraday_csv(item) for item in matches], ignore_index=True)
            day = bars_for_date(frame, trade_date)
            if not day.empty:
                return day
    try:
        frame = fetch_sina_5m(code, sina_datalen=sina_datalen)
        write_intraday_by_date(cache_dir, folder, code, name, frame)
        return bars_for_date(frame, trade_date)
    except Exception:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])


def morning_entry(day_bars: pd.DataFrame, boundary: float) -> dict[str, Any] | None:
    morning = day_bars[(day_bars["time"].str[11:16] >= "09:30") & (day_bars["time"].str[11:16] <= "10:00")]
    for row in morning.sort_values("time").to_dict("records"):
        open_price = float(row["open"])
        low = float(row["low"])
        if open_price <= boundary:
            return {"time": row["time"], "price": open_price, "reason": "open_lte_boundary"}
        if low <= boundary:
            return {"time": row["time"], "price": boundary, "reason": "touch_boundary"}
    return None


def morning_pullback_exit(day_bars: pd.DataFrame, pullback_drop_pct: float) -> dict[str, Any] | None:
    morning = day_bars[(day_bars["time"].str[11:16] >= "09:30") & (day_bars["time"].str[11:16] <= "10:00")]
    if morning.empty:
        return None
    high_so_far = -math.inf
    threshold_factor = 1.0 - pullback_drop_pct / 100.0
    for row in morning.sort_values("time").to_dict("records"):
        high_so_far = max(high_so_far, float(row["high"]))
        threshold = high_so_far * threshold_factor
        if float(row["low"]) <= threshold:
            return {
                "time": row["time"],
                "price": threshold,
                "reason": "morning_high_pullback_0.8pct",
                "high": high_so_far,
            }
    last = morning.sort_values("time").iloc[-1]
    return {
        "time": str(last["time"]),
        "price": float(last["close"]),
        "reason": "morning_1000_exit",
        "high": float(morning["high"].max()),
    }


def next_trade_date(trade_dates: list[str], execution_date: str) -> str | None:
    later = [item for item in trade_dates if item > execution_date]
    return later[0] if later else None


def next_weekday(day: str) -> str:
    current = date.fromisoformat(day) + timedelta(days=1)
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return current.isoformat()


def daily_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["execution_date", "trades", "wins", "win_rate_pct", "invested", "gross_pnl", "net_pnl", "net_return_on_invested_pct"])
    grouped = trades.groupby("execution_date", as_index=False).agg(
        trades=("code", "count"),
        wins=("net_pnl", lambda s: int((s > 0).sum())),
        invested=("invested", "sum"),
        gross_pnl=("gross_pnl", "sum"),
        net_pnl=("net_pnl", "sum"),
    )
    grouped["win_rate_pct"] = grouped["wins"] / grouped["trades"] * 100
    grouped["net_return_on_invested_pct"] = grouped["net_pnl"] / grouped["invested"] * 100
    return grouped[["execution_date", "trades", "wins", "win_rate_pct", "invested", "gross_pnl", "net_pnl", "net_return_on_invested_pct"]].round(4)


def capital_flow_summary(trades: pd.DataFrame, commission_rate: float) -> tuple[pd.DataFrame, dict[str, float]]:
    columns = [
        "date",
        "sell_value",
        "buy_amount",
        "needed_topup_sell_first",
        "capital_required_sell_first",
        "cash_after_sell_first",
        "needed_topup_buy_first",
        "capital_required_buy_first",
        "cash_after_buy_first",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns), {
            "rolling_capital_required": 0.0,
            "conservative_capital_required": 0.0,
            "max_daily_buy_amount": 0.0,
        }
    buys = trades.groupby("execution_date", as_index=False)["invested"].sum().rename(
        columns={"execution_date": "date", "invested": "buy_amount"}
    )
    sell_values = trades.assign(sell_value=trades["shares"] * trades["exit_price"] * (1.0 - commission_rate))
    sells = sell_values.groupby("exit_date", as_index=False)["sell_value"].sum().rename(columns={"exit_date": "date"})
    dates = sorted(set(buys["date"]).union(set(sells["date"])))
    flow = pd.DataFrame({"date": dates}).merge(buys, on="date", how="left").merge(sells, on="date", how="left")
    flow[["buy_amount", "sell_value"]] = flow[["buy_amount", "sell_value"]].fillna(0.0)

    cash = 0.0
    sell_first_required = 0.0
    sell_first_rows = []
    for row in flow.itertuples(index=False):
        cash += float(row.sell_value)
        need = max(0.0, float(row.buy_amount) - cash)
        if need:
            sell_first_required += need
            cash += need
        cash -= float(row.buy_amount)
        sell_first_rows.append((need, sell_first_required, cash))

    cash = 0.0
    buy_first_required = 0.0
    buy_first_rows = []
    for row in flow.itertuples(index=False):
        need = max(0.0, float(row.buy_amount) - cash)
        if need:
            buy_first_required += need
            cash += need
        cash -= float(row.buy_amount)
        cash += float(row.sell_value)
        buy_first_rows.append((need, buy_first_required, cash))

    flow["needed_topup_sell_first"] = [item[0] for item in sell_first_rows]
    flow["capital_required_sell_first"] = [item[1] for item in sell_first_rows]
    flow["cash_after_sell_first"] = [item[2] for item in sell_first_rows]
    flow["needed_topup_buy_first"] = [item[0] for item in buy_first_rows]
    flow["capital_required_buy_first"] = [item[1] for item in buy_first_rows]
    flow["cash_after_buy_first"] = [item[2] for item in buy_first_rows]
    summary = {
        "rolling_capital_required": round_float(sell_first_required, 2),
        "conservative_capital_required": round_float(buy_first_required, 2),
        "max_daily_buy_amount": round_float(float(flow["buy_amount"].max()), 2),
    }
    return flow[columns].round(2), summary


def make_summary(
    start_date: str,
    end_date: str,
    top_n: int,
    trades: pd.DataFrame,
    diagnostics: dict[str, int],
    per_position: float,
    price_cap: float,
    commission_rate: float,
    capital_summary: dict[str, float],
) -> dict[str, Any]:
    if trades.empty:
        base = {"trades": 0, "wins": 0, "win_rate_pct": None, "total_invested": 0, "gross_pnl": 0, "net_pnl": 0, "net_return_on_invested_pct": None}
    else:
        total_invested = float(trades["invested"].sum())
        net_pnl = float(trades["net_pnl"].sum())
        base = {
            "trades": int(len(trades)),
            "wins": int((trades["net_pnl"] > 0).sum()),
            "win_rate_pct": round_float((trades["net_pnl"] > 0).mean() * 100, 4),
            "total_invested": round_float(total_invested, 2),
            "gross_pnl": round_float(float(trades["gross_pnl"].sum()), 2),
            "net_pnl": round_float(net_pnl, 2),
            "net_return_on_invested_pct": round_float(net_pnl / total_invested * 100 if total_invested else 0, 4),
            "avg_net_return_pct": round_float(float(trades["net_return_pct"].mean()), 4),
            "median_net_return_pct": round_float(float(trades["net_return_pct"].median()), 4),
            "max_single_trade_pnl": round_float(float(trades["net_pnl"].max()), 2),
            "min_single_trade_pnl": round_float(float(trades["net_pnl"].min()), 2),
        }
    base.update(
        {
            "start_date": start_date,
            "end_date": end_date,
            "top_n": top_n,
            "observation_days": OBSERVATION_DAYS,
            "per_position": per_position,
            "price_cap": price_cap,
            "commission_rate": commission_rate,
            "rolling_capital_required": capital_summary.get("rolling_capital_required", 0.0),
            "conservative_capital_required": capital_summary.get("conservative_capital_required", 0.0),
            "max_daily_buy_amount": capital_summary.get("max_daily_buy_amount", 0.0),
            "entry_window": "09:30-10:00",
            "entry_rule": "5m K: open<=boundary uses open; otherwise low<=boundary uses boundary",
            "exit_rule": "next trading day 09:30-10:00: sell on 0.8% pullback from morning high, otherwise 10:00 close",
            "diagnostics": diagnostics,
            "data_note": "Historical entries/exits use cached Sina 5-minute bars, so exact fills are a 5-minute proxy. Live page uses fresher quote/1-minute data when available.",
        }
    )
    return base


def render_html(summary: dict[str, Any], daily: pd.DataFrame, trades: pd.DataFrame) -> str:
    def money(value: Any) -> str:
        try:
            return f"{float(value):,.2f}"
        except Exception:
            return "-"

    def pct(value: Any) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "-"
        return f"{float(value):.2f}%"

    cards = [
        ("交易笔数", str(summary.get("trades", 0))),
        ("胜率", pct(summary.get("win_rate_pct"))),
        ("累计投入流水", money(summary.get("total_invested"))),
        ("滚动资金需求", money(summary.get("rolling_capital_required"))),
        ("保守资金需求", money(summary.get("conservative_capital_required"))),
        ("毛盈亏", money(summary.get("gross_pnl"))),
        ("扣佣后盈亏", money(summary.get("net_pnl"))),
        ("投入收益率", pct(summary.get("net_return_on_invested_pct"))),
    ]
    daily_rows = "".join(
        f"<tr><td>{e(row.execution_date)}</td><td>{int(row.trades)}</td><td>{int(row.wins)}</td><td>{pct(row.win_rate_pct)}</td><td>{money(row.invested)}</td><td>{money(row.gross_pnl)}</td><td>{money(row.net_pnl)}</td><td>{pct(row.net_return_on_invested_pct)}</td></tr>"
        for row in daily.itertuples(index=False)
    )
    trade_rows = "".join(
        "<tr>"
        f"<td>{e(row.execution_date)}</td><td>{e(row.exit_date)}</td><td>{e(row.code)}</td><td>{e(row.name)}</td>"
        f"<td>{int(row.best_rank)}</td><td>{e(row.source)}</td><td>{e(row.source_ranks)}</td>"
        f"<td>{money(row.boundary_price)}</td><td>{e(row.entry_time)}</td><td>{money(row.entry_price)}</td>"
        f"<td>{e(row.exit_time)}</td><td>{money(row.exit_price)}</td><td>{e(row.exit_reason)}</td>"
        f"<td>{int(row.shares)}</td><td>{money(row.invested)}</td><td class='{cls(row.net_pnl)}'>{money(row.net_pnl)}</td><td class='{cls(row.net_return_pct)}'>{pct(row.net_return_pct)}</td>"
        "</tr>"
        for row in trades.itertuples(index=False)
    )
    diagnostics = summary.get("diagnostics", {})
    diag_text = "；".join(f"{e(k)}={e(v)}" for k, v in diagnostics.items())
    card_html = "".join(f"<div class='card'><div>{e(label)}</div><strong>{e(value)}</strong></div>" for label, value in cards)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Top20 MA5 真实走势回测 2026-06-01 至 2026-07-01</title>
<style>
body {{ font-family: Arial, 'Microsoft YaHei', sans-serif; margin: 0; color: #17202a; background: #f6f7f9; }}
header {{ padding: 22px 28px; background: #ffffff; border-bottom: 1px solid #d9dee7; }}
h1 {{ margin: 0 0 8px; font-size: 22px; }}
.meta {{ color: #5d6878; font-size: 13px; line-height: 1.7; }}
main {{ padding: 20px 28px 40px; }}
.cards {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 10px; margin-bottom: 18px; }}
.card {{ background: #fff; border: 1px solid #dfe4ea; border-radius: 6px; padding: 12px; }}
.card div {{ color: #647084; font-size: 12px; margin-bottom: 8px; }}
.card strong {{ font-size: 20px; }}
section {{ margin-top: 22px; }}
h2 {{ font-size: 17px; margin: 0 0 10px; }}
.table-wrap {{ overflow: auto; background: #fff; border: 1px solid #dfe4ea; border-radius: 6px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid #edf0f4; white-space: nowrap; text-align: right; }}
th {{ position: sticky; top: 0; background: #f1f4f8; color: #344052; z-index: 1; }}
th:first-child, td:first-child, td:nth-child(3), td:nth-child(4) {{ text-align: left; }}
.pos {{ color: #16803c; }} .neg {{ color: #c92a2a; }}
.note {{ background: #fff; border: 1px solid #dfe4ea; border-radius: 6px; padding: 12px; color: #536174; font-size: 13px; line-height: 1.7; }}
@media (max-width: 900px) {{ .cards {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }} }}
</style>
</head>
<body>
<header>
  <h1>Top20 MA5 真实走势回测</h1>
  <div class="meta">执行日期：{e(summary['start_date'])} 至 {e(summary['end_date'])}；Top{summary['top_n']}；观察期 {summary['observation_days']} 个交易日；每票约 {money(summary['per_position'])}；挂单价 <= {money(summary['price_cap'])}；佣金双边各 {summary['commission_rate']:.4%}</div>
</header>
<main>
  <div class="cards">{card_html}</div>
  <div class="note">入场：{e(summary['entry_rule'])}。出场：{e(summary['exit_rule'])}。滚动资金需求按“当日先卖出回款、再买入新触发”计算；保守资金需求按“当日先买入、再卖出回款”计算。{e(summary['data_note'])}<br>诊断：{diag_text}</div>
  <section><h2>按执行日汇总</h2><div class="table-wrap"><table><thead><tr><th>执行日期</th><th>交易数</th><th>盈利数</th><th>胜率</th><th>投入</th><th>毛盈亏</th><th>扣佣后盈亏</th><th>投入收益率</th></tr></thead><tbody>{daily_rows}</tbody></table></div></section>
  <section><h2>交易明细</h2><div class="table-wrap"><table><thead><tr><th>执行日</th><th>卖出日</th><th>代码</th><th>名称</th><th>最佳排名</th><th>来源</th><th>来源排名</th><th>挂单价</th><th>理论买入时间</th><th>理论成交价</th><th>卖出时间</th><th>卖出价</th><th>卖出原因</th><th>股数</th><th>投入</th><th>扣佣后盈亏</th><th>收益率</th></tr></thead><tbody>{trade_rows}</tbody></table></div></section>
</main>
</body>
</html>"""


def e(value: Any) -> str:
    return html.escape(str(value))


def cls(value: Any) -> str:
    try:
        return "pos" if float(value) >= 0 else "neg"
    except Exception:
        return ""


if __name__ == "__main__":
    main()
