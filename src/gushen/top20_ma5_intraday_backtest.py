from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from gushen.top20_ma5_pullback_strategy import (
    DEFAULT_CACHE_DIR,
    CandidateRow,
    StrategyConfig,
    TradeRow,
    build_market_frame,
    infer_latest_cache_date,
    load_daily_frames,
    net_trade_return,
    round_float,
    select_candidates,
    simulate_portfolio,
    write_outputs,
)
from gushen.trade_calendar import latest_research_trade_date


DEFAULT_OUTPUT_DIR = Path("reports/generated/top20_ma5_intraday_last_month")
DEFAULT_INTRADAY_CACHE_DIR = Path("data/local/intraday/sina_5m")
SINA_URL = "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest Top20 MA5 pullback strategy with 5-minute intraday bars."
    )
    parser.add_argument("--daily-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--intraday-cache-dir", type=Path, default=DEFAULT_INTRADAY_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-positions", type=int, default=1)
    parser.add_argument("--position-pct", type=float, default=0.20)
    parser.add_argument("--pullback-drop-pct", type=float, default=0.8)
    parser.add_argument("--sina-datalen", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end_date = args.end_date or infer_latest_cache_date(args.daily_cache_dir) or latest_research_trade_date()
    start_date = args.start_date or pd.Timestamp(end_date).date().replace(day=1).isoformat()
    config = StrategyConfig(
        start_date=start_date,
        end_date=end_date,
        top_n=args.top_n,
        wait_days=1,
        max_hold_days=1,
        max_positions=args.max_positions,
        position_pct=args.position_pct,
        slippage_rate=0.0,
        source_note=(
            "5-minute intraday proxy: T close selects candidates; T+1 only watches fixed T MA5; "
            "T+2 exits on 0.8% pullback from 09:30-10:00 high or at 10:00."
        ),
    )
    result = run_intraday_backtest(
        daily_cache_dir=args.daily_cache_dir,
        intraday_cache_dir=args.intraday_cache_dir,
        output_dir=args.output_dir,
        config=config,
        pullback_drop_pct=args.pullback_drop_pct,
        sina_datalen=args.sina_datalen,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_intraday_backtest(
    daily_cache_dir: Path,
    intraday_cache_dir: Path,
    output_dir: Path,
    config: StrategyConfig,
    pullback_drop_pct: float = 0.8,
    sina_datalen: int = 5000,
) -> dict[str, Any]:
    frames = load_daily_frames(daily_cache_dir, config)
    if not frames:
        raise RuntimeError(f"no daily bar CSV files found in {daily_cache_dir}")
    market = build_market_frame(frames, config)
    candidates = select_candidates(market, config)
    trades, diagnostics = build_intraday_trades(
        frames=frames,
        candidates=candidates,
        intraday_cache_dir=intraday_cache_dir,
        config=config,
        pullback_drop_pct=pullback_drop_pct,
        sina_datalen=sina_datalen,
    )
    ledger, summary = simulate_portfolio(trades, len(candidates), config)
    write_outputs(output_dir, config, candidates, trades, ledger, summary)
    (output_dir / "intraday_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "summary": asdict(summary),
        "diagnostics": diagnostics,
        "output_dir": str(output_dir),
        "candidates_path": str(output_dir / "top20_ma5_candidates.csv"),
        "trades_path": str(output_dir / "top20_ma5_trades.csv"),
        "portfolio_path": str(output_dir / "top20_ma5_portfolio.csv"),
        "summary_path": str(output_dir / "top20_ma5_summary.json"),
        "diagnostics_path": str(output_dir / "intraday_diagnostics.json"),
    }


def build_intraday_trades(
    frames: dict[str, pd.DataFrame],
    candidates: list[CandidateRow],
    intraday_cache_dir: Path,
    config: StrategyConfig,
    pullback_drop_pct: float,
    sina_datalen: int,
) -> tuple[list[TradeRow], dict[str, Any]]:
    trades: list[TradeRow] = []
    missing_entry = 0
    missing_exit = 0
    no_touch = 0
    no_exit = 0
    touched = 0
    for candidate in candidates:
        frame = frames.get(candidate.code)
        if frame is None:
            continue
        matches = frame.index[frame["trade_date"] == candidate.signal_date].tolist()
        if not matches:
            continue
        signal_index = matches[0]
        if signal_index + 2 >= len(frame):
            continue
        entry_date = str(frame.iloc[signal_index + 1]["trade_date"])
        exit_date = str(frame.iloc[signal_index + 2]["trade_date"])
        entry_bars = load_or_fetch_sina_5m(
            candidate.code,
            candidate.name,
            intraday_cache_dir,
            sina_datalen=sina_datalen,
        )
        entry_day = bars_for_date(entry_bars, entry_date)
        if entry_day.empty:
            missing_entry += 1
            continue
        entry = find_fixed_ma_touch(entry_day, candidate.ma5)
        if entry is None:
            no_touch += 1
            continue
        touched += 1
        exit_day = bars_for_date(entry_bars, exit_date)
        if exit_day.empty:
            missing_exit += 1
            continue
        exit_result = morning_pullback_exit(exit_day, pullback_drop_pct)
        if exit_result is None:
            no_exit += 1
            continue
        exit_price, exit_reason, exit_high = exit_result
        entry_price = float(candidate.ma5)
        gross_return = exit_price / entry_price - 1.0
        net_return = net_trade_return(entry_price, exit_price, config)
        trades.append(
            TradeRow(
                signal_date=candidate.signal_date,
                entry_date=entry_date,
                exit_date=exit_date,
                code=candidate.code,
                name=candidate.name,
                amount_rank=candidate.amount_rank,
                entry_price=round_float(entry_price),
                exit_price=round_float(exit_price),
                gross_return_pct=round_float(gross_return * 100),
                net_return_pct=round_float(net_return * 100),
                exit_reason=exit_reason,
                wait_days=1,
                entry_ma5=round_float(candidate.ma5),
                next_day_high=round_float(exit_high),
                next_day_close=round_float(exit_price),
                note=f"entry_time={entry['time']}; exit_rule={exit_reason}; source=sina_5m",
            )
        )
    diagnostics = {
        "intraday_source": "sina_5m",
        "pullback_drop_pct": pullback_drop_pct,
        "sina_datalen": sina_datalen,
        "candidates": len(candidates),
        "ma5_touched": touched,
        "no_touch": no_touch,
        "missing_entry_intraday": missing_entry,
        "missing_exit_intraday": missing_exit,
        "no_exit_price": no_exit,
        "trade_signals": len(trades),
        "note": "Sina scale=5 bars are used because public 1-minute sources only expose recent days.",
    }
    return trades, diagnostics


def load_or_fetch_sina_5m(
    code: str,
    name: str,
    cache_dir: Path,
    sina_datalen: int,
) -> pd.DataFrame:
    folder = intraday_cache_folder(cache_dir, code, name)
    cached = sorted(folder.glob("*.csv")) if folder.exists() else []
    if cached:
        return pd.concat([read_intraday_csv(path) for path in cached], ignore_index=True)
    frame = fetch_sina_5m(code, sina_datalen=sina_datalen)
    write_intraday_by_date(cache_dir, folder, code, name, frame)
    return frame


def fetch_sina_5m(code: str, sina_datalen: int) -> pd.DataFrame:
    symbol = sina_symbol(code)
    response = requests.get(
        SINA_URL,
        params={"symbol": symbol, "scale": 5, "ma": "no", "datalen": sina_datalen},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Sina returned non-list payload for {code}: {str(data)[:120]}")
    frame = pd.DataFrame(data)
    if frame.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    frame = frame.rename(columns={"day": "time"})
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["time"] = pd.to_datetime(frame["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return frame[["time", "open", "high", "low", "close", "volume"]].dropna()


def read_intraday_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["time"] = pd.to_datetime(frame["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["open", "high", "low", "close"])


def bars_for_date(frame: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    bars = frame[frame["time"].str.startswith(trade_date)].copy()
    return bars.sort_values("time").reset_index(drop=True)


def find_fixed_ma_touch(day_bars: pd.DataFrame, ma5: float) -> dict[str, Any] | None:
    if not math.isfinite(ma5):
        return None
    for row in day_bars.to_dict("records"):
        if float(row["low"]) <= ma5 <= float(row["high"]):
            return row
    return None


def morning_pullback_exit(
    day_bars: pd.DataFrame,
    pullback_drop_pct: float,
) -> tuple[float, str, float] | None:
    morning = day_bars[(day_bars["time"].str[11:16] >= "09:30") & (day_bars["time"].str[11:16] <= "10:00")]
    if morning.empty:
        return None
    high_so_far = -math.inf
    threshold_factor = 1.0 - pullback_drop_pct / 100.0
    for row in morning.to_dict("records"):
        high_so_far = max(high_so_far, float(row["high"]))
        threshold = high_so_far * threshold_factor
        if float(row["low"]) <= threshold:
            return threshold, "morning_high_pullback_0.8pct", high_so_far
    last = morning.iloc[-1]
    return float(last["close"]), "morning_1000_exit", max(float(item) for item in morning["high"])


def write_intraday_by_date(
    cache_dir: Path,
    folder: Path,
    code: str,
    name: str,
    frame: pd.DataFrame,
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        append_manifest(cache_dir, folder, code, name, frame, None)
        return
    for trade_date, group in frame.groupby(frame["time"].str[:10]):
        path = folder / f"{trade_date}.csv"
        group.sort_values("time").to_csv(path, index=False)
        append_manifest(cache_dir, path, code, name, group, trade_date)


def intraday_cache_folder(cache_dir: Path, code: str, name: str) -> Path:
    folder = f"{code}-{safe_name(name)}"
    return cache_dir / folder


def append_manifest(
    cache_dir: Path,
    path: Path,
    code: str,
    name: str,
    frame: pd.DataFrame,
    trade_date: str | None,
) -> None:
    manifest = cache_dir / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "sina_5m",
        "code": code,
        "name": name,
        "trade_date": trade_date,
        "path": str(path),
        "rows": int(len(frame)),
        "first_time": str(frame["time"].iloc[0]) if not frame.empty else None,
        "last_time": str(frame["time"].iloc[-1]) if not frame.empty else None,
    }
    with manifest.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def sina_symbol(code: str) -> str:
    raw, market = code.split(".")
    return ("sh" if market == "SH" else "sz") + raw


def safe_name(value: str) -> str:
    name = str(value).strip() or "UNKNOWN"
    return re.sub(r'[\\/:*?"<>|\\s]+', "_", name)


if __name__ == "__main__":
    main()
