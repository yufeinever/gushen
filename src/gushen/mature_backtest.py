from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover


def sma(values, window: int):
    return pd.Series(values).rolling(window).mean().to_numpy()


class SmaVolumeTrendStrategy(Strategy):
    fast_window = 10
    slow_window = 30
    volume_window = 20
    volume_ratio = 1.2
    stop_loss_pct = 0.08
    take_profit_pct = 0.18

    def init(self) -> None:
        close = self.data.Close
        volume = self.data.Volume
        self.fast_ma = self.I(sma, close, self.fast_window, name=f"SMA{self.fast_window}")
        self.slow_ma = self.I(sma, close, self.slow_window, name=f"SMA{self.slow_window}")
        self.volume_ma = self.I(sma, volume, self.volume_window, name=f"VolumeSMA{self.volume_window}")

    def next(self) -> None:
        if any(pd.isna(value) for value in (self.fast_ma[-1], self.slow_ma[-1], self.volume_ma[-1])):
            return

        price = self.data.Close[-1]
        volume_confirmed = self.data.Volume[-1] >= self.volume_ma[-1] * self.volume_ratio
        if not self.position:
            if crossover(self.fast_ma, self.slow_ma) and volume_confirmed:
                self.buy(
                    sl=price * (1 - self.stop_loss_pct),
                    tp=price * (1 + self.take_profit_pct),
                )
            return

        if crossover(self.slow_ma, self.fast_ma):
            self.position.close()


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"trade_date", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"daily bar file missing columns: {missing}")

    data = frame.rename(
        columns={
            "trade_date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.set_index("Date").sort_index()
    return data.astype(float)


def build_anchor_window_low_hold_benchmark(
    data: pd.DataFrame,
    cash: float = 100_000.0,
    commission: float = 0.0008,
    end_date: str | None = None,
    mature_years: int = 2,
    anchor_window_bars: int = 10,
    ipo_window_start_bar: int = 60,
    ipo_window_end_bar: int = 120,
) -> dict[str, Any]:
    required = {"Close"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"OHLCV data missing columns: {missing}")
    if anchor_window_bars < 0:
        raise ValueError("anchor window bars must be non-negative")
    if ipo_window_start_bar < 1 or ipo_window_end_bar < ipo_window_start_bar:
        raise ValueError("IPO window bars must be positive and ordered")
    if data.empty:
        return {
            "name": "anchor_window_low_hold",
            "status": "insufficient_data",
            "return_pct": None,
        }

    effective_end = pd.Timestamp(end_date) if end_date else data.index[-1]
    first_date = data.index[0]
    two_year_anchor = effective_end - pd.DateOffset(years=mature_years)
    use_two_year_anchor = first_date <= two_year_anchor
    benchmark: dict[str, Any] = {
        "name": "anchor_window_low_hold",
        "description": (
            "For stocks listed more than two years, enter at the lowest close within +/-10 "
            "trading bars around the two-year anchor. For newer stocks, enter at the lowest "
            "close between listed bars 60 and 120. Hold to the final close."
        ),
        "cash": cash,
        "commission": commission,
        "end_date": effective_end.date().isoformat(),
        "mature_years": mature_years,
        "anchor_window_bars": anchor_window_bars,
        "ipo_window_start_bar": ipo_window_start_bar,
        "ipo_window_end_bar": ipo_window_end_bar,
    }

    if use_two_year_anchor:
        anchor_pos = int(data.index.searchsorted(two_year_anchor, side="left"))
        if anchor_pos >= len(data):
            anchor_pos = len(data) - 1
        start_pos = max(anchor_pos - anchor_window_bars, 0)
        end_pos = min(anchor_pos + anchor_window_bars, len(data) - 1)
        mode = "two_year_anchor_window"
        benchmark |= {
            "mode": mode,
            "anchor_date": data.index[anchor_pos].date().isoformat(),
            "target_anchor_date": two_year_anchor.date().isoformat(),
        }
    else:
        if len(data) < ipo_window_start_bar + 1:
            return benchmark | {
                "mode": "ipo_window",
                "status": "insufficient_data",
                "return_pct": None,
            }
        start_pos = ipo_window_start_bar - 1
        end_pos = min(ipo_window_end_bar - 1, len(data) - 1)
        mode = "ipo_window"
        benchmark |= {"mode": mode}

    window = data.iloc[start_pos : end_pos + 1]
    close = window["Close"].dropna()
    if close.empty:
        return benchmark | {"status": "insufficient_window_data", "return_pct": None}

    entry_date = close.idxmin()
    entry_price = float(data.loc[entry_date, "Close"])
    exit_date = data.index[-1]
    exit_price = float(data.iloc[-1]["Close"])
    final_equity = cash / (entry_price * (1 + commission)) * exit_price * (1 - commission)
    return benchmark | {
        "status": "triggered",
        "entry_date": entry_date.date().isoformat(),
        "exit_date": exit_date.date().isoformat(),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "window_start_date": data.index[start_pos].date().isoformat(),
        "window_end_date": data.index[end_pos].date().isoformat(),
        "window_low_close": entry_price,
        "window_low_bar": int(data.index.get_loc(entry_date) + 1),
        "final_equity": final_equity,
        "return_pct": (final_equity / cash - 1) * 100,
    }

def build_aligned_index_hold_benchmark(
    index_data: pd.DataFrame,
    entry_date: str | None,
    exit_date: str | None,
    cash: float = 100_000.0,
    commission: float = 0.0008,
    name: str = "SSE Composite",
) -> dict[str, Any]:
    benchmark: dict[str, Any] = {
        "name": "aligned_index_hold",
        "index_name": name,
        "cash": cash,
        "commission": commission,
        "aligned_entry_date": entry_date,
        "aligned_exit_date": exit_date,
    }
    if not entry_date or not exit_date:
        return benchmark | {"status": "missing_aligned_dates", "return_pct": None}
    required = {"Open", "Close"}
    missing = sorted(required - set(index_data.columns))
    if missing:
        raise ValueError(f"index OHLCV data missing columns: {missing}")

    entry_ts = pd.Timestamp(entry_date)
    exit_ts = pd.Timestamp(exit_date)
    entry_rows = index_data.loc[index_data.index >= entry_ts]
    exit_rows = index_data.loc[index_data.index <= exit_ts]
    if entry_rows.empty or exit_rows.empty:
        return benchmark | {"status": "insufficient_index_data", "return_pct": None}

    actual_entry = entry_rows.index[0]
    actual_exit = exit_rows.index[-1]
    if actual_entry > actual_exit:
        return benchmark | {"status": "invalid_aligned_window", "return_pct": None}

    entry_price = float(index_data.loc[actual_entry, "Open"])
    exit_price = float(index_data.loc[actual_exit, "Close"])
    final_equity = cash / (entry_price * (1 + commission)) * exit_price * (1 - commission)
    return benchmark | {
        "status": "triggered",
        "entry_date": actual_entry.date().isoformat(),
        "exit_date": actual_exit.date().isoformat(),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "final_equity": final_equity,
        "return_pct": (final_equity / cash - 1) * 100,
    }


def run_backtesting_py_report(
    data_path: str | Path,
    output_dir: str | Path = "reports/generated/backtesting_py/603759.SH",
    cash: float = 100_000.0,
    commission: float = 0.0008,
    trade_on_close: bool = False,
    open_browser: bool = False,
    benchmark_anchor_window_bars: int = 10,
    benchmark_mature_years: int = 2,
    benchmark_ipo_window_start_bar: int = 60,
    benchmark_ipo_window_end_bar: int = 120,
    index_data_path: str | Path | None = None,
    index_name: str = "SSE Composite",
) -> dict[str, Any]:
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_ohlcv(data_path)
    backtest = Backtest(
        data,
        SmaVolumeTrendStrategy,
        cash=cash,
        commission=commission,
        trade_on_close=trade_on_close,
        exclusive_orders=True,
        finalize_trades=True,
    )
    stats = backtest.run()

    panel_path = output_dir / "backtesting_py_panel.html"
    backtest.plot(results=stats, filename=str(panel_path), open_browser=open_browser)

    stats_json_path = output_dir / "backtesting_py_stats.json"
    trades_path = output_dir / "backtesting_py_trades.csv"
    equity_path = output_dir / "backtesting_py_equity_curve.csv"

    trades = stats.get("_trades", pd.DataFrame())
    equity = stats.get("_equity_curve", pd.DataFrame())
    if isinstance(trades, pd.DataFrame):
        trades.to_csv(trades_path, index=False)
    if isinstance(equity, pd.DataFrame):
        equity.to_csv(equity_path)

    json_stats = {}
    for key, value in stats.items():
        if str(key).startswith("_"):
            continue
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, pd.Timestamp):
            value = value.isoformat()
        json_stats[str(key)] = value
    anchor_low_benchmark = build_anchor_window_low_hold_benchmark(
        data,
        cash=cash,
        commission=commission,
        end_date=str(data.index.max().date()),
        mature_years=benchmark_mature_years,
        anchor_window_bars=benchmark_anchor_window_bars,
        ipo_window_start_bar=benchmark_ipo_window_start_bar,
        ipo_window_end_bar=benchmark_ipo_window_end_bar,
    )
    json_stats["Anchor Window Low Hold Return [%]"] = anchor_low_benchmark.get("return_pct")
    json_stats["Anchor Window Low Hold Status"] = anchor_low_benchmark.get("status")
    json_stats["Anchor Window Low Hold Entry"] = anchor_low_benchmark.get("entry_date")

    aligned_index_benchmark = None
    if index_data_path is not None:
        aligned_index_benchmark = build_aligned_index_hold_benchmark(
            load_ohlcv(index_data_path),
            entry_date=anchor_low_benchmark.get("entry_date"),
            exit_date=anchor_low_benchmark.get("exit_date"),
            cash=cash,
            commission=commission,
            name=index_name,
        )
        json_stats["Aligned Index Hold Return [%]"] = aligned_index_benchmark.get("return_pct")
        json_stats["Aligned Index Hold Status"] = aligned_index_benchmark.get("status")
        if anchor_low_benchmark.get("return_pct") is not None and aligned_index_benchmark.get(
            "return_pct"
        ) is not None:
            json_stats["Anchor Window Low Hold Excess vs Index [%]"] = (
                anchor_low_benchmark["return_pct"] - aligned_index_benchmark["return_pct"]
            )
    stats_json_path.write_text(json.dumps(json_stats, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    summary = {
        "engine": "backtesting.py",
        "strategy": SmaVolumeTrendStrategy.__name__,
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "panel_path": str(panel_path),
        "stats_path": str(stats_json_path),
        "trades_path": str(trades_path),
        "equity_path": str(equity_path),
        "start": str(data.index.min().date()),
        "end": str(data.index.max().date()),
        "bars": int(len(data)),
        "cash": cash,
        "commission": commission,
        "benchmarks": {
            "anchor_window_low_hold": anchor_low_benchmark,
            "aligned_index_hold": aligned_index_benchmark,
        },
        "stats": json_stats,
    }
    (output_dir / "backtesting_py_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a backtesting.py HTML report for a daily OHLCV CSV.")
    parser.add_argument(
        "--data",
        default="data/raw/daily_bars/eastmoney/qfq/603759.SH/603759.SH_2021-01-01_2026-06-02.csv",
        help="Daily OHLCV CSV from gushen data/raw.",
    )
    parser.add_argument("--output-dir", default="reports/generated/backtesting_py/603759.SH")
    parser.add_argument("--cash", type=float, default=100_000.0)
    parser.add_argument("--commission", type=float, default=0.0008)
    parser.add_argument("--benchmark-anchor-window-bars", type=int, default=10)
    parser.add_argument("--benchmark-mature-years", type=int, default=2)
    parser.add_argument("--benchmark-ipo-window-start-bar", type=int, default=60)
    parser.add_argument("--benchmark-ipo-window-end-bar", type=int, default=120)
    parser.add_argument("--index-data", default=None)
    parser.add_argument("--index-name", default="SSE Composite")
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args(argv)
    summary = run_backtesting_py_report(
        data_path=args.data,
        output_dir=args.output_dir,
        cash=args.cash,
        commission=args.commission,
        benchmark_anchor_window_bars=args.benchmark_anchor_window_bars,
        benchmark_mature_years=args.benchmark_mature_years,
        benchmark_ipo_window_start_bar=args.benchmark_ipo_window_start_bar,
        benchmark_ipo_window_end_bar=args.benchmark_ipo_window_end_bar,
        index_data_path=args.index_data,
        index_name=args.index_name,
        open_browser=args.open_browser,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
