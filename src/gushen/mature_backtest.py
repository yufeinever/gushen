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


def build_causal_trough_recovery_benchmark(
    data: pd.DataFrame,
    cash: float = 100_000.0,
    commission: float = 0.0008,
    ipo_wait_bars: int = 120,
    lookback_bars: int = 120,
    low_proximity_pct: float = 0.05,
    ma_window: int = 20,
) -> dict[str, Any]:
    required = {"Open", "Close"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"OHLCV data missing columns: {missing}")
    if ipo_wait_bars < 1 or lookback_bars < 1 or ma_window < 1:
        raise ValueError("benchmark windows must be positive")
    if low_proximity_pct < 0:
        raise ValueError("low proximity must be non-negative")

    close = data["Close"]
    open_ = data["Open"]
    rolling_low = close.rolling(lookback_bars).min()
    moving_average = close.rolling(ma_window).mean()
    first_signal_index = max(ipo_wait_bars, lookback_bars - 1, ma_window)
    benchmark: dict[str, Any] = {
        "name": "causal_trough_recovery_hold",
        "description": (
            "After listed bars mature, buy next open when close is near the lookback low "
            "and crosses back above the moving average, then hold to the final close."
        ),
        "ipo_wait_bars": ipo_wait_bars,
        "lookback_bars": lookback_bars,
        "low_proximity_pct": low_proximity_pct,
        "ma_window": ma_window,
        "cash": cash,
        "commission": commission,
    }
    if len(data) <= first_signal_index + 1:
        return benchmark | {"status": "insufficient_data", "return_pct": None}

    for signal_pos in range(first_signal_index, len(data) - 1):
        previous_close = close.iloc[signal_pos - 1]
        previous_ma = moving_average.iloc[signal_pos - 1]
        signal_close = close.iloc[signal_pos]
        signal_ma = moving_average.iloc[signal_pos]
        signal_low = rolling_low.iloc[signal_pos]
        if any(pd.isna(value) for value in (previous_ma, signal_ma, signal_low)):
            continue

        near_low = signal_close <= signal_low * (1 + low_proximity_pct)
        crossed_above_ma = previous_close <= previous_ma and signal_close > signal_ma
        if not (near_low and crossed_above_ma):
            continue

        entry_pos = signal_pos + 1
        entry_price = float(open_.iloc[entry_pos])
        exit_price = float(close.iloc[-1])
        final_equity = cash / (entry_price * (1 + commission)) * exit_price * (1 - commission)
        return benchmark | {
            "status": "triggered",
            "signal_date": data.index[signal_pos].date().isoformat(),
            "entry_date": data.index[entry_pos].date().isoformat(),
            "exit_date": data.index[-1].date().isoformat(),
            "signal_close": float(signal_close),
            "signal_ma": float(signal_ma),
            "signal_lookback_low": float(signal_low),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "final_equity": final_equity,
            "return_pct": (final_equity / cash - 1) * 100,
        }

    return benchmark | {"status": "no_signal", "return_pct": None}


def run_backtesting_py_report(
    data_path: str | Path,
    output_dir: str | Path = "reports/generated/backtesting_py/603759.SH",
    cash: float = 100_000.0,
    commission: float = 0.0008,
    trade_on_close: bool = False,
    open_browser: bool = False,
    benchmark_ipo_wait_bars: int = 120,
    benchmark_lookback_bars: int = 120,
    benchmark_low_proximity_pct: float = 0.05,
    benchmark_ma_window: int = 20,
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
    trough_recovery_benchmark = build_causal_trough_recovery_benchmark(
        data,
        cash=cash,
        commission=commission,
        ipo_wait_bars=benchmark_ipo_wait_bars,
        lookback_bars=benchmark_lookback_bars,
        low_proximity_pct=benchmark_low_proximity_pct,
        ma_window=benchmark_ma_window,
    )
    json_stats["Causal Trough Recovery Return [%]"] = trough_recovery_benchmark.get(
        "return_pct"
    )
    json_stats["Causal Trough Recovery Status"] = trough_recovery_benchmark.get("status")
    json_stats["Causal Trough Recovery Signal"] = trough_recovery_benchmark.get("signal_date")
    json_stats["Causal Trough Recovery Entry"] = trough_recovery_benchmark.get("entry_date")
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
            "causal_trough_recovery_hold": trough_recovery_benchmark,
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
    parser.add_argument("--benchmark-ipo-wait-bars", type=int, default=120)
    parser.add_argument("--benchmark-lookback-bars", type=int, default=120)
    parser.add_argument("--benchmark-low-proximity-pct", type=float, default=0.05)
    parser.add_argument("--benchmark-ma-window", type=int, default=20)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args(argv)
    summary = run_backtesting_py_report(
        data_path=args.data,
        output_dir=args.output_dir,
        cash=args.cash,
        commission=args.commission,
        benchmark_ipo_wait_bars=args.benchmark_ipo_wait_bars,
        benchmark_lookback_bars=args.benchmark_lookback_bars,
        benchmark_low_proximity_pct=args.benchmark_low_proximity_pct,
        benchmark_ma_window=args.benchmark_ma_window,
        open_browser=args.open_browser,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
