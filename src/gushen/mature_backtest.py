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


def run_backtesting_py_report(
    data_path: str | Path,
    output_dir: str | Path = "reports/generated/backtesting_py/603759.SH",
    cash: float = 100_000.0,
    commission: float = 0.0008,
    trade_on_close: bool = False,
    open_browser: bool = False,
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
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args(argv)
    summary = run_backtesting_py_report(
        data_path=args.data,
        output_dir=args.output_dir,
        cash=args.cash,
        commission=args.commission,
        open_browser=args.open_browser,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
