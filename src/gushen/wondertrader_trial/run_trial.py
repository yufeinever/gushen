from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd


DEFAULT_DATA = Path(
    "data/local/guided_factor_backtests/daily_bars/qfq/603759.SH_1990-01-01_2026-06-05.csv"
)
DEFAULT_OUTPUT_ROOT = Path("reports/generated/wondertrader_trial")


def ensure_vendor(vendor_dir: Path) -> None:
    if not vendor_dir.exists():
        raise RuntimeError(
            f"wtpy vendor dir not found: {vendor_dir}. "
            "Run: python -m gushen.wondertrader_trial.bootstrap"
        )
    sys.path.insert(0, str(vendor_dir.resolve()))


def ts_code_to_wt(ts_code: str) -> tuple[str, str, str]:
    symbol, suffix = ts_code.split(".")
    exchange = {"SH": "SSE", "SZ": "SZSE"}[suffix]
    return exchange, "STK", symbol


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_common_files(common_dir: Path, exchange: str, product: str, symbol: str, name: str) -> None:
    write_json(
        common_dir / "stk_comms.json",
        {
            "SSE": {
                "STK": {
                    "category": 0,
                    "covermode": 0,
                    "exchg": "SSE",
                    "holiday": "CHINA",
                    "name": "上证股票",
                    "precision": 2,
                    "pricemode": 1,
                    "pricetick": 0.01,
                    "session": "SD0930",
                    "volscale": 1,
                    "trademode": 2,
                }
            },
            "SZSE": {
                "STK": {
                    "category": 0,
                    "covermode": 0,
                    "exchg": "SZSE",
                    "holiday": "CHINA",
                    "name": "深证股票",
                    "precision": 2,
                    "pricemode": 1,
                    "pricetick": 0.01,
                    "session": "SD0930",
                    "volscale": 1,
                    "trademode": 2,
                }
            },
        },
    )
    write_json(
        common_dir / "stocks.json",
        {
            exchange: {
                symbol: {
                    "code": symbol,
                    "exchg": exchange,
                    "name": name,
                    "product": product,
                }
            }
        },
    )
    write_json(
        common_dir / "stk_sessions.json",
        {
            "SD0930": {
                "name": "股票白盘0930",
                "offset": 0,
                "auction": {"from": 929, "to": 930},
                "sections": [{"from": 930, "to": 1130}, {"from": 1300, "to": 1500}],
            }
        },
    )
    write_json(common_dir / "holidays.json", {"CHINA": []})
    write_json(
        common_dir / "fees_stk.json",
        {
            "SSE.STK": {"open": 0.001, "close": 0.0012, "closetoday": 0.0, "byvolume": False},
            "SZSE.STK": {"open": 0.001, "close": 0.0012, "closetoday": 0.0, "byvolume": False},
        },
    )


def write_config_files(workspace: Path) -> None:
    (workspace / "logcfgbt.yaml").write_text(
        """root:
  async: false
  level: info
  sinks:
    - type: basic_file_sink
      filename: BtLogs/Runner.log
      pattern: '[%Y.%m.%d %H:%M:%S - %-5l] %v'
      truncate: true
    - type: console_sink
      pattern: '[%m.%d %H:%M:%S - %^%-5l%$] %v'
dyn_pattern:
  strategy:
    async: false
    level: info
    sinks:
      - type: basic_file_sink
        filename: BtLogs/Strategy_%s.log
        pattern: '[%Y.%m.%d %H:%M:%S - %-5l] %v'
        truncate: false
""",
        encoding="utf-8",
    )
    (workspace / "configbt.yaml").write_text(
        """env:
  mocker: cta
replayer:
  basefiles:
    commodity: common/stk_comms.json
    contract: common/stocks.json
    holiday: common/holidays.json
    session: common/stk_sessions.json
  fees: common/fees_stk.json
  mode: csv
  path: storage/
""",
        encoding="utf-8",
    )


def convert_daily_csv(source: Path, target: Path) -> dict:
    frame = pd.read_csv(source)
    required = {"trade_date", "open", "high", "low", "close", "volume", "amount", "code", "name"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing daily bar columns: {missing}")
    output = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["trade_date"]).dt.strftime("%Y/%m/%d"),
            "open": frame["open"],
            "high": frame["high"],
            "low": frame["low"],
            "close": frame["close"],
            "volume": frame["volume"],
            "turnover": frame["amount"],
        }
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(target, index=False)
    return {
        "ts_code": str(frame["code"].iloc[0]),
        "name": str(frame["name"].iloc[0]),
        "rows": int(len(output)),
        "start": str(frame["trade_date"].iloc[0]),
        "end": str(frame["trade_date"].iloc[-1]),
    }


def run_trial(
    source_csv: Path = DEFAULT_DATA,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    vendor_dir: Path = Path(".wtpy_trial_vendor"),
    strategy_name: str = "wt_dt_603759",
    bar_count: int = 80,
    days: int = 30,
    k1: float = 0.1,
    k2: float = 0.1,
    init_capital: float = 100_000.0,
) -> dict:
    ensure_vendor(vendor_dir)

    from wtpy import EngineType, WtBtEngine
    from wtpy.apps.WtBtAnalyst import WtBtAnalyst

    from gushen.wondertrader_trial.strategy import DailyDualThrustStockStrategy

    source_csv = source_csv.resolve()
    workspace = output_root / "603759.SH_daily_dual_thrust"
    if workspace.exists():
        shutil.rmtree(workspace)
    common_dir = workspace / "common"
    storage_csv_dir = workspace / "storage" / "csv"
    workspace.mkdir(parents=True, exist_ok=True)

    metadata = convert_daily_csv(source_csv, storage_csv_dir / "SSE.STK.603759_d.csv")
    exchange, product, symbol = ts_code_to_wt(metadata["ts_code"])
    std_code = f"{exchange}.{product}.{symbol}"
    write_common_files(common_dir, exchange, product, symbol, metadata["name"])
    write_config_files(workspace)
    write_json(
        workspace / "trial_metadata.json",
        {
            **metadata,
            "engine": "WonderTrader/wtpy",
            "strategy": "DailyDualThrustStockStrategy",
            "std_code": std_code,
            "adjust": "qfq input data, no wtpy suffix",
            "workspace": str(workspace),
        },
    )

    old_cwd = Path.cwd()
    os.chdir(workspace)
    engine = WtBtEngine(EngineType.ET_CTA, outDir="./outputs_bt")
    try:
        engine.init(folder="common/", cfgfile="configbt.yaml", commfile="stk_comms.json", contractfile="stocks.json")
        engine.configBacktest(
            int(metadata["start"].replace("-", "") + "0930"),
            int(metadata["end"].replace("-", "") + "1500"),
        )
        engine.configBTStorage(mode="csv", path="storage/")
        engine.commitBTConfig()
        strategy = DailyDualThrustStockStrategy(
            name=strategy_name,
            code=std_code,
            bar_count=bar_count,
            days=days,
            k1=k1,
            k2=k2,
            trade_unit=100,
        )
        engine.set_cta_strategy(strategy)
        engine.run_backtest()

        analyst = WtBtAnalyst()
        analyst.add_strategy(strategy_name, folder="./outputs_bt/", init_capital=init_capital, rf=0.0, annual_trading_days=240)
        analyst.run()
    finally:
        engine.release_backtest()
        os.chdir(old_cwd)

    strategy_dir = workspace / "outputs_bt" / strategy_name
    result = {
        "workspace": str(workspace),
        "strategy_dir": str(strategy_dir),
        "metadata": str(workspace / "trial_metadata.json"),
        "btchart": str(strategy_dir / "btchart.json"),
        "funds": str(strategy_dir / "funds.csv"),
        "trades": str(strategy_dir / "trades.csv"),
        "signals": str(strategy_dir / "signals.csv"),
        "snooper_url_hint": "http://192.168.31.53:<port>/backtest/backtest.html",
    }
    write_json(workspace / "trial_result.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a WonderTrader/wtpy trial on cached A-share daily bars.")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--vendor-dir", default=".wtpy_trial_vendor")
    args = parser.parse_args(argv)
    result = run_trial(Path(args.data), Path(args.output_root), Path(args.vendor_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

