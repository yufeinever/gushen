from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


DEFAULT_WORKSPACE = Path("reports/generated/wondertrader_trial/603759.SH_daily_dual_thrust")


def ensure_vendor(vendor_dir: Path) -> None:
    if not vendor_dir.exists():
        raise RuntimeError(
            f"wtpy vendor dir not found: {vendor_dir}. "
            "Run: python -m gushen.wondertrader_trial.bootstrap"
        )
    sys.path.insert(0, str(vendor_dir.resolve()))


def serve_snooper(
    workspace: Path = DEFAULT_WORKSPACE,
    vendor_dir: Path = Path(".wtpy_trial_vendor"),
    host: str = "0.0.0.0",
    port: int = 18501,
) -> None:
    ensure_vendor(vendor_dir)

    from wtpy import WtDtServo
    from wtpy.monitor import WtBtSnooper

    workspace = workspace.resolve()
    outputs_bt = workspace / "outputs_bt"
    if not outputs_bt.exists():
        raise RuntimeError(f"WonderTrader output workspace not found: {workspace}")

    data_json = Path("data.json")
    old_cwd = Path.cwd()
    os.chdir(workspace)
    try:
        data_json.write_text(
            json.dumps(
                {
                    "workspace": [
                        {
                            "name": "gushen-603759-daily-dual-thrust",
                            "path": str(outputs_bt),
                            "id": "gushen_603759_daily_dual_thrust",
                        }
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        dt_servo = WtDtServo()
        dt_servo.setBasefiles(folder=str(workspace / "common"))
        dt_servo.setStorage(path=str(workspace / "storage"))
        snooper = WtBtSnooper(dt_servo)
        snooper.run_as_server(port=port, host=host)
    finally:
        os.chdir(old_cwd)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve WonderTrader Backtest Snooper for trial outputs.")
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--vendor-dir", default=".wtpy_trial_vendor")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18501)
    args = parser.parse_args(argv)
    serve_snooper(Path(args.workspace), Path(args.vendor_dir), args.host, args.port)


if __name__ == "__main__":
    main()
