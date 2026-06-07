# WonderTrader Trial

This is a small, isolated trial for evaluating WonderTrader/wtpy as the long-term
multi-strategy and multi-stock observability layer.

It deliberately does not add `wtpy` to the main project dependencies. The current
T1000 environment runs Python 3.14, while `wtpy==0.9.9.3` pins older pandas in
its dependency metadata. The bootstrap installs `wtpy` and the required runtime
packages into `.wtpy_trial_vendor` without touching the main `.venv`.

## Bootstrap

```bash
/home/yu/projects/gushen/.venv/bin/python -m gushen.wondertrader_trial.bootstrap
```

## Run 603759.SH Smoke Test

```bash
PYTHONPATH=. /home/yu/projects/gushen/.venv/bin/python -m gushen.wondertrader_trial.run_trial
```

The trial uses:

- source data: `data/local/guided_factor_backtests/daily_bars/qfq/603759.SH_1990-01-01_2026-06-05.csv`
- engine: `WonderTrader/wtpy`
- strategy: `DailyDualThrustStockStrategy`
- output workspace: `reports/generated/wondertrader_trial/603759.SH_daily_dual_thrust`

Expected output files include:

- `outputs_bt/wt_dt_603759/btchart.json`
- `outputs_bt/wt_dt_603759/funds.csv`
- `outputs_bt/wt_dt_603759/trades.csv`
- `outputs_bt/wt_dt_603759/signals.csv`
- `Strategy[wt_dt_603759]_PnLAnalyzing_*.xlsx`

## Serve Backtest Snooper

```bash
PYTHONPATH=. /home/yu/projects/gushen/.venv/bin/python -m gushen.wondertrader_trial.serve_snooper --port 18501
```

Open:

```text
http://192.168.31.53:18501/backtest/backtest.html
```

## Why This Trial Exists

The existing Streamlit panel is useful for a single backtesting.py run, but it is
not a durable platform for many strategies landing on many stocks. WonderTrader
has native concepts for strategy IDs, stock contracts, backtest outputs, runtime
monitoring, and a backtest snooper UI. This trial verifies whether our A-share
cache can enter that ecosystem before we migrate strategy observability.
