# 2026-06-03 Top100 Guided Factor Run

Input: `data/local/Table_4860_2026-06-03.xlsx` from the user-provided intraday transaction amount ranking.

Method:

- Use the first 100 rows by ranking as the external stock pool.
- Normalize A-share codes before backtesting, including zero-padding short numeric codes such as `62` to `000062`.
- For each stock, fetch or reuse front-adjusted daily bars for the two-year research window ending on 2026-06-03.
- Run the guided per-stock factor workflow with train/validation/holdout separation inside that two-year window and per-stock factor selection.
- Execute generated factor signals through `backtesting.py` rather than the previous hand-rolled trade loop.
- Use the same research timing convention: signal on the factor bar, enter on the next bar open, and close through the framework after the configured holding period.
- Compare holdout strategy return against the aligned SSE Composite index hold return for strategy excess.
- Keep the stock anchor-window-low hold baseline separately as `anchor_window_low_hold_return_pct`.
- Compare the strategy against that stock hold baseline as `excess_vs_anchor_window_low_pct`.
- Keep the stock hold baseline's own excess versus SSE Composite as `anchor_window_low_excess_vs_index_pct`.

Generated artifacts are intentionally under ignored paths:

- `reports/generated/top100_2026-06-03_guided_factor_backtests/guided_batch_summary.json`
- `reports/generated/top100_2026-06-03_guided_factor_backtests/guided_batch_summary.csv`
- `reports/generated/top100_2026-06-03_guided_factor_backtests/top100_ranked_strategy_summary.md`
- `reports/generated/top100_2026-06-03_guided_factor_backtests/top100_ranked_strategy_summary.csv`

Run summary:

- Stock count: 100
- Completed with holdout trades: 94
- No holdout factor trade: 6
- Positive strategy excess versus SSE Composite: generated in the latest summary CSV.
- Positive strategy excess versus stock hold baseline: generated in the latest summary CSV.
- Negative strategy return: generated in the latest summary CSV.

Research-only note: these outputs are for simulation and strategy research. They do not account for all event risk, announcements, limit-up/limit-down execution, suspension constraints, intraday VWAP execution, or portfolio-level concentration risk, and must not be treated as buy/sell recommendations.
