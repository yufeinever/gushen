# 2026-06-03 Top100 Guided Factor Run

Input: `data/local/Table_4860_2026-06-03.xlsx` from the user-provided intraday transaction amount ranking.

Method:

- Use the first 100 rows by ranking as the external stock pool.
- Normalize A-share codes before backtesting, including zero-padding short numeric codes such as `62` to `000062`.
- For each stock, fetch or reuse front-adjusted daily bars from 2021-05-06 through 2026-06-03.
- Run the guided per-stock factor workflow with a recent two-year factor window, train/validation/holdout separation, and per-stock factor selection.
- Compare holdout strategy return against the aligned SSE Composite index hold return for strategy excess.
- Keep the stock anchor-window-low hold baseline separately as `anchor_window_low_hold_return_pct` and `anchor_window_low_excess_vs_index_pct`.

Generated artifacts are intentionally under ignored paths:

- `reports/generated/top100_2026-06-03_guided_factor_backtests/guided_batch_summary.json`
- `reports/generated/top100_2026-06-03_guided_factor_backtests/guided_batch_summary.csv`
- `reports/generated/top100_2026-06-03_guided_factor_backtests/top100_ranked_strategy_summary.md`
- `reports/generated/top100_2026-06-03_guided_factor_backtests/top100_ranked_strategy_summary.csv`

Run summary:

- Stock count: 100
- Completed with holdout trades: 94
- No holdout factor trade: 6
- Positive strategy excess versus SSE Composite: 37
- Negative strategy return: 17

Research-only note: these outputs are for simulation and strategy research. They do not account for all event risk, announcements, limit-up/limit-down execution, suspension constraints, intraday VWAP execution, or portfolio-level concentration risk, and must not be treated as buy/sell recommendations.
