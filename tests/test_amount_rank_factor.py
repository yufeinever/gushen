import csv
from pathlib import Path

from gushen.amount_rank_factor import (
    PoolStock,
    build_amount_ranks,
    build_current_amount_rank_rows,
    build_simulation_eval_dates,
    build_simulation_target_dates,
    load_target_points,
    run_rolling_top_amount_simulation,
    summarize_simulation,
)


def test_current_top_amount_rows_capture_prior_rank_improvement(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    dates = [f"2026-01-{day:02d}" for day in range(1, 7)]
    write_history(cache_dir / "000001.SZ_1990-01-01_2026-01-06.csv", "000001.SZ", dates, [10, 9, 8, 7, 6, 100])
    write_history(cache_dir / "000002.SZ_1990-01-01_2026-01-06.csv", "000002.SZ", dates, [90, 80, 70, 60, 50, 40])
    write_history(cache_dir / "000003.SZ_1990-01-01_2026-01-06.csv", "000003.SZ", dates, [80, 70, 60, 50, 40, 30])
    targets = {"2026-01-01", "2026-01-06"}

    points_by_date, points_by_stock, loaded = load_target_points(cache_dir, targets)
    ranks = build_amount_ranks(points_by_date)
    rows = build_current_amount_rank_rows(
        pool=[PoolStock(1, "000001", "000001.SZ", "A")],
        trade_date="2026-01-06",
        lookbacks=[5],
        trading_dates=dates,
        points_by_stock=points_by_stock,
        ranks_by_date=ranks,
    )

    assert loaded == 3
    assert rows[0].current_amount_rank == 1
    assert rows[0].anchor_amount_rank == 3
    assert rows[0].rank_improvement == 2
    assert rows[0].rank_bucket == "001-100"


def test_rolling_simulation_summarizes_rank_buckets(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    dates = [f"2026-01-{day:02d}" for day in range(1, 11)]
    write_history(cache_dir / "000001.SZ_1990-01-01_2026-01-10.csv", "000001.SZ", dates, [10, 9, 8, 7, 6, 100, 100, 100, 100, 100])
    write_history(cache_dir / "000002.SZ_1990-01-01_2026-01-10.csv", "000002.SZ", dates, [90, 80, 70, 60, 50, 40, 40, 40, 40, 40])
    write_history(cache_dir / "000003.SZ_1990-01-01_2026-01-10.csv", "000003.SZ", dates, [80, 70, 60, 50, 40, 30, 30, 30, 30, 30])
    eval_dates = build_simulation_eval_dates(
        trading_dates=dates,
        simulation_start="2026-01-06",
        trade_date="2026-01-10",
        lookbacks=[5],
        forward_horizons=[2],
        step=1,
    )
    targets = build_simulation_target_dates(dates, eval_dates, [5], [2])
    points_by_date, points_by_stock, _ = load_target_points(cache_dir, targets)
    ranks = build_amount_ranks(points_by_date)

    rows = run_rolling_top_amount_simulation(
        eval_dates=eval_dates,
        top_n=1,
        lookbacks=[5],
        forward_horizons=[2],
        trading_dates=dates,
        points_by_stock=points_by_stock,
        ranks_by_date=ranks,
    )
    summaries = summarize_simulation(rows)

    assert rows
    assert any(summary.kind == "anchor_rank" for summary in summaries)
    assert any(summary.kind == "rank_improvement" for summary in summaries)


def write_history(path: Path, ts_code: str, dates: list[str], amounts: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "trade_date",
                "code",
                "name",
                "open",
                "close",
                "high",
                "low",
                "volume",
                "amount",
                "amplitude",
                "pct_change",
                "turnover",
            ],
        )
        writer.writeheader()
        for index, (date, amount) in enumerate(zip(dates, amounts), start=1):
            writer.writerow(
                {
                    "trade_date": date,
                    "code": ts_code,
                    "name": ts_code,
                    "open": index,
                    "close": index,
                    "high": index,
                    "low": index,
                    "volume": amount,
                    "amount": amount,
                    "amplitude": 0,
                    "pct_change": 0,
                    "turnover": 0,
                }
            )
