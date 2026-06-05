from pathlib import Path

import pandas as pd

from gushen.tushare_daily_update import (
    build_output_paths,
    normalize_adj_factor_frame,
    normalize_daily_frame,
    normalize_trade_date,
    update_tushare_daily_market,
)


class FakeTusharePro:
    def daily(self, trade_date: str) -> pd.DataFrame:
        assert trade_date == "20260605"
        return pd.DataFrame(
            [
                {
                    "ts_code": "000002.SZ",
                    "trade_date": "20260605",
                    "open": 2.0,
                    "high": 2.2,
                    "low": 1.9,
                    "close": 2.1,
                    "pre_close": 2.0,
                    "change": 0.1,
                    "pct_chg": 5.0,
                    "vol": 200.0,
                    "amount": 420.0,
                },
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20260605",
                    "open": 1.0,
                    "high": 1.2,
                    "low": 0.9,
                    "close": 1.1,
                    "pre_close": 1.0,
                    "change": 0.1,
                    "pct_chg": 10.0,
                    "vol": 100.0,
                    "amount": 110.0,
                },
            ]
        )

    def adj_factor(self, trade_date: str) -> pd.DataFrame:
        assert trade_date == "20260605"
        return pd.DataFrame(
            [
                {"ts_code": "000002.SZ", "trade_date": "20260605", "adj_factor": 2.0},
                {"ts_code": "000001.SZ", "trade_date": "20260605", "adj_factor": 1.5},
            ]
        )


def test_normalize_trade_date_accepts_dash_and_compact() -> None:
    assert normalize_trade_date("2026-06-05") == "20260605"
    assert normalize_trade_date("20260605") == "20260605"


def test_normalize_daily_frame_filters_date_and_sorts() -> None:
    frame = pd.DataFrame(
        [
            {
                "ts_code": "000002.SZ",
                "trade_date": "20260605",
                "open": 2,
                "high": 2,
                "low": 2,
                "close": 2,
                "pre_close": 2,
                "change": 0,
                "pct_chg": 0,
                "vol": 2,
                "amount": 2,
            },
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260604",
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
                "pre_close": 1,
                "change": 0,
                "pct_chg": 0,
                "vol": 1,
                "amount": 1,
            },
        ]
    )

    normalized = normalize_daily_frame(frame, "2026-06-05")

    assert normalized["ts_code"].tolist() == ["000002.SZ"]
    assert normalized["trade_date"].tolist() == ["20260605"]


def test_normalize_adj_factor_frame_deduplicates() -> None:
    frame = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": "20260605", "adj_factor": 1.1},
            {"ts_code": "000001.SZ", "trade_date": "20260605", "adj_factor": 1.2},
        ]
    )

    normalized = normalize_adj_factor_frame(frame, "20260605")

    assert len(normalized) == 1
    assert normalized.iloc[0]["adj_factor"] == 1.2


def test_update_tushare_daily_market_writes_daily_adj_and_manifest(tmp_path: Path) -> None:
    result = update_tushare_daily_market(
        trade_date="2026-06-05",
        output_root=tmp_path,
        pro=FakeTusharePro(),
    )
    daily_path, adj_path, manifest_path = build_output_paths(tmp_path, "20260605")

    assert result.daily_rows == 2
    assert result.adj_factor_rows == 2
    assert daily_path.exists()
    assert adj_path.exists()
    assert manifest_path.exists()
    assert pd.read_csv(daily_path)["ts_code"].tolist() == ["000001.SZ", "000002.SZ"]
