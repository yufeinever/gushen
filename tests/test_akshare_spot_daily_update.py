from pathlib import Path

import pandas as pd

from gushen.akshare_spot_daily_update import (
    assess_valid_rows,
    normalize_spot_frame,
    update_akshare_spot_daily,
)


def sample_spot_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "序号": 2,
                "代码": "000002",
                "名称": "B",
                "最新价": 11.0,
                "涨跌幅": 1.5,
                "涨跌额": 0.1,
                "成交量": 1000,
                "成交额": 11000,
                "振幅": 2.0,
                "最高": 11.2,
                "最低": 10.8,
                "今开": 10.9,
                "昨收": 10.8,
                "量比": 1.1,
                "换手率": 3.0,
            },
            {
                "序号": 1,
                "代码": "600001",
                "名称": "A",
                "最新价": 10.0,
                "涨跌幅": 2.5,
                "涨跌额": 0.2,
                "成交量": 2000,
                "成交额": 20000,
                "振幅": 3.0,
                "最高": 10.5,
                "最低": 9.8,
                "今开": 9.9,
                "昨收": 9.8,
                "量比": 1.2,
                "换手率": 4.0,
            },
        ]
    )


def test_normalize_spot_frame_maps_eastmoney_columns() -> None:
    frame = normalize_spot_frame(sample_spot_frame(), "2026-06-05", "2026-06-05T16:00:00")

    assert frame["ts_code"].tolist() == ["000002.SZ", "600001.SH"]
    assert frame.iloc[0]["trade_date"] == "2026-06-05"
    assert frame.iloc[0]["close"] == 11.0
    assert assess_valid_rows(frame) == 2


def test_update_akshare_spot_daily_accepts_injected_frame(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"

    result = update_akshare_spot_daily(
        trade_date="2026-06-05",
        output_root=tmp_path / "market",
        status_path=status_path,
        frame=sample_spot_frame(),
        min_rows=2,
        min_valid_rows=2,
    )

    assert result.rows == 2
    assert result.valid_rows == 2
    assert Path(result.output_path).exists()
    assert status_path.exists()
