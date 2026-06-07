from __future__ import annotations

from gushen.wondertrader_trial.run_trial import ts_code_to_wt


def test_ts_code_to_wt_shanghai() -> None:
    assert ts_code_to_wt("603759.SH") == ("SSE", "STK", "603759")


def test_ts_code_to_wt_shenzhen() -> None:
    assert ts_code_to_wt("000001.SZ") == ("SZSE", "STK", "000001")

