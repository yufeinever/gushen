from gushen.intraday_top_amount import _parse_row


def test_parse_intraday_top_amount_row() -> None:
    row = _parse_row(
        {
            "f12": "600584",
            "f14": "长电科技",
            "f2": 85.16,
            "f3": 6.22,
            "f4": 4.99,
            "f5": 904332,
            "f6": 7730492669.0,
            "f7": 7.25,
            "f8": 5.05,
            "f15": 88.0,
            "f16": 82.19,
            "f17": 83.0,
            "f18": 80.17,
            "f124": 1779759544,
            "f297": 20260526,
        },
        1,
        "2026-05-26T09:39:07",
    )

    assert row.code == "600584.SH"
    assert row.source_trade_date == "20260526"
    assert row.source_time == "2026-05-26T09:39:04"
    assert row.amount == 7730492669.0
