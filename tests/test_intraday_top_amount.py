from gushen.intraday_top_amount import _parse_eastmoney_row, _parse_sina_row


def test_parse_intraday_top_amount_row() -> None:
    row = _parse_eastmoney_row(
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


def test_parse_sina_intraday_top_amount_row() -> None:
    row = _parse_sina_row(
        {
            "\u4ee3\u7801": "sz002156",
            "\u540d\u79f0": "\u901a\u5bcc\u5fae\u7535",
            "\u6700\u65b0\u4ef7": 73.68,
            "\u6da8\u8dcc\u989d": 3.9,
            "\u6da8\u8dcc\u5e45": 5.59,
            "\u6628\u6536": 69.78,
            "\u4eca\u5f00": 73.02,
            "\u6700\u9ad8": 74.89,
            "\u6700\u4f4e": 72.0,
            "\u6210\u4ea4\u91cf": 942208,
            "\u6210\u4ea4\u989d": 6944739885.41,
            "\u65f6\u95f4\u6233": "12:54:59",
        },
        1,
        "2026-05-26T12:55:01",
    )

    assert row.code == "002156.SZ"
    assert row.source_trade_date == "20260526"
    assert row.source_time == "2026-05-26T12:54:59"
    assert row.amount == 6944739885.41
