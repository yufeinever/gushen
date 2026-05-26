from gushen.data import load_sample_top_amount


def test_load_sample_top_amount() -> None:
    result = load_sample_top_amount()

    assert result.stocks
    assert result.stocks[0].amount_rank == 1
    assert result.source.endswith("top_amount_sample.csv")


def test_daily_bar_falls_back_to_tencent(monkeypatch) -> None:
    import pandas as pd

    from gushen import data

    def fail_eastmoney(*args, **kwargs):
        raise RuntimeError("eastmoney unavailable")

    def fake_tencent(*args, **kwargs):
        return pd.DataFrame(
            [
                {
                    "date": "2026-05-20",
                    "open": 9.8,
                    "close": 10.0,
                    "high": 10.1,
                    "low": 9.7,
                    "amount": 1000.0,
                },
                {
                    "date": "2026-05-21",
                    "open": 10.1,
                    "close": 10.5,
                    "high": 10.8,
                    "low": 10.0,
                    "amount": 1200.0,
                },
            ]
        )

    monkeypatch.setattr(data, "_fetch_daily_bar_eastmoney", fail_eastmoney)
    monkeypatch.setattr("akshare.stock_zh_a_hist_tx", fake_tencent)

    bar = data.fetch_daily_bar("000001", "平安银行", "2026-05-21")

    assert bar is not None
    assert bar.code == "000001.SZ"
    assert bar.close == 10.5
    assert round(bar.pct_change, 4) == 0.05
    assert bar.amount == 1200.0 * 100.0 * 10.5


def test_daily_bars_falls_back_to_tencent(monkeypatch) -> None:
    import pandas as pd

    from gushen import data

    def fail_eastmoney(*args, **kwargs):
        raise RuntimeError("eastmoney unavailable")

    def fake_tencent(*args, **kwargs):
        return pd.DataFrame(
            [
                {
                    "date": "2026-05-20",
                    "open": 100.0,
                    "close": 110.0,
                    "high": 112.0,
                    "low": 98.0,
                    "amount": 2000.0,
                }
            ]
        )

    monkeypatch.setattr(data, "_fetch_daily_bars_eastmoney", fail_eastmoney)
    monkeypatch.setattr("akshare.stock_zh_a_hist_tx", fake_tencent)

    bars = data.fetch_daily_bars("603986.SH", "兆易创新", "2026-05-01", "2026-05-21")

    assert len(bars) == 1
    assert bars[0].code == "603986.SH"
    assert bars[0].volume == 2000.0 * 100.0
