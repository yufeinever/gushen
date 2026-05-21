from gushen.data import load_sample_top_amount


def test_load_sample_top_amount() -> None:
    result = load_sample_top_amount()

    assert result.stocks
    assert result.stocks[0].amount_rank == 1
    assert result.source.endswith("top_amount_sample.csv")
