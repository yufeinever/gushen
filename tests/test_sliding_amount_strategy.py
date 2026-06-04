from gushen.sliding_amount_strategy import (
    CandidateRow,
    StrategyRule,
    choose_rule,
    compound_return,
    rule_filter,
)


def make_row(**overrides) -> CandidateRow:
    data = {
        "eval_date": "2026-01-01",
        "ts_code": "000001.SZ",
        "name": "A",
        "amount_rank": 20,
        "anchor_rank_100": 80,
        "anchor_rank_200": 120,
        "amount_ratio_20": 1.5,
        "amount_ma20_ma60": 1.1,
        "ret20_pct": 8.0,
        "ret60_pct": 25.0,
        "ma20_gap_pct": 8.0,
        "ma60_gap_pct": 18.0,
        "state": "steady_price_volume_confirm",
        "forward_date": "2026-02-01",
        "forward_return_pct": 5.0,
    }
    data.update(overrides)
    return CandidateRow(**data)


def test_rule_filter_excludes_overheated_rank_persistent_stock() -> None:
    assert rule_filter("rank_persistent_not_hot", make_row())
    assert not rule_filter("rank_persistent_not_hot", make_row(ret60_pct=60.0))
    assert not rule_filter("rank_persistent_not_hot", make_row(anchor_rank_100=500, anchor_rank_200=700))


def test_choose_rule_uses_validation_window_only() -> None:
    rules = [StrategyRule("anti_overheat", "anti"), StrategyRule("quiet_or_pullback", "quiet")]
    rows_by_date = {
        "d1": [make_row(state="quiet_trend", forward_return_pct=-5.0), make_row(forward_return_pct=1.0)],
        "d2": [make_row(state="quiet_trend", forward_return_pct=10.0), make_row(forward_return_pct=0.0)],
    }

    selected, scores = choose_rule(rules, rows_by_date, train_dates=["d1"], validation_dates=["d2"], min_validation_count=1)

    assert selected.rule_id == "quiet_or_pullback"
    assert any(score.selected and score.rule_id == "quiet_or_pullback" for score in scores)


def test_compound_return_skips_missing_periods() -> None:
    assert compound_return([10.0, None, -10.0]) == -1.0