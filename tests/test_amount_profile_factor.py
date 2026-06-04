from gushen.amount_profile_factor import (
    amount_ratio_bucket,
    profile_state,
    return_bucket,
)


def test_profile_state_classifies_steady_price_volume_confirmation() -> None:
    assert (
        profile_state(
            amount_ratio_20=1.5,
            amount_ma20_ma60=1.2,
            ret20_pct=8.0,
            ret60_pct=30.0,
            ma20_gap_pct=6.0,
        )
        == "steady_price_volume_confirm"
    )


def test_profile_state_flags_late_climax_and_weak_high_volume() -> None:
    assert (
        profile_state(
            amount_ratio_20=4.2,
            amount_ma20_ma60=1.8,
            ret20_pct=55.0,
            ret60_pct=80.0,
            ma20_gap_pct=35.0,
        )
        == "late_climax_volume"
    )
    assert (
        profile_state(
            amount_ratio_20=2.5,
            amount_ma20_ma60=1.3,
            ret20_pct=-5.0,
            ret60_pct=-12.0,
            ma20_gap_pct=-4.0,
        )
        == "weak_high_volume"
    )


def test_factor_buckets_are_stable() -> None:
    assert amount_ratio_bucket(0.9) == "<1"
    assert amount_ratio_bucket(4.0) == ">=4"
    assert return_bucket(-0.1) == "<0"
    assert return_bucket(51.0) == ">=50"
