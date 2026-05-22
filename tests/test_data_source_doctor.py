from gushen.data_source_doctor import DataSourceCheck, _run_check


def test_data_source_doctor_records_failure() -> None:
    check = DataSourceCheck(
        name="sample",
        category="test",
        provider="local",
        endpoint="sample",
        purpose="sample failure",
        domestic_direct=False,
        required_for_gate=False,
        runner=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    status = _run_check(check, "2026-05-22T09:30:00")

    assert status.status == "failed"
    assert status.error_type == "RuntimeError"
    assert status.rows == 0
