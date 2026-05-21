from gushen.akshare_doctor import Check


def test_check_runner_contract() -> None:
    check = Check("sample", "sample check", lambda: 1)

    assert check.runner() == 1
