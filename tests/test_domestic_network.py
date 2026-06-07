import os
from concurrent.futures import ThreadPoolExecutor

import requests

from gushen.domestic_network import direct_requests_get, domestic_data_no_proxy


def test_domestic_data_no_proxy_restores_request_after_nested_use() -> None:
    original_request = requests.sessions.Session.request

    with domestic_data_no_proxy():
        with domestic_data_no_proxy():
            assert requests.sessions.Session.request is not original_request

    assert requests.sessions.Session.request is original_request


def test_domestic_data_no_proxy_is_safe_for_concurrent_entry() -> None:
    original_request = requests.sessions.Session.request

    def enter_context() -> None:
        with domestic_data_no_proxy():
            assert requests.sessions.Session.request is not original_request

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(enter_context) for _ in range(12)]
        for future in futures:
            future.result()

    assert requests.sessions.Session.request is original_request


def test_direct_requests_get_ignores_proxy_env(monkeypatch) -> None:
    captured = {}

    class DummyResponse:
        pass

    class DummySession:
        def __enter__(self):
            captured["initial_trust_env"] = self.trust_env
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __init__(self):
            self.trust_env = True

        def get(self, url, **kwargs):
            captured["trust_env"] = self.trust_env
            captured["url"] = url
            captured["kwargs"] = kwargs
            return DummyResponse()

    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setattr(requests, "Session", DummySession)

    response = direct_requests_get("https://example.test", timeout=3)

    assert isinstance(response, DummyResponse)
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:9"
    assert captured["trust_env"] is False
    assert captured["kwargs"]["proxies"] == {}
