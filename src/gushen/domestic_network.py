from __future__ import annotations

import os
import urllib.request
from contextlib import contextmanager
from typing import Iterator


_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "FTP_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "ftp_proxy",
    "no_proxy",
)


@contextmanager
def domestic_data_no_proxy() -> Iterator[None]:
    """Temporarily force direct connections for domestic market-data providers."""
    import requests

    old_env = {key: os.environ.get(key) for key in _PROXY_ENV_KEYS}
    original_getproxies = urllib.request.getproxies
    original_request = requests.sessions.Session.request

    for key in _PROXY_ENV_KEYS:
        os.environ.pop(key, None)

    def direct_request(self, method, url, **kwargs):
        previous_trust_env = self.trust_env
        self.trust_env = False
        kwargs["proxies"] = {}
        try:
            return original_request(self, method, url, **kwargs)
        finally:
            self.trust_env = previous_trust_env

    urllib.request.getproxies = lambda: {}
    requests.sessions.Session.request = direct_request
    try:
        yield
    finally:
        requests.sessions.Session.request = original_request
        urllib.request.getproxies = original_getproxies
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
