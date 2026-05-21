from __future__ import annotations

import requests
from rich.console import Console

from gushen.llm_config import get_llm_config


def main() -> None:
    console = Console()
    config = get_llm_config()
    if not config.is_configured:
        console.print("[red]LLM config is incomplete.[/]")
        raise SystemExit(2)

    response = requests.post(
        f"{config.base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a terse connectivity checker.",
                },
                {
                    "role": "user",
                    "content": "Reply with exactly: ok",
                },
            ],
            "temperature": 0,
            "max_tokens": 8,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    console.print(f"LLM smoke response: {content}")


if __name__ == "__main__":
    main()
