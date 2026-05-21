from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def get_llm_config() -> LLMConfig:
    load_dotenv()
    return LLMConfig(
        base_url=os.getenv("OPENAI_BASE_URL", ""),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("GUSHEN_LLM_MODEL", ""),
    )
