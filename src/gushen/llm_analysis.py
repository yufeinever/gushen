from __future__ import annotations

import json
from pathlib import Path

import requests
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from gushen.llm_config import get_llm_config


class LLMObservation(BaseModel):
    code: str
    name: str
    action: str = Field(description="research_only, observe, avoid, or data_insufficient")
    thesis: str
    positive_factors: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    follow_up: str


class LLMAnalysisReport(BaseModel):
    data_sufficiency: str
    disclaimer: str
    market_context: str
    observations: list[LLMObservation]


def run_llm_analysis(trade_date: str = "2026-05-20", limit: int = 12) -> LLMAnalysisReport:
    console = Console()
    config = get_llm_config()
    if not config.is_configured:
        console.print("[red]LLM config is incomplete.[/]")
        raise SystemExit(2)

    pack_path = Path(f"reports/generated/top100_ai_analysis_pack_{trade_date}.json")
    if not pack_path.exists():
        console.print(f"[red]Missing analysis pack:[/] {pack_path}")
        raise SystemExit(2)

    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    rows = pack["rows"][:limit]
    prompt = _build_prompt(pack, rows)
    raw = _call_llm(config.base_url, config.api_key, config.model, prompt)
    report = _parse_report(raw)
    output_path = Path(f"reports/generated/top100_llm_research_{trade_date}.json")
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    _print_report(console, trade_date, report)
    console.print(f"Report: {output_path}")
    return report


def _call_llm(base_url: str, api_key: str, model: str, prompt: str) -> str:
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a cautious A-share research agent. You must never recommend "
                        "buying or selling when data is insufficient. Output valid JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 4096,
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"]


def _parse_report(raw: str) -> LLMAnalysisReport:
    text = raw.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return LLMAnalysisReport.model_validate_json(text)


def _build_prompt(pack: dict, rows: list[dict]) -> str:
    return json.dumps(
        {
            "task": "Analyze A-share Top100 candidates for research-only observation.",
            "hard_rules": [
                "Data sufficiency must be stated first.",
                "Do not output buy/sell/recommendation.",
                "Allowed actions: research_only, observe, avoid, data_insufficient.",
                "Mention missing data explicitly.",
                "Focus on liquidity, 5/10/20-day returns, MA gaps, 20-day volatility, amount_ratio_5d.",
            ],
            "required_json_schema": {
                "data_sufficiency": "string",
                "disclaimer": "string",
                "market_context": "string",
                "observations": [
                    {
                        "code": "string",
                        "name": "string",
                        "action": "research_only|observe|avoid|data_insufficient",
                        "thesis": "string",
                        "positive_factors": ["string"],
                        "risk_factors": ["string"],
                        "missing_data": ["string"],
                        "follow_up": "string",
                    }
                ],
            },
            "data_sufficiency": pack["data_sufficiency"],
            "rows": rows,
        },
        ensure_ascii=False,
    )


def _print_report(console: Console, trade_date: str, report: LLMAnalysisReport) -> None:
    console.print(f"Data sufficiency: [yellow]{report.data_sufficiency}[/]")
    console.print(report.disclaimer)
    table = Table(title=f"{trade_date} LLM research observations")
    table.add_column("Code")
    table.add_column("Name")
    table.add_column("Action")
    table.add_column("Thesis")
    for item in report.observations:
        table.add_row(item.code, item.name, item.action, item.thesis)
    console.print(table)


def main() -> None:
    run_llm_analysis()


if __name__ == "__main__":
    main()
