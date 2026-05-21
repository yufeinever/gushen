from __future__ import annotations

import csv
import json
from pathlib import Path

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from gushen.llm_analysis import _call_llm
from gushen.llm_config import get_llm_config


class TradingAgentsObservation(BaseModel):
    code: str
    name: str
    action: str = Field(description="中文动作：研究观察、继续观察、回避、数据不足")
    market_view: str
    risk_view: str
    fundamental_view: str
    event_view: str
    missing_data: list[str] = Field(default_factory=list)
    next_step: str


class TradingAgentsLLMReport(BaseModel):
    data_sufficiency: str
    disclaimer: str
    analyst_summary: str
    observations: list[TradingAgentsObservation]


def run_tradingagents_llm(trade_date: str = "2026-05-20", limit: int = 12) -> TradingAgentsLLMReport:
    console = Console()
    config = get_llm_config()
    if not config.is_configured:
        console.print("[red]LLM config is incomplete.[/]")
        raise SystemExit(2)

    dataset_dir = Path(f"reports/generated/tradingagents_dataset_{trade_date}")
    if not dataset_dir.exists():
        console.print(f"[red]Missing TradingAgents dataset:[/] {dataset_dir}")
        raise SystemExit(2)

    payload = _load_dataset_payload(dataset_dir, limit)
    prompt = _build_prompt(payload)
    raw = _call_llm(config.base_url, config.api_key, config.model, prompt)
    report = _parse_report(raw)
    output_path = Path(f"reports/generated/tradingagents_llm_research_{trade_date}.json")
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    _print_report(console, trade_date, report)
    console.print(f"Report: {output_path}")
    return report


def _load_dataset_payload(dataset_dir: Path, limit: int) -> dict:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    market = _read_csv(dataset_dir / "market_technical.csv", limit)
    risk = _by_code(_read_csv(dataset_dir / "risk_tradability.csv", limit=10_000))
    fundamentals = _by_code(_read_csv(dataset_dir / "fundamentals.csv", limit=10_000))
    events = _by_code(_read_csv(dataset_dir / "events.csv", limit=10_000))
    rows = []
    for item in market[:limit]:
        code = item["code"]
        rows.append(
            {
                "market_technical": item,
                "risk_tradability": risk.get(code, {}),
                "fundamentals": fundamentals.get(code, {}),
                "events": events.get(code, {}),
            }
        )
    return {"manifest": manifest, "rows": rows}


def _build_prompt(payload: dict) -> str:
    return json.dumps(
        {
            "task": "Run a TradingAgents-style A-share research analysis.",
            "hard_rules": [
                "State data sufficiency first.",
                "Do not output buy/sell/recommendation.",
                "Allowed Chinese actions: 研究观察, 继续观察, 回避, 数据不足.",
                "Use all four datasets: market_technical, risk_tradability, fundamentals, events.",
                "If events are not_loaded or fundamentals are missing, say so explicitly.",
            ],
            "required_json_schema": {
                "data_sufficiency": "string",
                "disclaimer": "string",
                "analyst_summary": "string",
                "observations": [
                    {
                        "code": "string",
                        "name": "string",
                        "action": "研究观察|继续观察|回避|数据不足",
                        "market_view": "string",
                        "risk_view": "string",
                        "fundamental_view": "string",
                        "event_view": "string",
                        "missing_data": ["string"],
                        "next_step": "string",
                    }
                ],
            },
            "dataset": payload,
        },
        ensure_ascii=False,
    )


def _parse_report(raw: str) -> TradingAgentsLLMReport:
    text = raw.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return TradingAgentsLLMReport.model_validate_json(text)


def _print_report(console: Console, trade_date: str, report: TradingAgentsLLMReport) -> None:
    console.print(f"Data sufficiency: [yellow]{report.data_sufficiency}[/]")
    console.print(report.disclaimer)
    table = Table(title=f"{trade_date} TradingAgents LLM research")
    table.add_column("Code")
    table.add_column("Name")
    table.add_column("Action")
    table.add_column("Market")
    table.add_column("Risk")
    for item in report.observations:
        table.add_row(item.code, item.name, item.action, item.market_view, item.risk_view)
    console.print(table)


def _read_csv(path: Path, limit: int) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return rows[:limit]


def _by_code(rows: list[dict]) -> dict[str, dict]:
    return {row["code"]: row for row in rows}


def main() -> None:
    run_tradingagents_llm()


if __name__ == "__main__":
    main()
