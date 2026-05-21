from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from gushen.agents import CandidateState, SentimentNarrativeAgent, StockContext
from gushen.agent_schemas import SentimentNarrative


@dataclass(frozen=True)
class NarrativeRow:
    trade_date: str
    code: str
    name: str
    action: str
    crowding_risk: str
    quality_score: float
    dominant_narratives: str
    evidence_backed_claims: str
    unsupported_claims: str
    verification_needed: str


def build_sentiment_narratives(trade_date: str = "2026-05-20") -> list[NarrativeRow]:
    events_path = Path(f"reports/generated/tradingagents_dataset_{trade_date}/events.csv")
    if not events_path.exists():
        raise FileNotFoundError(events_path)
    rows = []
    agent = SentimentNarrativeAgent()
    with events_path.open(newline="", encoding="utf-8") as file:
        for event in csv.DictReader(file):
            stock = StockContext(
                date=trade_date,
                code=event["code"],
                name=event["name"],
                amount_rank=int(event["amount_rank"]),
                amount=0.0,
                pct_change=0.0,
                momentum_5d=0.0,
                volatility_20d=0.0,
            )
            state = CandidateState(stock=stock, artifacts={"event_summary": event["event_summary"]})
            decision = agent.decide(state)
            narrative = state.artifacts["sentiment_narrative"]
            if not isinstance(narrative, SentimentNarrative):
                continue
            rows.append(
                NarrativeRow(
                    trade_date=trade_date,
                    code=event["code"],
                    name=event["name"],
                    action=decision.verdict,
                    crowding_risk=narrative.crowding_risk,
                    quality_score=narrative.quality_score,
                    dominant_narratives=" | ".join(narrative.dominant_narratives),
                    evidence_backed_claims=" | ".join(narrative.evidence_backed_claims),
                    unsupported_claims=" | ".join(narrative.unsupported_claims),
                    verification_needed=" | ".join(narrative.verification_needed),
                )
            )
    write_narratives(trade_date, rows)
    print_narratives(trade_date, rows)
    return rows


def write_narratives(trade_date: str, rows: list[NarrativeRow]) -> None:
    output = Path(f"reports/generated/sentiment_narratives_{trade_date}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(NarrativeRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def print_narratives(trade_date: str, rows: list[NarrativeRow]) -> None:
    console = Console()
    table = Table(title=f"{trade_date} A-share sentiment narratives")
    table.add_column("Code")
    table.add_column("Name")
    table.add_column("Action")
    table.add_column("Crowding")
    table.add_column("Quality", justify="right")
    table.add_column("Narrative")
    for row in rows[:15]:
        table.add_row(
            row.code,
            row.name,
            row.action,
            row.crowding_risk,
            f"{row.quality_score:.2f}",
            row.dominant_narratives[:80],
        )
    console.print(table)
    console.print(f"Rows: {len(rows)}")


def main() -> None:
    build_sentiment_narratives()


if __name__ == "__main__":
    main()
