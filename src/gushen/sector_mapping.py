from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from gushen.data import DailyBar
from gushen.domestic_network import domestic_data_no_proxy
from gushen.research import load_or_fetch_daily_snapshot


@dataclass(frozen=True)
class StockSectorMapRow:
    trade_date: str
    code: str
    name: str
    industry: str
    concepts: str
    source_status: str
    source: str
    confidence: float
    updated_at: str
    note: str


def load_or_build_stock_sector_map(
    top100: list[DailyBar],
    trade_date: str,
    cache_dir: Path = Path("data/local/sector_maps"),
) -> dict[str, StockSectorMapRow]:
    cache_path = cache_dir / f"stock_sector_map_{trade_date}.csv"
    cached = _read_cache(cache_path)
    target_codes = {bar.code.split(".")[0] for bar in top100}
    if target_codes and target_codes.issubset(cached):
        return {code: row for code, row in cached.items() if code in target_codes}
    rows = build_stock_sector_map(top100, trade_date)
    if rows:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _write_cache(cache_path, rows)
        return {row.code: row for row in rows}
    return {code: row for code, row in cached.items() if code in target_codes}


def build_stock_sector_map(top100: list[DailyBar], trade_date: str) -> list[StockSectorMapRow]:
    targets = {bar.code.split(".")[0]: bar.name for bar in top100}
    if not targets:
        return []
    industries = _fetch_sina_members(["\u65b0\u6d6a\u884c\u4e1a", "\u884c\u4e1a"], targets)
    concepts = _fetch_sina_members(["\u6982\u5ff5"], targets, max_boards=80)
    updated_at = datetime.now().isoformat(timespec="seconds")
    rows: list[StockSectorMapRow] = []
    for code, name in targets.items():
        industry_names = industries.get(code, [])
        concept_names = concepts.get(code, [])
        source_parts = []
        if industry_names:
            source_parts.append("Sina industry constituents")
        if concept_names:
            source_parts.append("Sina concept constituents")
        confidence = 0.0
        if industry_names:
            confidence += 0.72
        if concept_names:
            confidence += min(0.2, len(concept_names) * 0.04)
        rows.append(
            StockSectorMapRow(
                trade_date=trade_date,
                code=code,
                name=name,
                industry=industry_names[0] if industry_names else "",
                concepts=";".join(concept_names[:8]),
                source_status="ok" if industry_names else "missing",
                source="; ".join(source_parts) if source_parts else "",
                confidence=round(min(confidence, 0.92), 2),
                updated_at=updated_at,
                note=(
                    "constituent mapping loaded from Sina board details"
                    if industry_names
                    else "no Sina industry constituent match for target stock"
                ),
            )
        )
    return rows


def _fetch_sina_members(
    indicators: list[str],
    targets: dict[str, str],
    max_boards: int | None = None,
) -> dict[str, list[str]]:
    import akshare as ak

    result: dict[str, list[str]] = {code: [] for code in targets}
    remaining = set(targets)
    boards = []
    for indicator in indicators:
        try:
            with domestic_data_no_proxy():
                spot = ak.stock_sector_spot(indicator=indicator)
        except Exception:
            continue
        if spot is None or spot.empty:
            continue
        label_col = _find_column(spot.columns, ["label"])
        name_col = _find_column(spot.columns, ["\u677f\u5757"])
        amount_col = _find_column(spot.columns, ["\u603b\u6210\u4ea4\u989d"])
        pct_col = _find_column(spot.columns, ["\u6da8\u8dcc\u5e45"])
        if not label_col or not name_col:
            continue
        for _, row in spot.iterrows():
            label = str(row.get(label_col, "")).strip()
            name = str(row.get(name_col, "")).strip()
            if label and name:
                boards.append(
                    (
                        indicator,
                        label,
                        name,
                        _float(row.get(amount_col)) if amount_col else 0.0,
                        _float(row.get(pct_col)) if pct_col else 0.0,
                    )
                )
    boards.sort(key=lambda item: (item[3], item[4]), reverse=True)
    if max_boards is not None:
        boards = boards[:max_boards]
    for indicator, label, board_name, _, _ in boards:
        if not remaining and indicator != "\u6982\u5ff5":
            break
        member_codes = _fetch_sina_board_member_codes(label)
        if not member_codes:
            continue
        for code in targets:
            if code not in member_codes:
                continue
            current = result.setdefault(code, [])
            if board_name not in current:
                current.append(board_name)
            remaining.discard(code)
    return {code: names for code, names in result.items() if names}


def _fetch_sina_board_member_codes(label: str) -> set[str]:
    try:
        import akshare as ak

        with domestic_data_no_proxy():
            frame = ak.stock_sector_detail(sector=label)
    except Exception:
        return set()
    if frame is None or frame.empty:
        return set()
    code_col = _find_column(frame.columns, ["code", "symbol"])
    if not code_col:
        return set()
    codes = set()
    for value in frame[code_col].dropna().tolist():
        text = str(value).strip()
        if len(text) >= 6:
            codes.add(text[-6:])
    return codes


def _read_cache(path: Path) -> dict[str, StockSectorMapRow]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as file:
        result = {}
        for row in csv.DictReader(file):
            row["confidence"] = _float(row.get("confidence"))
            result[row["code"]] = StockSectorMapRow(**row)
        return result


def _write_cache(path: Path, rows: Iterable[StockSectorMapRow]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(StockSectorMapRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _find_column(columns, candidates: list[str]) -> str | None:
    lookup = {str(column): column for column in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    for column in columns:
        text = str(column)
        if any(candidate in text for candidate in candidates):
            return column
    return None


def _float(value) -> float:
    try:
        if value in {None, "-", ""}:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    from collections import Counter

    from rich.console import Console
    from rich.table import Table

    trade_date = "2026-05-20"
    snapshot = load_or_fetch_daily_snapshot(trade_date)
    top100 = sorted(snapshot, key=lambda item: item.amount, reverse=True)[:100]
    rows = list(load_or_build_stock_sector_map(top100, trade_date).values())
    status = Counter(row.source_status for row in rows)
    industry_coverage = sum(1 for row in rows if row.industry)
    concept_coverage = sum(1 for row in rows if row.concepts)
    table = Table(title=f"{trade_date} Top100 stock-sector map")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("rows", str(len(rows)))
    table.add_row("ok", str(status.get("ok", 0)))
    table.add_row("missing", str(status.get("missing", 0)))
    table.add_row("industry coverage", f"{industry_coverage}/{len(rows)}")
    table.add_row("concept coverage", f"{concept_coverage}/{len(rows)}")
    console = Console()
    console.print(table)
    if rows:
        console.print(
            "Cache: data/local/sector_maps/"
            f"stock_sector_map_{trade_date}.csv; concept coverage remains empty when Sina concept details fail."
        )


if __name__ == "__main__":
    main()
