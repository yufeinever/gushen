from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "https://web.mt180.com"
DEFAULT_OUTPUT_DIR = Path("data/local/factor_library/mt180")
DEFAULT_PAGE_SIZE = 32
DEFAULT_SLEEP_SECONDS = 0.0
DEFAULT_WORKERS = 8
SOURCE_SITE = "mt180"
SOURCE_NAME = "指标公式评测室/指标广场"


CATEGORY_LABELS = {
    "trend": "趋势",
    "oscillator": "摆动",
    "volume": "成交量",
    "selection": "选股",
    "pattern": "形态",
    "moneyflow": "资金",
    "other": "其他",
}

INDICATOR_TYPE_LABELS = {
    1: "主图",
    2: "副图",
    3: "分时图",
}


@dataclass(frozen=True)
class Mt180IndicatorRecord:
    source_site: str
    source_name: str
    source_url: str
    source_id: str
    imported_at: str
    name: str
    short_name: str
    category: str
    category_label: str
    indicator_type: int
    indicator_type_label: str
    description: str
    usage_guide: str | None
    author_id: str | None
    author_phone: str | None
    author_nickname: str | None
    sales_count: int
    favorites_count: int
    rating_avg: float | None
    rating_count: int
    tip_price: float
    is_public: bool | None
    hide_formula: bool
    can_view_formula: bool
    can_apply: bool
    formula_sha256: str
    formula: str


@dataclass(frozen=True)
class Mt180SkippedRecord:
    source_site: str
    source_id: str
    source_url: str
    name: str
    reason: str
    status: str
    imported_at: str


@dataclass(frozen=True)
class Mt180ImportSummary:
    source_site: str
    source_name: str
    base_url: str
    imported_at: str
    list_total: int
    requested_limit: int | None
    query_categories: list[str | None]
    fetched_list_items: int
    imported_count: int
    skipped_count: int
    output_dir: str
    manifest_path: str
    skipped_path: str


class Mt180Client:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def clone(self) -> "Mt180Client":
        clone = Mt180Client(base_url=self.base_url, timeout=self.timeout)
        clone.session.headers.update(dict(self.session.headers))
        return clone

    def login(self, phone: str, password: str) -> None:
        response = self.session.post(
            f"{self.base_url}/api/auth/login",
            json={"phone": phone, "password": password},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        token = payload.get("token")
        if not token:
            raise RuntimeError("mt180 login succeeded but did not return a token")
        self.session.headers.update({"Authorization": f"Bearer {token}"})

    def marketplace_page(
        self,
        page: int,
        page_size: int,
        sort: str,
        category: str | None = None,
        indicator_type: int | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"page": page, "pageSize": page_size, "sort": sort}
        if category:
            params["category"] = category
        if indicator_type is not None:
            params["indicatorType"] = indicator_type
        return self._get_json("/api/marketplace", params=params)

    def marketplace_detail(self, indicator_id: str) -> dict[str, Any]:
        payload = self._get_json(f"/api/marketplace/{indicator_id}")
        indicator = payload.get("indicator")
        if not isinstance(indicator, dict):
            raise RuntimeError(f"mt180 detail response missing indicator for {indicator_id}")
        return indicator

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}{path}",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"mt180 response is not a JSON object: {path}")
        return payload


def import_visible_marketplace_factors(
    client: Mt180Client,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    sort: str = "sales",
    category: str | None = None,
    indicator_type: int | None = None,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    workers: int = DEFAULT_WORKERS,
    all_categories: bool = False,
) -> Mt180ImportSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    indicators_dir = output_dir / "indicators"
    formulas_dir = output_dir / "formulas"
    indicators_dir.mkdir(parents=True, exist_ok=True)
    formulas_dir.mkdir(parents=True, exist_ok=True)

    imported_at = utc_now()
    manifest_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    fetched_list_items = 0
    processed_items = 0
    list_total = 0
    query_categories: list[str | None] = (
        [None, *CATEGORY_LABELS] if all_categories else [category]
    )

    for query_category in query_categories:
        page = 1
        query_total = 0
        while True:
            payload = client.marketplace_page(
                page=page,
                page_size=page_size,
                sort=sort,
                category=query_category,
                indicator_type=indicator_type,
            )
            page_items = payload.get("list") or []
            if not isinstance(page_items, list):
                raise RuntimeError("mt180 marketplace page has no list array")
            query_total = int(payload.get("total") or query_total or 0)
            if page == 1:
                list_total += query_total
            if not page_items:
                break

            batch: list[tuple[str, dict[str, Any]]] = []
            for item in page_items:
                if not isinstance(item, dict):
                    continue
                indicator_id = str(item.get("id") or "").strip()
                if not indicator_id or indicator_id in seen_ids:
                    continue
                seen_ids.add(indicator_id)
                if limit is not None and processed_items >= limit:
                    break
                fetched_list_items += 1
                processed_items += 1
                batch.append((indicator_id, item))

            for imported, skipped in _fetch_batch(client, batch, imported_at, workers):
                if imported is not None:
                    _write_indicator_record(imported, indicators_dir, formulas_dir)
                    manifest_rows.append(_manifest_row(imported))
                elif skipped is not None:
                    skipped_rows.append(asdict(skipped))
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

            if limit is not None and processed_items >= limit:
                break
            if page * page_size >= query_total:
                break
            page += 1
        if limit is not None and processed_items >= limit:
            break

    manifest_path = output_dir / "manifest.json"
    skipped_path = output_dir / "skipped.jsonl"
    manifest_payload = {
        "source_site": SOURCE_SITE,
        "source_name": SOURCE_NAME,
        "base_url": client.base_url,
        "imported_at": imported_at,
        "list_total": list_total,
        "requested_limit": limit,
        "query_categories": query_categories,
        "fetched_list_items": fetched_list_items,
        "imported_count": len(manifest_rows),
        "skipped_count": len(skipped_rows),
        "items": manifest_rows,
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    skipped_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in skipped_rows),
        encoding="utf-8",
    )
    return Mt180ImportSummary(
        source_site=SOURCE_SITE,
        source_name=SOURCE_NAME,
        base_url=client.base_url,
        imported_at=imported_at,
        list_total=list_total,
        requested_limit=limit,
        query_categories=query_categories,
        fetched_list_items=fetched_list_items,
        imported_count=len(manifest_rows),
        skipped_count=len(skipped_rows),
        output_dir=str(output_dir),
        manifest_path=str(manifest_path),
        skipped_path=str(skipped_path),
    )


def _fetch_batch(
    client: Mt180Client,
    batch: list[tuple[str, dict[str, Any]]],
    imported_at: str,
    workers: int,
) -> list[tuple[Mt180IndicatorRecord | None, Mt180SkippedRecord | None]]:
    if not batch:
        return []
    worker_count = max(1, min(workers, len(batch)))
    if worker_count == 1:
        return [
            _fetch_visible_record(client, indicator_id, item, imported_at)
            for indicator_id, item in batch
        ]

    def fetch_one(
        indicator_id: str,
        item: dict[str, Any],
    ) -> tuple[Mt180IndicatorRecord | None, Mt180SkippedRecord | None]:
        return _fetch_visible_record(client.clone(), indicator_id, item, imported_at)

    results: list[tuple[Mt180IndicatorRecord | None, Mt180SkippedRecord | None]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(fetch_one, indicator_id, item) for indicator_id, item in batch]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _fetch_visible_record(
    client: Mt180Client,
    indicator_id: str,
    list_item: dict[str, Any],
    imported_at: str,
) -> tuple[Mt180IndicatorRecord | None, Mt180SkippedRecord | None]:
    source_url = f"{client.base_url}/#/marketplace?indicator={indicator_id}"
    name = str(list_item.get("name") or "")
    try:
        detail = client.marketplace_detail(indicator_id)
    except requests.HTTPError as exc:
        status = str(exc.response.status_code if exc.response is not None else "http_error")
        return None, Mt180SkippedRecord(
            source_site=SOURCE_SITE,
            source_id=indicator_id,
            source_url=source_url,
            name=name,
            reason="detail_request_failed",
            status=status,
            imported_at=imported_at,
        )
    except Exception as exc:
        return None, Mt180SkippedRecord(
            source_site=SOURCE_SITE,
            source_id=indicator_id,
            source_url=source_url,
            name=name,
            reason=f"detail_error:{type(exc).__name__}",
            status="error",
            imported_at=imported_at,
        )

    formula = str(detail.get("formula") or "")
    can_view_formula = bool(detail.get("canViewFormula"))
    hide_formula = bool(detail.get("hideFormula"))
    if not can_view_formula or hide_formula or not formula.strip():
        reason = "formula_hidden" if hide_formula else "formula_not_visible"
        if can_view_formula and not formula.strip():
            reason = "empty_formula"
        return None, Mt180SkippedRecord(
            source_site=SOURCE_SITE,
            source_id=indicator_id,
            source_url=source_url,
            name=str(detail.get("name") or name),
            reason=reason,
            status="skipped",
            imported_at=imported_at,
        )

    category = str(detail.get("category") or list_item.get("category") or "other")
    indicator_type = int(detail.get("indicatorType") or list_item.get("indicatorType") or 0)
    formula_sha256 = hashlib.sha256(formula.encode("utf-8")).hexdigest()
    return Mt180IndicatorRecord(
        source_site=SOURCE_SITE,
        source_name=SOURCE_NAME,
        source_url=source_url,
        source_id=indicator_id,
        imported_at=imported_at,
        name=str(detail.get("name") or name),
        short_name=str(detail.get("shortName") or list_item.get("shortName") or ""),
        category=category,
        category_label=CATEGORY_LABELS.get(category, "其他"),
        indicator_type=indicator_type,
        indicator_type_label=INDICATOR_TYPE_LABELS.get(indicator_type, "未知"),
        description=str(detail.get("description") or ""),
        usage_guide=detail.get("usageGuide"),
        author_id=_optional_str(detail.get("authorId")),
        author_phone=_optional_str(detail.get("authorPhone")),
        author_nickname=_optional_str(detail.get("authorNickname")),
        sales_count=int(detail.get("salesCount") or 0),
        favorites_count=int(detail.get("favoritesCount") or 0),
        rating_avg=_optional_float(detail.get("ratingAvg")),
        rating_count=int(detail.get("ratingCount") or 0),
        tip_price=float(detail.get("tipPrice") or detail.get("tip_price") or 0),
        is_public=_optional_bool(detail.get("is_public")),
        hide_formula=hide_formula,
        can_view_formula=can_view_formula,
        can_apply=bool(detail.get("canApply")),
        formula_sha256=formula_sha256,
        formula=formula,
    ), None


def _write_indicator_record(
    record: Mt180IndicatorRecord,
    indicators_dir: Path,
    formulas_dir: Path,
) -> None:
    slug = indicator_slug(record)
    (indicators_dir / f"{slug}.json").write_text(
        json.dumps(asdict(record), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (formulas_dir / f"{slug}.tdx").write_text(record.formula, encoding="utf-8")


def _manifest_row(record: Mt180IndicatorRecord) -> dict[str, Any]:
    data = asdict(record)
    data.pop("formula")
    data["formula_file"] = f"formulas/{indicator_slug(record)}.tdx"
    data["record_file"] = f"indicators/{indicator_slug(record)}.json"
    return data


def indicator_slug(record: Mt180IndicatorRecord) -> str:
    safe_name = "".join(
        character if character.isalnum() else "_" for character in record.name.strip().lower()
    ).strip("_")
    safe_name = "_".join(part for part in safe_name.split("_") if part)[:48] or "indicator"
    return f"{safe_name}_{record.source_id}"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import mt180 marketplace indicators visible to the authenticated account."
    )
    parser.add_argument("--phone", default=os.getenv("MT180_PHONE"), help="mt180 login phone.")
    parser.add_argument(
        "--password",
        default=None,
        help="mt180 login password. Prefer --password-env for shell history hygiene.",
    )
    parser.add_argument(
        "--password-env",
        default="MT180_PASSWORD",
        help="Environment variable containing the mt180 password.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--sort", default="sales")
    parser.add_argument("--category", choices=sorted(CATEGORY_LABELS), default=None)
    parser.add_argument("--all-categories", action="store_true")
    parser.add_argument(
        "--indicator-type",
        type=int,
        choices=sorted(INDICATOR_TYPE_LABELS),
        default=None,
    )
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    password = args.password or os.getenv(args.password_env)
    if not args.phone:
        raise SystemExit("missing --phone or MT180_PHONE")
    if not password:
        raise SystemExit(f"missing --password or {args.password_env}")
    client = Mt180Client(base_url=args.base_url)
    client.login(args.phone, password)
    summary = import_visible_marketplace_factors(
        client=client,
        output_dir=args.output_dir,
        limit=args.limit,
        page_size=args.page_size,
        sort=args.sort,
        category=args.category,
        indicator_type=args.indicator_type,
        sleep_seconds=args.sleep_seconds,
        workers=args.workers,
        all_categories=args.all_categories,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
