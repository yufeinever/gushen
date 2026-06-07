from __future__ import annotations

import json

from gushen.mt180_factor_import import Mt180Client, import_visible_marketplace_factors


class FakeMt180Client(Mt180Client):
    def __init__(self) -> None:
        self.base_url = "https://web.mt180.test"
        self.timeout = 30.0

    def clone(self):
        return self

    def marketplace_page(self, page, page_size, sort, category=None, indicator_type=None):
        assert page == 1
        assert page_size == 10
        assert sort == "sales"
        assert category is None
        assert indicator_type is None
        return {
            "total": 2,
            "list": [
                {"id": "visible-1", "name": "可见公式", "category": "trend", "indicatorType": 2},
                {"id": "hidden-1", "name": "隐藏公式", "category": "trend", "indicatorType": 2},
            ],
        }

    def marketplace_detail(self, indicator_id):
        if indicator_id == "visible-1":
            return {
                "id": "visible-1",
                "name": "可见公式",
                "shortName": "可见",
                "category": "trend",
                "indicatorType": 2,
                "description": "公开可见样例",
                "formula": "MA1:MA(CLOSE,5);",
                "hideFormula": False,
                "canViewFormula": True,
                "canApply": True,
                "salesCount": 10,
                "favoritesCount": 3,
                "ratingAvg": "4.5",
                "ratingCount": 2,
                "tipPrice": 0,
                "authorId": "author-1",
                "authorPhone": "13800000000",
                "authorNickname": "作者",
            }
        return {
            "id": "hidden-1",
            "name": "隐藏公式",
            "formula": "",
            "hideFormula": True,
            "canViewFormula": False,
            "canApply": True,
        }


def test_import_visible_marketplace_factors_writes_source_metadata(tmp_path) -> None:
    summary = import_visible_marketplace_factors(
        client=FakeMt180Client(),
        output_dir=tmp_path,
        limit=2,
        page_size=10,
        sleep_seconds=0,
    )

    assert summary.imported_count == 1
    assert summary.skipped_count == 1

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    item = manifest["items"][0]
    assert item["source_site"] == "mt180"
    assert item["source_id"] == "visible-1"
    assert item["category_label"] == "趋势"
    assert item["indicator_type_label"] == "副图"
    assert item["formula_sha256"]
    assert "formula" not in item

    record = json.loads((tmp_path / item["record_file"]).read_text(encoding="utf-8"))
    assert record["formula"] == "MA1:MA(CLOSE,5);"
    assert (tmp_path / item["formula_file"]).read_text(encoding="utf-8") == "MA1:MA(CLOSE,5);"

    skipped = (tmp_path / "skipped.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(skipped) == 1
    assert json.loads(skipped[0])["reason"] == "formula_hidden"


class FakeCategorizedMt180Client(Mt180Client):
    def __init__(self) -> None:
        self.base_url = "https://web.mt180.test"
        self.timeout = 30.0
        self.calls = []

    def clone(self):
        return self

    def marketplace_page(self, page, page_size, sort, category=None, indicator_type=None):
        assert page == 1
        assert page_size == 10
        assert sort == "sales"
        assert indicator_type is None
        self.calls.append(category)
        items_by_category = {
            None: [
                {"id": "shared-1", "name": "重复公式", "category": "trend", "indicatorType": 2},
                {"id": "global-1", "name": "全局公式", "category": "other", "indicatorType": 2},
            ],
            "trend": [
                {"id": "shared-1", "name": "重复公式", "category": "trend", "indicatorType": 2},
                {"id": "trend-1", "name": "趋势公式", "category": "trend", "indicatorType": 1},
            ],
            "oscillator": [
                {
                    "id": "shared-1",
                    "name": "重复公式",
                    "category": "oscillator",
                    "indicatorType": 2,
                },
                {"id": "osc-1", "name": "摆动公式", "category": "oscillator", "indicatorType": 2},
            ],
        }
        page_items = items_by_category.get(category, [])
        return {"total": len(page_items), "list": page_items}

    def marketplace_detail(self, indicator_id):
        return {
            "id": indicator_id,
            "name": indicator_id,
            "shortName": indicator_id,
            "category": "trend",
            "indicatorType": 2,
            "description": "公开可见样例",
            "formula": f"{indicator_id}:CLOSE;",
            "hideFormula": False,
            "canViewFormula": True,
            "canApply": True,
            "salesCount": 0,
            "favoritesCount": 0,
            "ratingCount": 0,
            "tipPrice": 0,
        }


def test_import_all_categories_deduplicates_ids_and_records_queries(tmp_path) -> None:
    client = FakeCategorizedMt180Client()
    summary = import_visible_marketplace_factors(
        client=client,
        output_dir=tmp_path,
        limit=None,
        page_size=10,
        sleep_seconds=0,
        all_categories=True,
    )

    assert client.calls == [
        None,
        "trend",
        "oscillator",
        "volume",
        "selection",
        "pattern",
        "moneyflow",
        "other",
    ]
    assert summary.query_categories == client.calls
    assert summary.list_total == 6
    assert summary.fetched_list_items == 4
    assert summary.imported_count == 4

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["query_categories"] == client.calls
    assert {item["source_id"] for item in manifest["items"]} == {
        "shared-1", "global-1", "trend-1", "osc-1"}
