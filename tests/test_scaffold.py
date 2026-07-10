import json
from pathlib import Path

import yaml

from mcp_dataframe_qa.datasets import load_dataframe
from mcp_dataframe_qa.llm import LLMConfig, LLMPlanner
from mcp_dataframe_qa.scaffold import (
    build_starter_config,
    dataset_slug,
    describe_columns_with_llm,
    write_starter_config,
)

LISTINGS_CSV = Path(__file__).parent.parent / "data" / "listings.csv"


def test_dataset_slug_handles_spaces_and_punctuation() -> None:
    assert dataset_slug("~/Downloads/my data (2).csv") == "my_data_2"
    assert dataset_slug("data/listings.csv") == "listings"


def test_build_starter_config_covers_every_column() -> None:
    config = build_starter_config(str(LISTINGS_CSV))
    assert set(config["columns"]) == {
        "listing_id", "price", "bedrooms", "bathrooms",
        "sqft", "zip_code", "neighborhood", "status",
    }
    # Every entry starts blank and ready to annotate, never guessed.
    for column in config["columns"].values():
        assert column["description"] == ""
        assert column["synonyms"] == []


def test_build_starter_config_infers_reasonable_semantic_types() -> None:
    config = build_starter_config(str(LISTINGS_CSV))
    columns = config["columns"]
    assert columns["listing_id"]["semantic_type"] == "identifier"
    assert columns["price"]["semantic_type"] == "currency"
    assert columns["neighborhood"]["semantic_type"] == "dimension"
    assert columns["status"]["semantic_type"] == "dimension"
    # Ambiguous numeric columns are left blank rather than guessed.
    assert columns["bedrooms"]["semantic_type"] is None


def test_write_starter_config_produces_loadable_yaml(tmp_path) -> None:
    out_path = tmp_path / "generated.yaml"
    written = write_starter_config(str(LISTINGS_CSV), str(out_path))
    assert written == out_path

    with out_path.open() as handle:
        raw = yaml.safe_load(handle)
    assert raw["dataset"]["id"] == "listings"
    assert raw["dataset"]["path"] == str(LISTINGS_CSV)
    assert len(raw["columns"]) == 8


def test_describe_columns_with_llm_parses_batched_response(monkeypatch) -> None:
    def fake_complete(self: LLMPlanner, system: str, user: str) -> str:
        assert "price" in user
        return json.dumps(
            {
                "columns": {
                    "price": {
                        "description": "Listing price in US dollars.",
                        "semantic_type": "currency",
                        "synonyms": ["asking price"],
                    },
                    "status": {
                        "description": "Listing status such as active or sold.",
                        "semantic_type": "dimension",
                        "synonyms": [],
                    },
                }
            }
        )

    monkeypatch.setattr(LLMPlanner, "complete", fake_complete)
    frame = load_dataframe(str(LISTINGS_CSV))
    result = describe_columns_with_llm(
        frame, LLMConfig(provider="openai", api_key="test-key", model="test-model")
    )
    assert result["price"]["semantic_type"] == "currency"
    assert result["price"]["synonyms"] == ["asking price"]
    assert result["status"]["description"].startswith("Listing status")


def test_describe_columns_with_llm_drops_unknown_columns(monkeypatch) -> None:
    def fake_complete(self: LLMPlanner, system: str, user: str) -> str:
        return json.dumps(
            {
                "columns": {
                    "price": {"description": "Price.", "semantic_type": "currency"},
                    "not_a_real_column": {"description": "Invented.", "semantic_type": "dimension"},
                }
            }
        )

    monkeypatch.setattr(LLMPlanner, "complete", fake_complete)
    frame = load_dataframe(str(LISTINGS_CSV))
    result = describe_columns_with_llm(
        frame, LLMConfig(provider="openai", api_key="test-key", model="test-model")
    )
    assert "not_a_real_column" not in result
    assert "price" in result


def test_build_starter_config_uses_column_info_overlay() -> None:
    column_info = {
        "price": {
            "description": "Listing price in US dollars.",
            "semantic_type": "currency",
            "synonyms": ["asking price"],
        }
    }
    config = build_starter_config(str(LISTINGS_CSV), column_info=column_info)
    columns = config["columns"]
    assert columns["price"]["description"] == "Listing price in US dollars."
    assert columns["price"]["synonyms"] == ["asking price"]
    # Columns missing from the overlay still fall back to the heuristic guess.
    assert columns["neighborhood"]["semantic_type"] == "dimension"
    assert columns["neighborhood"]["description"] == ""


def test_write_starter_config_header_reflects_llm_usage(tmp_path) -> None:
    plain_path = tmp_path / "plain.yaml"
    write_starter_config(str(LISTINGS_CSV), str(plain_path))
    assert "blank on purpose" in plain_path.read_text()

    llm_path = tmp_path / "llm.yaml"
    write_starter_config(
        str(LISTINGS_CSV),
        str(llm_path),
        column_info={"price": {"description": "Price.", "semantic_type": "currency"}},
    )
    assert "LLM-drafted" in llm_path.read_text()
