from pathlib import Path

import yaml

from mcp_dataframe_qa.scaffold import build_starter_config, dataset_slug, write_starter_config

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
