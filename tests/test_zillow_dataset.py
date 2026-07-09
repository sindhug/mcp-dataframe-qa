from pathlib import Path

from mcp_dataframe_qa.config import load_config
from mcp_dataframe_qa.engine import DataFrameQA


def default_qa() -> DataFrameQA:
    return DataFrameQA.from_config(load_config("dataframe_qa.yaml"))


def test_default_dataset_is_public_zillow_market_data() -> None:
    dataset_path = Path("data/zillow_metro_market.csv")
    assert dataset_path.exists()
    assert dataset_path.stat().st_size < 10_000_000

    profile = default_qa().profile()
    assert profile["row_count"] > 50_000
    assert profile["column_count"] == 11
    assert "median_list_price" in profile["columns"]


def test_top_metros_by_median_list_price() -> None:
    result = default_qa().query("What are the top metros by median list price?")
    assert result.kind == "table"
    assert result.table is not None
    assert result.table.columns == ["region_name", "median_list_price"]
    assert len(result.table.rows) == 10


def test_count_market_months_with_large_inventory() -> None:
    result = default_qa().query("How many metro-months had more than 10,000 active listings?")
    assert result.kind == "scalar"
    assert isinstance(result.value, int)
    assert result.value > 0
