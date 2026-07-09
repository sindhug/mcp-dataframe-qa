from pathlib import Path

from mcp_dataframe_qa.config import load_config
from mcp_dataframe_qa.engine import DataFrameQA

LISTINGS_CONFIG = Path(__file__).parent / "fixtures" / "listings_config.yaml"


def qa() -> DataFrameQA:
    return DataFrameQA.from_config(load_config(str(LISTINGS_CONFIG)))


def test_count_under_one_million() -> None:
    result = qa().query("How many houses are under $1M?")
    assert result.kind == "scalar"
    assert result.value == 7


def test_average_price_by_bedroom_count() -> None:
    result = qa().query("Show average price by bedroom count.")
    assert result.kind == "table"
    assert result.table is not None
    rows = {row["bedrooms"]: row["avg_price"] for row in result.table.rows}
    assert round(rows[1], 2) == 595000.00
    assert round(rows[4], 2) == 1350000.00


def test_count_three_plus_bedrooms_and_sqft() -> None:
    result = qa().query("How many listings have 3+ bedrooms and 2,000+ sqft?")
    assert result.kind == "scalar"
    assert result.value == 5


def test_top_zip_codes_by_median_price() -> None:
    result = qa().query("What are the top ZIP codes by median price?")
    assert result.kind == "table"
    assert result.table is not None
    assert result.table.rows[0]["zip_code"] == 94123
    assert result.table.rows[0]["median_price"] == 1890000.0
