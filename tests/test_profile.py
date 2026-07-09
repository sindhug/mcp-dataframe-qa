from pathlib import Path

from mcp_dataframe_qa.config import load_config
from mcp_dataframe_qa.engine import DataFrameQA

LISTINGS_CONFIG = Path(__file__).parent / "fixtures" / "listings_config.yaml"


def test_profile_exposes_schema_not_full_dataset() -> None:
    qa = DataFrameQA.from_config(load_config(str(LISTINGS_CONFIG)))
    profile = qa.profile()
    assert profile["row_count"] == 12
    assert profile["column_count"] == 8
    assert "price" in profile["columns"]
    assert len(profile["examples"]) == 5
