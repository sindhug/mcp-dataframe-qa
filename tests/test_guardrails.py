from pathlib import Path

from mcp_dataframe_qa.config import load_config
from mcp_dataframe_qa.engine import DataFrameQA
from mcp_dataframe_qa.schemas import AnalysisPlan, Metric

LISTINGS_CONFIG = Path(__file__).parent / "fixtures" / "listings_config.yaml"


def test_unknown_column_returns_error() -> None:
    qa = DataFrameQA.from_config(load_config(str(LISTINGS_CONFIG)))
    result = qa.execute_plan(
        AnalysisPlan(metrics=[Metric(fn="avg", column="does_not_exist", name="avg_bad")])
    )
    assert result.kind == "error"
    assert "Unknown metric column" in result.answer


def test_limit_is_capped() -> None:
    qa = DataFrameQA.from_config(load_config(str(LISTINGS_CONFIG)))
    result = qa.query("What are the top 500 ZIP codes by median price?")
    assert result.table is not None
    assert len(result.table.rows) <= 100
