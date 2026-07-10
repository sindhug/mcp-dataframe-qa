import pandas as pd
import pytest

from mcp_dataframe_qa.config import ColumnConfig
from mcp_dataframe_qa.engine import DataFrameQA
from mcp_dataframe_qa.schemas import AnalysisPlan, Metric, SortSpec


def _movies_qa() -> DataFrameQA:
    frame = pd.DataFrame(
        {
            "title": ["Jurassic World", "Mad Max", "Her"],
            "genres": ["Action|Adventure", "Action|Thriller", "Drama"],
            "rating": [6.5, 7.1, 8.0],
        }
    )
    return DataFrameQA.from_dataframe(frame, columns={"genres": ColumnConfig(delimiter="|")})


def test_explode_splits_delimited_column_into_one_row_per_tag() -> None:
    qa = _movies_qa()
    result = qa.execute_plan(
        AnalysisPlan(
            explode=["genres"],
            group_by=["genres"],
            metrics=[Metric(fn="avg", column="rating", name="avg_rating")],
            sort=[SortSpec(column="genres", direction="asc")],
        )
    )

    assert result.kind == "table"
    assert result.table is not None
    rows = {row["genres"]: row for row in result.table.rows}
    assert set(rows) == {"Action", "Adventure", "Thriller", "Drama"}
    assert rows["Action"]["row_count"] == 2
    assert rows["Action"]["avg_rating"] == pytest.approx(6.8)
    assert rows["Drama"]["row_count"] == 1
    assert rows["Drama"]["avg_rating"] == pytest.approx(8.0)


def test_without_explode_genres_group_by_whole_combination() -> None:
    qa = _movies_qa()
    result = qa.execute_plan(
        AnalysisPlan(
            group_by=["genres"],
            metrics=[Metric(fn="avg", column="rating", name="avg_rating")],
        )
    )

    assert result.kind == "table"
    assert result.table is not None
    values = {row["genres"] for row in result.table.rows}
    # Without explode, "Action|Adventure" is one category, not two.
    assert "Action|Adventure" in values
    assert "Action" not in values


def test_explode_rejects_column_without_configured_delimiter() -> None:
    qa = _movies_qa()
    result = qa.execute_plan(
        AnalysisPlan(
            explode=["title"],
            group_by=["title"],
            metrics=[Metric(fn="count", column="*", name="n")],
        )
    )

    assert result.kind == "error"
    assert "no delimiter is configured" in result.answer


def test_explode_rejects_unknown_column() -> None:
    qa = _movies_qa()
    result = qa.execute_plan(
        AnalysisPlan(
            explode=["not_a_column"],
            metrics=[Metric(fn="count", column="*", name="n")],
        )
    )

    assert result.kind == "error"
    assert "Unknown explode column" in result.answer
