import pandas as pd
import pytest

from mcp_dataframe_qa.engine import DataFrameQA
from mcp_dataframe_qa.schemas import (
    AnalysisPlan,
    DerivedColumn,
    Expression,
    Metric,
    Regroup,
    SortSpec,
)


def _rainfall_qa() -> DataFrameQA:
    frame = pd.DataFrame(
        {
            "location": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "year": [2020, 2020, 2021, 2021, 2020, 2020, 2021, 2021],
            "rainfall": [10.0, 20.0, 5.0, 15.0, 100.0, 100.0, 50.0, 50.0],
        }
    )
    return DataFrameQA.from_dataframe(frame)


def _elo_qa() -> DataFrameQA:
    frame = pd.DataFrame(
        {
            "team": ["X", "X", "X", "X", "Y", "Y", "Y", "Y"],
            "season": [2020, 2020, 2021, 2021, 2020, 2020, 2021, 2021],
            "elo": [1500.0, 1600.0, 1550.0, 1560.0, 1500.0, 1520.0, 1510.0, 1700.0],
        }
    )
    return DataFrameQA.from_dataframe(frame)


def test_regroup_computes_average_of_yearly_sums() -> None:
    qa = _rainfall_qa()
    result = qa.execute_plan(
        AnalysisPlan(
            group_by=["location", "year"],
            metrics=[Metric(fn="sum", column="rainfall", name="yearly_total")],
            regroup=Regroup(
                group_by=["location"],
                metrics=[Metric(fn="avg", column="yearly_total", name="avg_annual")],
                sort=[SortSpec(column="avg_annual", direction="desc")],
            ),
        )
    )

    assert result.kind == "table"
    assert result.table is not None
    by_location = {row["location"]: row["avg_annual"] for row in result.table.rows}
    assert by_location["A"] == pytest.approx(25.0)
    assert by_location["B"] == pytest.approx(150.0)
    assert result.table.rows[0]["location"] == "B"


def test_regroup_without_group_by_derives_sorts_and_limits() -> None:
    qa = _elo_qa()
    result = qa.execute_plan(
        AnalysisPlan(
            group_by=["team", "season"],
            metrics=[
                Metric(fn="max", column="elo", name="max_elo"),
                Metric(fn="min", column="elo", name="min_elo"),
            ],
            regroup=Regroup(
                derive=[
                    DerivedColumn(
                        name="swing",
                        expr=Expression(
                            op="subtract",
                            left=Expression(op="column", column="max_elo"),
                            right=Expression(op="column", column="min_elo"),
                        ),
                    )
                ],
                sort=[SortSpec(column="swing", direction="desc")],
                limit=1,
            ),
        )
    )

    assert result.kind == "table"
    assert result.table is not None
    assert len(result.table.rows) == 1
    top = result.table.rows[0]
    assert top["team"] == "Y"
    assert top["season"] == 2021
    assert top["swing"] == pytest.approx(190.0)


def test_regroup_requires_outer_group_by() -> None:
    qa = _rainfall_qa()
    result = qa.execute_plan(
        AnalysisPlan(
            metrics=[Metric(fn="sum", column="rainfall", name="total")],
            regroup=Regroup(group_by=["location"], metrics=[Metric(fn="avg", column="total")]),
        )
    )
    assert result.kind == "error"
    assert "regroup requires the plan to have a group_by" in result.answer


def test_regroup_group_by_requires_metrics() -> None:
    qa = _rainfall_qa()
    result = qa.execute_plan(
        AnalysisPlan(
            group_by=["location", "year"],
            metrics=[Metric(fn="sum", column="rainfall", name="yearly_total")],
            regroup=Regroup(group_by=["location"]),
        )
    )
    assert result.kind == "error"
    assert "regroup.metrics is required" in result.answer


def test_regroup_metrics_require_group_by() -> None:
    qa = _rainfall_qa()
    result = qa.execute_plan(
        AnalysisPlan(
            group_by=["location", "year"],
            metrics=[Metric(fn="sum", column="rainfall", name="yearly_total")],
            regroup=Regroup(metrics=[Metric(fn="avg", column="yearly_total")]),
        )
    )
    assert result.kind == "error"
    assert "regroup.metrics requires regroup.group_by" in result.answer


def test_regroup_rejects_unknown_group_by_column() -> None:
    qa = _rainfall_qa()
    result = qa.execute_plan(
        AnalysisPlan(
            group_by=["location", "year"],
            metrics=[Metric(fn="sum", column="rainfall", name="yearly_total")],
            regroup=Regroup(
                group_by=["not_a_column"],
                metrics=[Metric(fn="avg", column="yearly_total")],
            ),
        )
    )
    assert result.kind == "error"
    assert "Unknown regroup group_by column" in result.answer


def test_regroup_sort_rejects_unknown_column() -> None:
    qa = _rainfall_qa()
    result = qa.execute_plan(
        AnalysisPlan(
            group_by=["location", "year"],
            metrics=[Metric(fn="sum", column="rainfall", name="yearly_total")],
            regroup=Regroup(sort=[SortSpec(column="not_a_column", direction="desc")]),
        )
    )
    assert result.kind == "error"
    assert "regroup sort column" in result.answer
