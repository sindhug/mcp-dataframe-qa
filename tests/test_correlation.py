import pandas as pd
import pytest

from mcp_dataframe_qa.engine import DataFrameQA
from mcp_dataframe_qa.schemas import AnalysisPlan, Metric


def _qa() -> DataFrameQA:
    frame = pd.DataFrame(
        {
            "g": ["a", "a", "a", "b", "b", "b"],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [2.0, 4.0, 6.0, 1.0, 2.0, 4.0],
            "label": ["p", "q", "r", "s", "t", "u"],
        }
    )
    return DataFrameQA.from_dataframe(frame)


def test_corr_computes_overall_pearson_correlation() -> None:
    qa = _qa()
    result = qa.execute_plan(
        AnalysisPlan(metrics=[Metric(fn="corr", column="x", column2="y", name="xy_corr")])
    )
    assert result.kind == "scalar"
    assert result.value == pytest.approx(-0.02913170599537296)


def test_corr_can_be_grouped() -> None:
    qa = _qa()
    result = qa.execute_plan(
        AnalysisPlan(
            group_by=["g"],
            metrics=[Metric(fn="corr", column="x", column2="y", name="xy_corr")],
        )
    )
    assert result.kind == "table"
    assert result.table is not None
    corr_by_group = {row["g"]: row["xy_corr"] for row in result.table.rows}
    assert corr_by_group["a"] == pytest.approx(1.0)
    assert corr_by_group["b"] == pytest.approx(0.981981, rel=1e-4)


def test_corr_default_output_name_includes_both_columns() -> None:
    qa = _qa()
    result = qa.execute_plan(AnalysisPlan(metrics=[Metric(fn="corr", column="x", column2="y")]))
    assert result.kind == "scalar"
    assert result.answer.startswith("corr x y is")


def test_corr_requires_column2() -> None:
    qa = _qa()
    result = qa.execute_plan(AnalysisPlan(metrics=[Metric(fn="corr", column="x")]))
    assert result.kind == "error"
    assert "requires column2" in result.answer


def test_corr_rejects_nonnumeric_column() -> None:
    qa = _qa()
    result = qa.execute_plan(
        AnalysisPlan(metrics=[Metric(fn="corr", column="label", column2="y")])
    )
    assert result.kind == "error"
    assert "requires numeric columns" in result.answer
