from pathlib import Path

import pytest

from mcp_dataframe_qa.config import load_config
from mcp_dataframe_qa.engine import DataFrameQA
from mcp_dataframe_qa.schemas import AnalysisPlan, DerivedColumn, Expression, Metric, SortSpec

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


def test_derived_ratio_can_be_aggregated() -> None:
    qa = DataFrameQA.from_config(load_config(str(LISTINGS_CONFIG)))
    result = qa.execute_plan(
        AnalysisPlan(
            derive=[
                DerivedColumn(
                    name="price_per_sqft",
                    expr=Expression(
                        op="divide",
                        left=Expression(op="column", column="price"),
                        right=Expression(op="column", column="sqft"),
                    ),
                )
            ],
            group_by=["neighborhood"],
            metrics=[
                Metric(
                    fn="median",
                    column="price_per_sqft",
                    name="median_price_per_sqft",
                )
            ],
            sort=[SortSpec(column="median_price_per_sqft", direction="desc")],
            limit=1,
        )
    )

    assert result.kind == "table"
    assert result.table is not None
    assert result.table.rows == [
        {
            "neighborhood": "SOMA",
            "median_price_per_sqft": 785.1973684210527,
        }
    ]


def test_derived_expression_rejects_unknown_column() -> None:
    qa = DataFrameQA.from_config(load_config(str(LISTINGS_CONFIG)))
    result = qa.execute_plan(
        AnalysisPlan(
            derive=[
                DerivedColumn(
                    name="bad_ratio",
                    expr=Expression(
                        op="divide",
                        left=Expression(op="column", column="missing_price"),
                        right=Expression(op="column", column="sqft"),
                    ),
                )
            ],
            metrics=[Metric(fn="avg", column="bad_ratio", name="avg_bad_ratio")],
        )
    )

    assert result.kind == "error"
    assert "Unknown expression column 'missing_price'" in result.answer


def test_derived_expression_rejects_nonnumeric_arithmetic() -> None:
    qa = DataFrameQA.from_config(load_config(str(LISTINGS_CONFIG)))
    result = qa.execute_plan(
        AnalysisPlan(
            derive=[
                DerivedColumn(
                    name="bad_ratio",
                    expr=Expression(
                        op="divide",
                        left=Expression(op="column", column="neighborhood"),
                        right=Expression(op="column", column="sqft"),
                    ),
                )
            ],
            metrics=[Metric(fn="avg", column="bad_ratio", name="avg_bad_ratio")],
        )
    )

    assert result.kind == "error"
    assert "requires numeric operands" in result.answer


def test_derived_comparison_indicator_gives_rate() -> None:
    qa = DataFrameQA.from_config(load_config(str(LISTINGS_CONFIG)))
    result = qa.execute_plan(
        AnalysisPlan(
            derive=[
                DerivedColumn(
                    name="is_sold",
                    expr=Expression(
                        op="==",
                        left=Expression(op="column", column="status"),
                        right=Expression(op="literal", value="sold"),
                    ),
                )
            ],
            metrics=[Metric(fn="avg", column="is_sold", name="sold_rate")],
        )
    )

    assert result.kind == "scalar"
    assert result.value == pytest.approx(1 / 12)


def test_derived_comparison_indicator_can_be_grouped() -> None:
    qa = DataFrameQA.from_config(load_config(str(LISTINGS_CONFIG)))
    result = qa.execute_plan(
        AnalysisPlan(
            derive=[
                DerivedColumn(
                    name="is_active",
                    expr=Expression(
                        op="==",
                        left=Expression(op="column", column="status"),
                        right=Expression(op="literal", value="active"),
                    ),
                )
            ],
            group_by=["neighborhood"],
            metrics=[Metric(fn="avg", column="is_active", name="active_rate")],
            sort=[SortSpec(column="neighborhood", direction="asc")],
        )
    )

    assert result.kind == "table"
    assert result.table is not None
    neighborhoods = [row["neighborhood"] for row in result.table.rows]
    assert neighborhoods == sorted(neighborhoods)


def test_comparison_expression_rejects_missing_operand() -> None:
    qa = DataFrameQA.from_config(load_config(str(LISTINGS_CONFIG)))
    result = qa.execute_plan(
        AnalysisPlan(
            derive=[
                DerivedColumn(
                    name="bad_flag",
                    expr=Expression(op="==", left=Expression(op="column", column="status")),
                )
            ],
            metrics=[Metric(fn="avg", column="bad_flag", name="avg_bad_flag")],
        )
    )

    assert result.kind == "error"
    assert "requires left and right" in result.answer


def _is_status(value: str) -> Expression:
    return Expression(
        op="==",
        left=Expression(op="column", column="status"),
        right=Expression(op="literal", value=value),
    )


def _is_neighborhood(value: str) -> Expression:
    return Expression(
        op="==",
        left=Expression(op="column", column="neighborhood"),
        right=Expression(op="literal", value=value),
    )


def test_derived_and_combines_two_comparisons() -> None:
    qa = DataFrameQA.from_config(load_config(str(LISTINGS_CONFIG)))
    result = qa.execute_plan(
        AnalysisPlan(
            derive=[
                DerivedColumn(
                    name="active_in_mission",
                    expr=Expression(
                        op="and", left=_is_status("active"), right=_is_neighborhood("Mission")
                    ),
                )
            ],
            metrics=[Metric(fn="sum", column="active_in_mission", name="count")],
        )
    )

    assert result.kind == "scalar"
    assert result.value == 2


def test_derived_or_combines_two_comparisons() -> None:
    qa = DataFrameQA.from_config(load_config(str(LISTINGS_CONFIG)))
    result = qa.execute_plan(
        AnalysisPlan(
            derive=[
                DerivedColumn(
                    name="active_or_mission",
                    expr=Expression(
                        op="or", left=_is_status("active"), right=_is_neighborhood("Mission")
                    ),
                )
            ],
            metrics=[Metric(fn="sum", column="active_or_mission", name="count")],
        )
    )

    assert result.kind == "scalar"
    assert result.value == 10


def test_derived_not_negates_a_comparison() -> None:
    qa = DataFrameQA.from_config(load_config(str(LISTINGS_CONFIG)))
    result = qa.execute_plan(
        AnalysisPlan(
            derive=[
                DerivedColumn(
                    name="not_active", expr=Expression(op="not", left=_is_status("active"))
                )
            ],
            metrics=[Metric(fn="sum", column="not_active", name="count")],
        )
    )

    assert result.kind == "scalar"
    assert result.value == 3


def test_logical_and_rejects_nonboolean_operand() -> None:
    qa = DataFrameQA.from_config(load_config(str(LISTINGS_CONFIG)))
    result = qa.execute_plan(
        AnalysisPlan(
            derive=[
                DerivedColumn(
                    name="bad_and",
                    expr=Expression(
                        op="and",
                        left=Expression(op="column", column="price"),
                        right=_is_status("active"),
                    ),
                )
            ],
            metrics=[Metric(fn="sum", column="bad_and", name="count")],
        )
    )

    assert result.kind == "error"
    assert "requires boolean operands" in result.answer


def test_logical_not_rejects_right_field() -> None:
    qa = DataFrameQA.from_config(load_config(str(LISTINGS_CONFIG)))
    result = qa.execute_plan(
        AnalysisPlan(
            derive=[
                DerivedColumn(
                    name="bad_not",
                    expr=Expression(op="not", left=_is_status("active"), right=_is_status("sold")),
                )
            ],
            metrics=[Metric(fn="sum", column="bad_not", name="count")],
        )
    )

    assert result.kind == "error"
    assert "does not allow field" in result.answer
