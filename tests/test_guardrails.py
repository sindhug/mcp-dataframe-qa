from pathlib import Path

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
