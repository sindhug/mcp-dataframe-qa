import pandas as pd
import pytest

from mcp_dataframe_qa.engine import DataFrameQA
from mcp_dataframe_qa.schemas import AnalysisPlan, DerivedColumn, Expression, Metric


def _orders_qa() -> DataFrameQA:
    frame = pd.DataFrame(
        {
            "order_date": pd.to_datetime(["2020-01-15", "2020-06-01", "2021-02-20"]),
            "ship_date": pd.to_datetime(["2020-01-18", "2020-06-04", "2021-02-21"]),
            "amount": [10.0, 20.0, 30.0],
        }
    )
    return DataFrameQA.from_dataframe(frame)


def test_month_of_extracts_calendar_month() -> None:
    qa = _orders_qa()
    result = qa.execute_plan(
        AnalysisPlan(
            derive=[
                DerivedColumn(
                    name="order_month",
                    expr=Expression(
                        op="month_of", left=Expression(op="column", column="order_date")
                    ),
                )
            ],
            group_by=["order_month"],
            metrics=[Metric(fn="sum", column="amount", name="total")],
        )
    )
    assert result.kind == "table"
    assert result.table is not None
    months = {row["order_month"] for row in result.table.rows}
    assert months == {1, 6, 2}


def test_year_of_extracts_calendar_year() -> None:
    qa = _orders_qa()
    result = qa.execute_plan(
        AnalysisPlan(
            derive=[
                DerivedColumn(
                    name="order_year",
                    expr=Expression(
                        op="year_of", left=Expression(op="column", column="order_date")
                    ),
                )
            ],
            group_by=["order_year"],
            metrics=[Metric(fn="count", column="*", name="n")],
        )
    )
    assert result.kind == "table"
    assert result.table is not None
    years = {row["order_year"] for row in result.table.rows}
    assert years == {2020, 2021}


def test_date_diff_computes_days_between_two_date_columns() -> None:
    qa = _orders_qa()
    result = qa.execute_plan(
        AnalysisPlan(
            derive=[
                DerivedColumn(
                    name="ship_time_days",
                    expr=Expression(
                        op="date_diff",
                        left=Expression(op="column", column="ship_date"),
                        right=Expression(op="column", column="order_date"),
                    ),
                )
            ],
            metrics=[Metric(fn="avg", column="ship_time_days", name="avg_ship_days")],
        )
    )
    assert result.kind == "scalar"
    # (3 + 3 + 1) / 3
    assert result.value == pytest.approx(7 / 3)


def test_date_diff_rejects_nondate_operand() -> None:
    qa = _orders_qa()
    result = qa.execute_plan(
        AnalysisPlan(
            derive=[
                DerivedColumn(
                    name="bad_diff",
                    expr=Expression(
                        op="date_diff",
                        left=Expression(op="column", column="amount"),
                        right=Expression(op="column", column="order_date"),
                    ),
                )
            ],
            metrics=[Metric(fn="avg", column="bad_diff", name="avg_bad_diff")],
        )
    )
    assert result.kind == "error"
    assert "requires a date operand" in result.answer


def test_month_of_rejects_nondate_operand() -> None:
    qa = _orders_qa()
    result = qa.execute_plan(
        AnalysisPlan(
            derive=[
                DerivedColumn(
                    name="bad_month",
                    expr=Expression(op="month_of", left=Expression(op="column", column="amount")),
                )
            ],
            group_by=["bad_month"],
            metrics=[Metric(fn="count", column="*", name="n")],
        )
    )
    assert result.kind == "error"
    assert "requires a date operand" in result.answer
