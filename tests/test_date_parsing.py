import pandas as pd

from mcp_dataframe_qa.config import ColumnConfig
from mcp_dataframe_qa.datasets import parse_date_columns


def test_parse_date_columns_converts_configured_string_dates() -> None:
    frame = pd.DataFrame({"order_date": ["1/2/2020", "3/4/2021"], "price": [10, 20]})
    columns = {"order_date": ColumnConfig(semantic_type="date")}

    parsed = parse_date_columns(frame, columns)

    assert pd.api.types.is_datetime64_any_dtype(parsed["order_date"])
    assert pd.api.types.is_numeric_dtype(parsed["price"])


def test_parse_date_columns_ignores_columns_without_date_semantic_type() -> None:
    frame = pd.DataFrame({"note": ["1/2/2020", "not a date"]})
    columns = {"note": ColumnConfig(semantic_type="dimension")}

    parsed = parse_date_columns(frame, columns)

    assert parsed["note"].dtype == frame["note"].dtype


def test_parse_date_columns_warns_when_mostly_unparseable(capsys) -> None:
    frame = pd.DataFrame({"maybe_date": ["not a date", "also not a date", "nope", "1/2/2020"]})
    columns = {"maybe_date": ColumnConfig(semantic_type="date")}

    parse_date_columns(frame, columns)

    err = capsys.readouterr().err
    assert "maybe_date" in err
    assert "only" in err


def test_parse_date_columns_leaves_already_parsed_dates_alone() -> None:
    frame = pd.DataFrame({"order_date": pd.to_datetime(["2020-01-02", "2021-03-04"])})
    columns = {"order_date": ColumnConfig(semantic_type="date")}

    parsed = parse_date_columns(frame, columns)

    assert pd.api.types.is_datetime64_any_dtype(parsed["order_date"])
