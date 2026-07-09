import pandas as pd

from mcp_dataframe_qa.config import ColumnConfig
from mcp_dataframe_qa.datasets import warn_column_mismatches


def test_no_warning_when_columns_match_exactly(capsys) -> None:
    frame = pd.DataFrame({"price": [1, 2], "city": ["a", "b"]})
    columns = {
        "price": ColumnConfig(description="price"),
        "city": ColumnConfig(description="city"),
    }
    warn_column_mismatches(frame, columns)
    assert capsys.readouterr().err == ""


def test_no_warning_when_no_columns_configured(capsys) -> None:
    frame = pd.DataFrame({"price": [1, 2]})
    warn_column_mismatches(frame, {})
    assert capsys.readouterr().err == ""


def test_warns_about_configured_columns_missing_from_data(capsys) -> None:
    frame = pd.DataFrame({"price": [1, 2]})
    columns = {"price": ColumnConfig(), "median_list_price": ColumnConfig()}
    warn_column_mismatches(frame, columns)
    err = capsys.readouterr().err
    assert "median_list_price" in err
    assert "not present in this dataset" in err


def test_warns_about_undescribed_data_columns(capsys) -> None:
    frame = pd.DataFrame({"price": [1, 2], "city": ["a", "b"]})
    columns = {"price": ColumnConfig()}
    warn_column_mismatches(frame, columns)
    err = capsys.readouterr().err
    assert "city" in err
    assert "no description in the config" in err
