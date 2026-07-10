from collections.abc import Mapping
from typing import Any

import pandas as pd

from mcp_dataframe_qa.config import ColumnConfig


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _truncate(value: Any, max_chars: int) -> Any:
    value = _json_safe(value)
    if isinstance(value, str) and len(value) > max_chars:
        return value[: max_chars - 3] + "..."
    return value


def profile_dataframe(
    frame: pd.DataFrame,
    dataset_id: str,
    table_name: str,
    columns: Mapping[str, ColumnConfig],
    max_examples: int = 5,
    max_cell_chars: int = 120,
) -> dict[str, Any]:
    column_profiles: dict[str, Any] = {}
    for name in frame.columns:
        series = frame[name]
        configured = columns.get(name, ColumnConfig())
        dtype = str(series.dtype)
        profile: dict[str, Any] = {
            "name": name,
            "dtype": dtype,
            "description": configured.description,
            "semantic_type": configured.semantic_type,
            "synonyms": configured.synonyms,
            "delimiter": configured.delimiter,
            "null_count": int(series.isna().sum()),
        }

        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if not non_null.empty:
                profile["stats"] = {
                    "min": _json_safe(non_null.min()),
                    "max": _json_safe(non_null.max()),
                    "mean": _json_safe(non_null.mean()),
                    "median": _json_safe(non_null.median()),
                }
        else:
            counts = series.dropna().astype(str).value_counts().head(10)
            profile["top_values"] = [
                {"value": _truncate(index, max_cell_chars), "count": int(count)}
                for index, count in counts.items()
            ]

        column_profiles[name] = profile

    examples: list[dict[str, Any]] = []
    for row in frame.head(max_examples).to_dict(orient="records"):
        examples.append({key: _truncate(value, max_cell_chars) for key, value in row.items()})

    return {
        "dataset_id": dataset_id,
        "table_name": table_name,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": column_profiles,
        "examples": examples,
    }
