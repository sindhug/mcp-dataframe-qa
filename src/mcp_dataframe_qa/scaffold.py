"""Generate a starter YAML config for a new dataset.

The config format needs every column name to match the dataframe exactly, and
writing that by hand is exactly the kind of transcription step that produces
silent typos. This module reads the real dataframe once and writes a config
with every column already present and correctly named. Descriptions,
semantic types, and synonyms are drafted by an LLM that reads the actual
column names and sample values (see ``describe_columns_with_llm``), falling
back to a conservative name-based guess when no LLM provider is configured.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from mcp_dataframe_qa.datasets import load_dataframe
from mcp_dataframe_qa.llm import LLMConfig, LLMPlanner, LLMResponseError, extract_json_object

_IDENTIFIER_HINTS = ("_id", "identifier", "code", "sku")
_CURRENCY_HINTS = ("price", "cost", "revenue", "profit", "sales", "amount", "fee", "fare")
_COUNT_HINTS = ("count", "quantity", "qty", "number", "num", "total")
_DELIMITER_CANDIDATES = ("|", ";")
_MAX_SAMPLE_VALUES = 5


def dataset_slug(data_path: str) -> str:
    """Turn a file path into a short, config-friendly identifier.

    ``~/Downloads/my data (2).csv`` becomes ``my_data_2``. Used both as the
    default dataset id/table name and as the default output filename, so a
    generated config is traceable back to the file it came from.
    """
    stem = Path(data_path).stem
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in stem)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "dataset"


def _infer_semantic_type(column: str, series: pd.Series) -> str | None:
    """Guess a semantic type from the column name and dtype.

    This is deliberately conservative. A wrong guess is worse than no guess,
    since it would mislead the planner instead of just leaving it uninformed.
    Numeric columns with no recognizable name pattern are left blank rather
    than guessed, since "count" versus "currency" versus "measurement" isn't
    reliably inferable from dtype alone. Used only as a fallback when no LLM
    provider is configured; a substring match here (for example "count"
    inside "Country" or "Discount") can misfire, which real column values
    would have caught.
    """
    lowered = column.lower()
    if lowered == "id" or lowered.endswith(_IDENTIFIER_HINTS) or "identifier" in lowered:
        return "identifier"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if any(hint in lowered for hint in _CURRENCY_HINTS):
        return "currency"
    if any(hint in lowered for hint in _COUNT_HINTS):
        return "count"
    if pd.api.types.is_numeric_dtype(series):
        return None
    return "dimension"


def _infer_delimiter(series: pd.Series) -> str | None:
    """Detect whether a string column stores a delimited multi-value tag list.

    Conservative on purpose, same reasoning as _infer_semantic_type: only fires
    when a candidate delimiter appears in most non-null values, so a column
    that merely contains an occasional "|" isn't mistaken for a tag list like
    genres ("Action|Adventure|Thriller").
    """
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return None
    samples = series.dropna().astype(str)
    if samples.empty:
        return None
    for delimiter in _DELIMITER_CANDIDATES:
        share = (samples.str.count(re.escape(delimiter)) > 0).mean()
        if share > 0.5:
            return delimiter
    return None


def _sample_values(series: pd.Series) -> list[str]:
    values = series.dropna().astype(str).unique()[:_MAX_SAMPLE_VALUES].tolist()
    return [value[:80] for value in values]


_DESCRIBE_SYSTEM_PROMPT = (
    "You are annotating a dataframe schema for a natural-language query tool. "
    "For each column you are given its name, pandas dtype, and a handful of real "
    "sample values. Return only a JSON object of the shape "
    '{"columns": {"<column name>": {"description": "...", "semantic_type": "...", '
    '"synonyms": ["...", "..."], "delimiter": null}}}, with one entry per input column, '
    "in the same order. "
    "description: one factual sentence about what the column holds. If a value like 0 "
    "or a blank string looks like a placeholder for missing data rather than a real "
    "measurement, say so. "
    "semantic_type: a short lowercase label such as identifier, dimension, measurement, "
    "currency, count, rate, percentage, date, year, rank, or tag_list. "
    "delimiter: if the sample values look like several tags joined into one string, for "
    'example "Action|Adventure|Thriller", set delimiter to that separator character (here '
    '"|") and semantic_type to tag_list. Otherwise set delimiter to null. '
    "synonyms: 0-4 short alternate phrases a user might type to refer to this column. "
    "Base every judgment on the actual sample values, not just the column name. "
    "Do not include markdown or prose outside the JSON object."
)


def describe_columns_with_llm(frame: pd.DataFrame, llm_config: LLMConfig) -> dict[str, dict]:
    """Ask an LLM to draft description/semantic_type/synonyms for every column.

    Sends one batched call covering every column (not one call per column) so
    scaffolding a wide dataset stays a single request. Columns the model
    omits or invents outside the dataframe are dropped; callers should treat
    the result as a best-effort overlay and fall back to blanks for any
    column missing from it.
    """
    columns_payload = [
        {
            "name": str(column),
            "dtype": str(frame[column].dtype),
            "sample_values": _sample_values(frame[column]),
        }
        for column in frame.columns
    ]
    user = json.dumps({"columns": columns_payload}, sort_keys=True)
    text = LLMPlanner(llm_config).complete(_DESCRIBE_SYSTEM_PROMPT, user)
    payload = extract_json_object(text)
    described = payload.get("columns")
    if not isinstance(described, dict):
        raise LLMResponseError("Model response did not contain a 'columns' object.")

    known_columns = {str(column) for column in frame.columns}
    result: dict[str, dict] = {}
    for name, info in described.items():
        if name not in known_columns or not isinstance(info, dict):
            continue
        result[name] = {
            "description": str(info.get("description") or ""),
            "semantic_type": info.get("semantic_type") or None,
            "synonyms": [str(s) for s in info.get("synonyms") or []],
            "delimiter": info.get("delimiter") or None,
        }
    return result


def build_starter_config(
    data_path: str, column_info: Mapping[str, dict[str, Any]] | None = None
) -> dict:
    """Build the config dict for ``data_path``, ready to serialize as YAML.

    ``column_info`` is an optional overlay (typically from
    ``describe_columns_with_llm``) keyed by column name. Columns missing from
    it fall back to a blank description and the name-based heuristic guess.
    """
    frame = load_dataframe(data_path)
    slug = dataset_slug(data_path)
    column_info = column_info or {}

    columns: dict[str, dict] = {}
    for column in frame.columns:
        name = str(column)
        info = column_info.get(name, {})
        semantic_type = info.get("semantic_type")
        used_heuristic_semantic_type = semantic_type is None
        if semantic_type is None:
            semantic_type = _infer_semantic_type(name, frame[column])
        delimiter = info.get("delimiter")
        if delimiter is None:
            delimiter = _infer_delimiter(frame[column])
            if delimiter and used_heuristic_semantic_type:
                semantic_type = "tag_list"
        columns[name] = {
            "description": info.get("description", ""),
            "semantic_type": semantic_type,
            "synonyms": info.get("synonyms", []),
            "delimiter": delimiter,
        }

    return {
        "dataset": {
            "id": slug,
            "path": data_path,
            "table_name": slug,
        },
        "limits": {
            "max_rows_returned": 100,
            "max_execution_ms": 3000,
            "max_cell_chars": 500,
            "max_preview_rows": 20,
        },
        "columns": columns,
    }


def write_starter_config(
    data_path: str,
    out_path: str,
    column_info: Mapping[str, dict[str, Any]] | None = None,
) -> Path:
    """Write a starter config for ``data_path`` to ``out_path`` and return it."""
    config = build_starter_config(data_path, column_info=column_info)
    out = Path(out_path)
    with out.open("w", encoding="utf-8") as handle:
        if column_info:
            handle.write(
                "# Generated by --init-config with LLM-drafted descriptions.\n"
                "# Review them for accuracy and adjust synonyms, then run:\n"
                f"#   uv run mcp-dataframe-chat --config {out.name}\n"
            )
        else:
            handle.write(
                "# Generated by --init-config. Descriptions and synonyms are\n"
                "# blank on purpose, fill them in for better answers, then run:\n"
                f"#   uv run mcp-dataframe-chat --config {out.name}\n"
            )
        yaml.safe_dump(
            config, handle, sort_keys=False, default_flow_style=False, allow_unicode=True
        )
    return out
