import time
from typing import Any

import pandas as pd

from mcp_dataframe_qa.config import LimitsConfig
from mcp_dataframe_qa.datasets import Dataset
from mcp_dataframe_qa.schemas import (
    AnalysisPlan,
    ChartSpec,
    Expression,
    Metric,
    StructuredResult,
    TableResult,
)
from mcp_dataframe_qa.validator import validate_plan


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _rows_safe(rows: list[dict[str, Any]], max_cell_chars: int) -> list[dict[str, Any]]:
    safe_rows: list[dict[str, Any]] = []
    for row in rows:
        safe_row: dict[str, Any] = {}
        for key, value in row.items():
            value = _json_safe(value)
            if isinstance(value, str) and len(value) > max_cell_chars:
                value = value[: max_cell_chars - 3] + "..."
            safe_row[key] = value
        safe_rows.append(safe_row)
    return safe_rows


def _safe_divide(left: Any, right: Any) -> Any:
    if isinstance(right, pd.Series):
        denominator = right.mask(right == 0)
        return left / denominator
    if right == 0:
        if isinstance(left, pd.Series):
            return pd.Series(pd.NA, index=left.index)
        return pd.NA
    return left / right


def _evaluate_expression(frame: pd.DataFrame, expr: Expression) -> Any:
    if expr.op == "column":
        return frame[expr.column]
    if expr.op == "literal":
        return expr.value
    if expr.op == "not":
        return ~_evaluate_expression(frame, expr.left)
    if expr.op == "year_of":
        return _evaluate_expression(frame, expr.left).dt.year
    if expr.op == "month_of":
        return _evaluate_expression(frame, expr.left).dt.month
    if expr.op == "day_of_week":
        return _evaluate_expression(frame, expr.left).dt.dayofweek

    left = _evaluate_expression(frame, expr.left)
    right = _evaluate_expression(frame, expr.right)
    if expr.op == "and":
        return left & right
    if expr.op == "or":
        return left | right
    if expr.op == "add":
        return left + right
    if expr.op == "subtract":
        return left - right
    if expr.op == "multiply":
        return left * right
    if expr.op in {"divide", "ratio"}:
        return _safe_divide(left, right)
    if expr.op == "==":
        return left == right
    if expr.op == "!=":
        return left != right
    if expr.op == "<":
        return left < right
    if expr.op == "<=":
        return left <= right
    if expr.op == ">":
        return left > right
    if expr.op == ">=":
        return left >= right
    if expr.op == "date_diff":
        return (left - right).dt.days
    raise ValueError(f"Unsupported expression op: {expr.op}")


def _apply_explode(frame: pd.DataFrame, plan: AnalysisPlan, dataset: Dataset) -> pd.DataFrame:
    """Turn a delimited tag-list column into one row per tag.

    A column like genres storing "Action|Adventure|Thriller" groups by the
    whole combination as a single category unless it's exploded first, so a
    "highest rated genre" question would silently rank exact genre
    combinations, most of them backed by a single movie, instead of genres.
    Only columns with a configured delimiter can be exploded, so a plan can't
    accidentally split an unrelated string column on a stray character.
    """
    if not plan.explode:
        return frame

    columns = dataset.columns or {}
    exploded = frame
    for column in plan.explode:
        delimiter = columns[column].delimiter
        split = exploded[column].astype(str).str.split(delimiter)
        exploded = exploded.assign(**{column: split}).explode(column, ignore_index=True)
        exploded[column] = exploded[column].str.strip()
    return exploded


def _apply_derived_columns(frame: pd.DataFrame, plan: AnalysisPlan) -> pd.DataFrame:
    if not plan.derive:
        return frame

    enriched = frame.copy()
    for derived in plan.derive:
        enriched[derived.name] = _evaluate_expression(enriched, derived.expr)
    return enriched


def _apply_filters(frame: pd.DataFrame, plan: AnalysisPlan) -> pd.DataFrame:
    filtered = frame
    for condition in plan.filters:
        series = filtered[condition.column]
        value = condition.value
        if condition.op == "==":
            mask = series == value
        elif condition.op == "!=":
            mask = series != value
        elif condition.op == "<":
            mask = series < value
        elif condition.op == "<=":
            mask = series <= value
        elif condition.op == ">":
            mask = series > value
        elif condition.op == ">=":
            mask = series >= value
        elif condition.op == "in":
            mask = series.isin(value)
        elif condition.op == "not_in":
            mask = ~series.isin(value)
        elif condition.op == "contains":
            mask = series.astype(str).str.contains(str(value), case=False, na=False)
        else:
            raise ValueError(f"Unsupported filter op: {condition.op}")
        filtered = filtered[mask]
    return filtered


def _compute_series(frame: pd.DataFrame, metric: Metric) -> Any:
    column = metric.column
    if metric.fn == "count":
        return len(frame) if column == "*" else frame[column].count()
    if metric.fn in {"avg", "mean"}:
        return frame[column].mean()
    if metric.fn == "median":
        return frame[column].median()
    if metric.fn == "sum":
        return frame[column].sum()
    if metric.fn == "min":
        return frame[column].min()
    if metric.fn == "max":
        return frame[column].max()
    if metric.fn == "nunique":
        return frame[column].nunique()
    if metric.fn == "corr":
        return frame[column].corr(frame[metric.column2])
    raise ValueError(f"Unsupported metric function: {metric.fn}")


def _compute_grouped(frame: pd.DataFrame, plan: AnalysisPlan) -> pd.DataFrame:
    grouped = frame.groupby(plan.group_by, dropna=False)
    group_sizes = grouped.size().reset_index(name="row_count")
    reserved_names = {metric.output_name for metric in plan.metrics}
    # How many rows back each group's aggregates matters: a tiny group can
    # otherwise dominate a ranking by average as if it were as reliable as a
    # group with thousands of rows. Surface it unless a metric already claims
    # that output name.
    result = group_sizes if "row_count" not in reserved_names else group_sizes[plan.group_by]

    for metric in plan.metrics:
        output = metric.output_name
        if metric.fn == "count" and metric.column == "*":
            values = grouped.size().reset_index(name=output)
        elif metric.fn in {"avg", "mean"}:
            values = grouped[metric.column].mean().reset_index(name=output)
        elif metric.fn == "median":
            values = grouped[metric.column].median().reset_index(name=output)
        elif metric.fn == "sum":
            values = grouped[metric.column].sum().reset_index(name=output)
        elif metric.fn == "min":
            values = grouped[metric.column].min().reset_index(name=output)
        elif metric.fn == "max":
            values = grouped[metric.column].max().reset_index(name=output)
        elif metric.fn == "nunique":
            values = grouped[metric.column].nunique().reset_index(name=output)
        elif metric.fn == "corr":
            values = grouped.apply(
                lambda g, c=metric.column, c2=metric.column2: g[c].corr(g[c2])
            ).reset_index(name=output)
        else:
            raise ValueError(f"Unsupported metric function: {metric.fn}")
        result = result.merge(values, on=plan.group_by, how="left")

    return result


def _format_answer(plan: AnalysisPlan, result: StructuredResult) -> str:
    if result.kind == "scalar":
        metric = plan.metrics[0]
        return "{} is {}.".format(metric.output_name.replace("_", " "), result.value)
    if result.table:
        return f"Returned {len(result.table.rows)} rows."
    return result.answer


def execute_plan(
    plan: AnalysisPlan,
    dataset: Dataset,
    limits: LimitsConfig,
    audit_id: str,
) -> StructuredResult:
    start = time.monotonic()
    plan = validate_plan(plan, dataset, limits)
    frame = _apply_explode(dataset.frame, plan, dataset)
    frame = _apply_derived_columns(frame, plan)
    filtered = _apply_filters(frame, plan)

    if plan.group_by:
        output = _compute_grouped(filtered, plan)
        for sort in plan.sort:
            output = output.sort_values(sort.column, ascending=sort.direction == "asc")
        output = output.head(plan.limit)
        rows = _rows_safe(output.to_dict(orient="records"), limits.max_cell_chars)
        table = TableResult(columns=list(output.columns), rows=rows)
        chart = None
        if plan.group_by and plan.metrics:
            chart = ChartSpec(kind="bar", x=plan.group_by[0], y=plan.metrics[0].output_name)
        result = StructuredResult(
            kind="table",
            answer=f"Returned {len(rows)} rows.",
            table=table,
            chart=chart,
            plan=plan,
            audit_id=audit_id,
        )
    else:
        values = [
            {metric.output_name: _json_safe(_compute_series(filtered, metric))}
            for metric in plan.metrics
        ]
        if len(values) == 1:
            value = next(iter(values[0].values()))
            result = StructuredResult(
                kind="scalar",
                answer="",
                value=value,
                plan=plan,
                audit_id=audit_id,
            )
        else:
            row: dict[str, Any] = {}
            for value in values:
                row.update(value)
            table = TableResult(
                columns=list(row.keys()),
                rows=_rows_safe([row], limits.max_cell_chars),
            )
            result = StructuredResult(
                kind="table",
                answer="Returned 1 row.",
                table=table,
                plan=plan,
                audit_id=audit_id,
            )

    elapsed_ms = int((time.monotonic() - start) * 1000)
    if elapsed_ms > limits.max_execution_ms:
        result.warnings.append(
            f"Execution completed in {elapsed_ms}ms, "
            f"which exceeds configured max_execution_ms={limits.max_execution_ms}."
        )
    result.answer = _format_answer(plan, result)
    return result
