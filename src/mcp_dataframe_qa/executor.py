import time
from typing import Any

import pandas as pd

from mcp_dataframe_qa.config import LimitsConfig
from mcp_dataframe_qa.datasets import Dataset
from mcp_dataframe_qa.schemas import (
    AnalysisPlan,
    ChartSpec,
    Expression,
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

    left = _evaluate_expression(frame, expr.left)
    right = _evaluate_expression(frame, expr.right)
    if expr.op == "add":
        return left + right
    if expr.op == "subtract":
        return left - right
    if expr.op == "multiply":
        return left * right
    if expr.op in {"divide", "ratio"}:
        return _safe_divide(left, right)
    raise ValueError(f"Unsupported expression op: {expr.op}")


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


def _compute_series(frame: pd.DataFrame, metric_fn: str, column: str) -> Any:
    if metric_fn == "count":
        return len(frame) if column == "*" else frame[column].count()
    if metric_fn in {"avg", "mean"}:
        return frame[column].mean()
    if metric_fn == "median":
        return frame[column].median()
    if metric_fn == "sum":
        return frame[column].sum()
    if metric_fn == "min":
        return frame[column].min()
    if metric_fn == "max":
        return frame[column].max()
    if metric_fn == "nunique":
        return frame[column].nunique()
    raise ValueError(f"Unsupported metric function: {metric_fn}")


def _compute_grouped(frame: pd.DataFrame, plan: AnalysisPlan) -> pd.DataFrame:
    grouped = frame.groupby(plan.group_by, dropna=False)
    result = grouped.size().reset_index(name="__rows__")[plan.group_by]

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
    frame = _apply_derived_columns(dataset.frame, plan)
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
            {metric.output_name: _json_safe(_compute_series(filtered, metric.fn, metric.column))}
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
