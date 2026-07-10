import re
from collections.abc import Iterable

import pandas as pd

from mcp_dataframe_qa.config import LimitsConfig
from mcp_dataframe_qa.datasets import Dataset
from mcp_dataframe_qa.schemas import AnalysisPlan, Expression, Metric

MAX_DERIVED_COLUMNS = 20
MAX_EXPRESSION_DEPTH = 8
DERIVED_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
BINARY_NUMERIC_OPS = {"add", "subtract", "multiply", "divide", "ratio"}
COMPARISON_OPS = {"==", "!=", "<", "<=", ">", ">="}
LOGICAL_BINARY_OPS = {"and", "or"}
DATE_PART_OPS = {"year_of", "month_of", "day_of_week"}
NUMERIC_METRIC_FNS = {"avg", "mean", "median", "sum"}


class PlanValidationError(ValueError):
    """Raised when an analysis plan violates schema or policy constraints."""


def _known_result_columns(plan: AnalysisPlan) -> set[str]:
    names = set(plan.group_by)
    names.update(metric.output_name for metric in plan.metrics)
    return names


def _source_columns(dataset: Dataset) -> set[str]:
    return {str(column) for column in dataset.frame.columns}


def _numeric_source_columns(dataset: Dataset) -> set[str]:
    return {
        str(column)
        for column in dataset.frame.columns
        if pd.api.types.is_numeric_dtype(dataset.frame[column])
    }


def _date_source_columns(dataset: Dataset) -> set[str]:
    return {
        str(column)
        for column in dataset.frame.columns
        if pd.api.types.is_datetime64_any_dtype(dataset.frame[column])
    }


def _validate_columns(columns: Iterable[str], known: set[str], context: str) -> None:
    for column in columns:
        if column not in known:
            raise PlanValidationError(
                "Unknown {} column '{}'. Available columns: {}".format(
                    context,
                    column,
                    ", ".join(sorted(known)),
                )
            )


def _require_no_fields(expr: Expression, fields: dict[str, object | None]) -> None:
    unexpected = [name for name, value in fields.items() if value is not None]
    if unexpected:
        raise PlanValidationError(
            "Expression op '{}' does not allow field(s): {}.".format(
                expr.op,
                ", ".join(sorted(unexpected)),
            )
        )


def _require_numeric(kind: str, expr: Expression) -> None:
    if kind != "numeric":
        raise PlanValidationError(f"Expression op '{expr.op}' requires numeric operands.")


def _require_boolean(kind: str, expr: Expression) -> None:
    if kind != "boolean":
        raise PlanValidationError(
            f"Expression op '{expr.op}' requires boolean operands, such as the "
            "result of a comparison or another and/or/not expression."
        )


def _require_date(kind: str, expr: Expression) -> None:
    if kind != "date":
        raise PlanValidationError(
            f"Expression op '{expr.op}' requires a date operand, a column with semantic_type: date."
        )


def _validate_expression(
    expr: Expression,
    known_columns: set[str],
    numeric_columns: set[str],
    date_columns: set[str],
    depth: int = 0,
) -> str:
    if depth > MAX_EXPRESSION_DEPTH:
        raise PlanValidationError(
            f"Derived expression depth exceeds limit of {MAX_EXPRESSION_DEPTH}."
        )

    if expr.op == "column":
        if not expr.column:
            raise PlanValidationError("Expression op 'column' requires a column.")
        _require_no_fields(expr, {"value": expr.value, "left": expr.left, "right": expr.right})
        _validate_columns([expr.column], known_columns, "expression")
        if expr.column in numeric_columns:
            return "numeric"
        if expr.column in date_columns:
            return "date"
        return "other"

    if expr.op == "literal":
        if expr.value is None or not isinstance(expr.value, str | int | float | bool):
            raise PlanValidationError(
                "Expression op 'literal' requires a string, number, or boolean value."
            )
        _require_no_fields(expr, {"column": expr.column, "left": expr.left, "right": expr.right})
        return "other" if isinstance(expr.value, str | bool) else "numeric"

    if expr.op in BINARY_NUMERIC_OPS:
        if expr.left is None or expr.right is None:
            raise PlanValidationError(f"Expression op '{expr.op}' requires left and right.")
        _require_no_fields(expr, {"column": expr.column, "value": expr.value})
        left_kind = _validate_expression(
            expr.left, known_columns, numeric_columns, date_columns, depth + 1
        )
        right_kind = _validate_expression(
            expr.right, known_columns, numeric_columns, date_columns, depth + 1
        )
        _require_numeric(left_kind, expr)
        _require_numeric(right_kind, expr)
        return "numeric"

    if expr.op in COMPARISON_OPS:
        if expr.left is None or expr.right is None:
            raise PlanValidationError(f"Expression op '{expr.op}' requires left and right.")
        _require_no_fields(expr, {"column": expr.column, "value": expr.value})
        _validate_expression(expr.left, known_columns, numeric_columns, date_columns, depth + 1)
        _validate_expression(expr.right, known_columns, numeric_columns, date_columns, depth + 1)
        return "boolean"

    if expr.op in LOGICAL_BINARY_OPS:
        if expr.left is None or expr.right is None:
            raise PlanValidationError(f"Expression op '{expr.op}' requires left and right.")
        _require_no_fields(expr, {"column": expr.column, "value": expr.value})
        left_kind = _validate_expression(
            expr.left, known_columns, numeric_columns, date_columns, depth + 1
        )
        right_kind = _validate_expression(
            expr.right, known_columns, numeric_columns, date_columns, depth + 1
        )
        _require_boolean(left_kind, expr)
        _require_boolean(right_kind, expr)
        return "boolean"

    if expr.op == "not":
        if expr.left is None:
            raise PlanValidationError("Expression op 'not' requires left.")
        _require_no_fields(expr, {"column": expr.column, "value": expr.value, "right": expr.right})
        operand_kind = _validate_expression(
            expr.left, known_columns, numeric_columns, date_columns, depth + 1
        )
        _require_boolean(operand_kind, expr)
        return "boolean"

    if expr.op in DATE_PART_OPS:
        if expr.left is None:
            raise PlanValidationError(f"Expression op '{expr.op}' requires left.")
        _require_no_fields(expr, {"column": expr.column, "value": expr.value, "right": expr.right})
        operand_kind = _validate_expression(
            expr.left, known_columns, numeric_columns, date_columns, depth + 1
        )
        _require_date(operand_kind, expr)
        return "numeric"

    if expr.op == "date_diff":
        if expr.left is None or expr.right is None:
            raise PlanValidationError(f"Expression op '{expr.op}' requires left and right.")
        _require_no_fields(expr, {"column": expr.column, "value": expr.value})
        left_kind = _validate_expression(
            expr.left, known_columns, numeric_columns, date_columns, depth + 1
        )
        right_kind = _validate_expression(
            expr.right, known_columns, numeric_columns, date_columns, depth + 1
        )
        _require_date(left_kind, expr)
        _require_date(right_kind, expr)
        return "numeric"

    raise PlanValidationError(f"Unsupported expression op: {expr.op}")


def _validate_derived_columns(plan: AnalysisPlan, dataset: Dataset) -> tuple[set[str], set[str]]:
    known_columns = _source_columns(dataset)
    numeric_columns = _numeric_source_columns(dataset)
    date_columns = _date_source_columns(dataset)
    derived_names: set[str] = set()

    if len(plan.derive) > MAX_DERIVED_COLUMNS:
        raise PlanValidationError(
            f"Analysis plan may define at most {MAX_DERIVED_COLUMNS} derived columns."
        )

    for derived in plan.derive:
        if not DERIVED_NAME_PATTERN.match(derived.name):
            raise PlanValidationError(
                f"Derived column name '{derived.name}' must be a simple identifier."
            )
        if derived.name in known_columns or derived.name in derived_names:
            raise PlanValidationError(
                f"Derived column '{derived.name}' conflicts with an existing or derived column."
            )
        kind = _validate_expression(derived.expr, known_columns, numeric_columns, date_columns)
        known_columns.add(derived.name)
        derived_names.add(derived.name)
        if kind in {"numeric", "boolean"}:
            numeric_columns.add(derived.name)

    return known_columns, numeric_columns


def _validate_explode(plan: AnalysisPlan, dataset: Dataset) -> None:
    _validate_columns(plan.explode, _source_columns(dataset), "explode")
    configured_columns = dataset.columns or {}
    for column in plan.explode:
        config = configured_columns.get(column)
        if not config or not config.delimiter:
            raise PlanValidationError(
                f"Column '{column}' cannot be exploded: no delimiter is configured for it "
                "in the dataset config."
            )


def _validate_metrics(
    metrics: list[Metric], known_columns: set[str], numeric_columns: set[str], context: str
) -> None:
    for metric in metrics:
        if metric.fn == "count" and metric.column == "*":
            continue
        if metric.column == "*":
            raise PlanValidationError("Only count may use column='*'.")
        _validate_columns([metric.column], known_columns, context)
        if metric.fn == "corr":
            if not metric.column2:
                raise PlanValidationError("Metric 'corr' requires column2.")
            _validate_columns([metric.column2], known_columns, context)
            if metric.column not in numeric_columns or metric.column2 not in numeric_columns:
                raise PlanValidationError(
                    "Metric 'corr' requires numeric columns, got "
                    f"'{metric.column}' and '{metric.column2}'."
                )
            continue
        if metric.fn in NUMERIC_METRIC_FNS and metric.column not in numeric_columns:
            raise PlanValidationError(
                f"Metric '{metric.fn}' requires numeric column '{metric.column}'."
            )


def _intermediate_columns(
    plan: AnalysisPlan, source_numeric_columns: set[str]
) -> tuple[set[str], set[str]]:
    """Columns (and which of them are numeric) available on the grouped result.

    count/sum/avg/mean/median/nunique/corr always produce numeric output.
    min/max only do when applied to an already-numeric source column.
    """
    known = set(plan.group_by)
    numeric: set[str] = set()
    reserved = {metric.output_name for metric in plan.metrics}
    if "row_count" not in reserved:
        known.add("row_count")
        numeric.add("row_count")
    for metric in plan.metrics:
        known.add(metric.output_name)
        if metric.fn in {"count", "sum", "avg", "mean", "median", "nunique", "corr"}:
            numeric.add(metric.output_name)
        elif metric.fn in {"min", "max"} and metric.column in source_numeric_columns:
            numeric.add(metric.output_name)
    return known, numeric


def _validate_regroup(plan: AnalysisPlan, numeric_columns: set[str]) -> None:
    regroup = plan.regroup
    if regroup is None:
        return
    if not plan.group_by:
        raise PlanValidationError("regroup requires the plan to have a group_by.")
    if regroup.group_by and not regroup.metrics:
        raise PlanValidationError("regroup.metrics is required when regroup.group_by is set.")
    if regroup.metrics and not regroup.group_by:
        raise PlanValidationError(
            "regroup.metrics requires regroup.group_by; omit both to just derive, sort, "
            "and limit the grouped result."
        )

    known, numeric = _intermediate_columns(plan, numeric_columns)
    date_columns: set[str] = set()

    for derived in regroup.derive:
        if not DERIVED_NAME_PATTERN.match(derived.name):
            raise PlanValidationError(
                f"Derived column name '{derived.name}' must be a simple identifier."
            )
        if derived.name in known:
            raise PlanValidationError(
                f"regroup derived column '{derived.name}' conflicts with an existing column."
            )
        kind = _validate_expression(derived.expr, known, numeric, date_columns)
        known.add(derived.name)
        if kind in {"numeric", "boolean"}:
            numeric.add(derived.name)

    _validate_columns(regroup.group_by, known, "regroup group_by")
    _validate_metrics(regroup.metrics, known, numeric, "regroup metric")

    if regroup.group_by:
        result_columns = set(regroup.group_by) | {m.output_name for m in regroup.metrics}
    else:
        result_columns = known
    for sort in regroup.sort:
        if sort.column not in result_columns:
            raise PlanValidationError(
                "regroup sort column '{}' is not produced by regroup. Result columns: {}".format(
                    sort.column, ", ".join(sorted(result_columns))
                )
            )

    if regroup.limit is not None and regroup.limit < 1:
        raise PlanValidationError("regroup limit must be at least 1.")


def validate_plan(plan: AnalysisPlan, dataset: Dataset, limits: LimitsConfig) -> AnalysisPlan:
    if not plan.metrics:
        raise PlanValidationError("Analysis plan must include at least one metric.")

    _validate_explode(plan, dataset)

    known_columns, numeric_columns = _validate_derived_columns(plan, dataset)

    _validate_columns([condition.column for condition in plan.filters], known_columns, "filter")
    _validate_columns(plan.group_by, known_columns, "group_by")

    _validate_metrics(plan.metrics, known_columns, numeric_columns, "metric")
    _validate_regroup(plan, numeric_columns)

    result_columns = _known_result_columns(plan)
    for sort in plan.sort:
        if sort.column not in result_columns:
            raise PlanValidationError(
                "Sort column '{}' is not produced by the plan. Result columns: {}".format(
                    sort.column,
                    ", ".join(sorted(result_columns)),
                )
            )

    if plan.limit is None:
        plan.limit = limits.max_rows_returned
    elif plan.limit < 1:
        raise PlanValidationError("Plan limit must be at least 1.")
    else:
        plan.limit = min(plan.limit, limits.max_rows_returned)

    if plan.regroup is not None and plan.regroup.limit is not None:
        plan.regroup.limit = min(plan.regroup.limit, limits.max_rows_returned)

    return plan
