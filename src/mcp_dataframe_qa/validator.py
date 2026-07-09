from collections.abc import Iterable

from mcp_dataframe_qa.config import LimitsConfig
from mcp_dataframe_qa.datasets import Dataset
from mcp_dataframe_qa.schemas import AnalysisPlan


class PlanValidationError(ValueError):
    """Raised when an analysis plan violates schema or policy constraints."""


def _known_result_columns(plan: AnalysisPlan) -> set[str]:
    names = set(plan.group_by)
    names.update(metric.output_name for metric in plan.metrics)
    return names


def _validate_columns(columns: Iterable[str], dataset: Dataset, context: str) -> None:
    known = set(str(column) for column in dataset.frame.columns)
    for column in columns:
        if column not in known:
            raise PlanValidationError(
                "Unknown {} column '{}'. Available columns: {}".format(
                    context,
                    column,
                    ", ".join(sorted(known)),
                )
            )


def validate_plan(plan: AnalysisPlan, dataset: Dataset, limits: LimitsConfig) -> AnalysisPlan:
    if not plan.metrics:
        raise PlanValidationError("Analysis plan must include at least one metric.")

    _validate_columns([condition.column for condition in plan.filters], dataset, "filter")
    _validate_columns(plan.group_by, dataset, "group_by")

    for metric in plan.metrics:
        if metric.fn == "count" and metric.column == "*":
            continue
        if metric.column == "*":
            raise PlanValidationError("Only count may use column='*'.")
        _validate_columns([metric.column], dataset, "metric")

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

    return plan
