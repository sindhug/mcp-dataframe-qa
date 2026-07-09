import re
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from mcp_dataframe_qa.config import ColumnConfig
from mcp_dataframe_qa.datasets import Dataset
from mcp_dataframe_qa.schemas import AnalysisPlan, FilterCondition, Metric, SortSpec


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("_", " ").lower()).strip()


def _parse_number(raw: str) -> float:
    value = raw.lower().replace("$", "").replace(",", "").strip()
    multiplier = 1.0
    if value.endswith("m"):
        multiplier = 1_000_000.0
        value = value[:-1]
    elif value.endswith("k"):
        multiplier = 1_000.0
        value = value[:-1]
    return float(value) * multiplier


def _comparison_op(word: str) -> str:
    if word in {"under", "below", "less than", "fewer than"}:
        return "<"
    if word in {"at most", "no more than", "maximum", "max"}:
        return "<="
    if word in {"at least", "minimum", "min"}:
        return ">="
    return ">"


class HeuristicPlanner:
    """Small deterministic planner for common dataframe questions.

    MCP hosts can also call execute_analysis_plan directly with a richer plan.
    This planner is intentionally conservative and local-only.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.column_terms = self._build_column_terms(dataset)

    def plan(self, question: str) -> AnalysisPlan:
        normalized = _normalize(question)
        filters = self._extract_filters(normalized)
        metrics = self._extract_metrics(normalized)
        group_by = self._extract_group_by(normalized)
        sort: List[SortSpec] = []
        limit: Optional[int] = None

        if "top" in normalized and group_by and metrics:
            sort = [SortSpec(column=metrics[0].output_name, direction="desc")]
            limit = self._extract_limit(normalized) or 10

        if not metrics:
            metrics = [Metric(fn="count", column="*", name="count")]

        return AnalysisPlan(filters=filters, group_by=group_by, metrics=metrics, sort=sort, limit=limit)

    def _build_column_terms(self, dataset: Dataset) -> Dict[str, List[str]]:
        terms: Dict[str, List[str]] = {}
        columns = dataset.columns or {}
        for column in dataset.frame.columns:
            configured = columns.get(column, ColumnConfig())
            candidates = {column, column.replace("_", " ")}
            candidates.update(configured.synonyms)
            if configured.description:
                candidates.add(configured.description)
            if configured.semantic_type:
                candidates.add(configured.semantic_type)
            terms[column] = sorted({_normalize(candidate) for candidate in candidates if candidate})
        return terms

    def _find_column(self, text: str, candidates: Optional[Iterable[str]] = None) -> Optional[str]:
        candidate_columns = list(candidates) if candidates else list(self.dataset.frame.columns)
        best: Optional[Tuple[int, str]] = None
        padded = " %s " % text
        for column in candidate_columns:
            for term in self.column_terms.get(column, []):
                if not term:
                    continue
                if " %s " % term in padded or term in text:
                    score = len(term)
                    if best is None or score > best[0]:
                        best = (score, column)
        return best[1] if best else None

    def _numeric_columns(self) -> List[str]:
        return [
            column
            for column in self.dataset.frame.columns
            if pd.api.types.is_numeric_dtype(self.dataset.frame[column]) and self._is_measure_column(column)
        ]

    def _is_measure_column(self, column: str) -> bool:
        configured = (self.dataset.columns or {}).get(column, ColumnConfig())
        semantic_type = (configured.semantic_type or "").lower()
        non_measure_types = {
            "date",
            "datetime",
            "dimension",
            "id",
            "identifier",
            "month",
            "postal_code",
            "rank",
            "year",
            "zipcode",
        }
        if semantic_type in non_measure_types:
            return False
        normalized = _normalize(column)
        if (
            normalized.endswith(" id")
            or normalized == "id"
            or normalized.endswith(" code")
            or normalized in {"year", "month"}
            or normalized.endswith(" rank")
        ):
            return False
        return True

    def _default_measure_column(self, text: str) -> str:
        metric_context = text
        if " by " in text:
            before_by, after_by = text.split(" by ", 1)
            if "top" in before_by and any(token in after_by for token in ["median", "average", "avg", "mean", "sum", "total"]):
                metric_context = after_by
            else:
                metric_context = before_by

        explicit = self._find_column(metric_context, self._numeric_columns())
        if explicit:
            return explicit
        for preferred in ["price", "revenue", "amount", "sales", "value"]:
            if preferred in self.dataset.frame.columns:
                return preferred
        numeric = self._numeric_columns()
        if numeric:
            return numeric[0]
        raise ValueError("No numeric column available for metric.")

    def _extract_metrics(self, text: str) -> List[Metric]:
        metric_fn: Optional[str] = None
        if any(token in text for token in ["average", "avg", "mean"]):
            metric_fn = "avg"
        elif "median" in text:
            metric_fn = "median"
        elif "sum" in text or "total" in text:
            metric_fn = "sum"
        elif "minimum" in text or "lowest" in text:
            metric_fn = "min"
        elif "maximum" in text or "highest" in text or "top" in text:
            metric_fn = "max"

        if metric_fn is None and (
            "how many" in text
            or "number of" in text
            or re.search(r"\bcount(?:\s+of)?\b", text) is not None
        ):
            return [Metric(fn="count", column="*", name="count")]

        if metric_fn is None:
            return []

        column = self._default_measure_column(text)
        if metric_fn == "max" and "top" in text and "median" in text:
            metric_fn = "median"
        output = column if column.startswith("%s_" % metric_fn) else "%s_%s" % (metric_fn, column)
        return [Metric(fn=metric_fn, column=column, name=output)]

    def _extract_group_by(self, text: str) -> List[str]:
        if " by " not in text and " each " not in text:
            return []

        if "top" in text and " by " in text:
            before_by = text.split(" by ", 1)[0]
            column = self._find_column(before_by)
            if column:
                return [column]

        after_by = text.split(" by ", 1)[1] if " by " in text else text.split(" each ", 1)[1]
        column = self._find_column(after_by)
        return [column] if column else []

    def _extract_filters(self, text: str) -> List[FilterCondition]:
        filters: List[FilterCondition] = []

        for column in self.dataset.frame.columns:
            if not pd.api.types.is_numeric_dtype(self.dataset.frame[column]):
                continue
            if not self._is_measure_column(column):
                continue

            terms = self.column_terms.get(column, [])
            term_pattern = "|".join(re.escape(term) for term in terms if term)
            if not term_pattern:
                continue

            plus_before = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*\+\s*(?:%s)" % term_pattern, text)
            at_least = re.search(
                r"(?:at least|minimum|min)\s+(\d[\d,]*(?:\.\d+)?)\s*(?:%s)" % term_pattern,
                text,
            )
            if plus_before or at_least:
                match = plus_before or at_least
                filters.append(
                    FilterCondition(column=column, op=">=", value=_parse_number(match.group(1)))
                )
                continue

            comparison_words = (
                "under|below|less than|fewer than|over|above|greater than|more than|"
                "at least|at most|no more than|minimum|maximum|min|max"
            )
            comparison = re.search(
                r"(?:%s).{0,18}?(%s)\s+\$?(\d[\d,]*(?:\.\d+)?[mk]?)"
                % (term_pattern, comparison_words),
                text,
            )
            reverse_comparison = re.search(
                r"(%s)\s+\$?(\d[\d,]*(?:\.\d+)?[mk]?)\s+(?:%s)"
                % (comparison_words, term_pattern),
                text,
            )
            if comparison or reverse_comparison:
                match = comparison or reverse_comparison
                word, raw = match.group(1), match.group(2)
                filters.append(
                    FilterCondition(column=column, op=_comparison_op(word), value=_parse_number(raw))
                )

        if not any(filter_.column == "price" for filter_ in filters) and "price" in self.dataset.frame.columns:
            money_match = re.search(
                r"(under|below|less than|fewer than|over|above|greater than|more than|at least|at most|no more than)\s+\$?(\d[\d,]*(?:\.\d+)?[mk]?)",
                text,
            )
            if money_match:
                word, raw = money_match.group(1), money_match.group(2)
                filters.append(FilterCondition(column="price", op=_comparison_op(word), value=_parse_number(raw)))

        for column in self.dataset.frame.columns:
            if pd.api.types.is_numeric_dtype(self.dataset.frame[column]):
                continue
            for term in self.column_terms.get(column, []):
                value_match = re.search(r"%s\s+(?:is|=|equals)\s+([a-z0-9 _-]+)" % re.escape(term), text)
                if value_match:
                    filters.append(
                        FilterCondition(column=column, op="contains", value=value_match.group(1).strip())
                    )

        return filters

    def _extract_limit(self, text: str) -> Optional[int]:
        match = re.search(r"top\s+(\d+)", text)
        if match:
            return int(match.group(1))
        return None
