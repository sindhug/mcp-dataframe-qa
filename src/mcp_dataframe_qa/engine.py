import time
from typing import Any

import pandas as pd

from mcp_dataframe_qa.audit import new_audit_id, write_audit_record
from mcp_dataframe_qa.config import AppConfig, ColumnConfig, LimitsConfig
from mcp_dataframe_qa.datasets import Dataset, DatasetRegistry, dataset_from_path
from mcp_dataframe_qa.executor import execute_plan
from mcp_dataframe_qa.planner import HeuristicPlanner
from mcp_dataframe_qa.schemas import AnalysisPlan, StructuredResult, TableResult
from mcp_dataframe_qa.validator import PlanValidationError


class DataFrameQA:
    def __init__(self, registry: DatasetRegistry, config: AppConfig) -> None:
        self.registry = registry
        self.config = config

    @classmethod
    def from_config(cls, config: AppConfig, data_path: str | None = None) -> "DataFrameQA":
        registry = DatasetRegistry()
        dataset = dataset_from_path(
            path=data_path or config.dataset.path,
            dataset_id=config.dataset.id,
            table_name=config.dataset.table_name,
            columns=config.columns,
        )
        registry.register(dataset)
        return cls(registry=registry, config=config)

    @classmethod
    def from_dataframe(
        cls,
        frame: pd.DataFrame,
        dataset_id: str = "default",
        table_name: str = "dataframe",
        columns: dict[str, ColumnConfig] | None = None,
        limits: LimitsConfig | None = None,
    ) -> "DataFrameQA":
        registry = DatasetRegistry()
        dataset = Dataset(
            dataset_id=dataset_id,
            table_name=table_name,
            frame=frame,
            columns=columns or {},
        )
        registry.register(dataset)
        config = AppConfig(
            dataset={"id": dataset_id, "path": "", "table_name": table_name},
            columns=columns or {},
            limits=limits or LimitsConfig(),
        )
        return cls(registry=registry, config=config)

    def profile(self, dataset_id: str = "default") -> dict[str, Any]:
        dataset = self._dataset(dataset_id)
        return dataset.profile(max_cell_chars=self.config.limits.max_cell_chars)

    def preview(self, dataset_id: str = "default", limit: int | None = None) -> StructuredResult:
        dataset = self._dataset(dataset_id)
        row_limit = min(
            limit or self.config.limits.max_preview_rows,
            self.config.limits.max_preview_rows,
        )
        rows = dataset.frame.head(row_limit).to_dict(orient="records")
        table = TableResult(columns=list(dataset.frame.columns), rows=rows)
        return StructuredResult(
            kind="table",
            answer=f"Returned {len(rows)} preview rows.",
            table=table,
            audit_id=new_audit_id("preview"),
        )

    def query(self, question: str, dataset_id: str = "default") -> StructuredResult:
        dataset = self._dataset(dataset_id)
        audit_id = new_audit_id()
        started = time.time()
        try:
            plan = HeuristicPlanner(dataset).plan(question)
            result = self.execute_plan(plan, dataset_id=dataset_id, audit_id=audit_id)
            result.answer = f"{result.answer} Query: {question}"
            return result
        except Exception as exc:
            return StructuredResult(
                kind="error",
                answer=str(exc),
                warnings=["The built-in planner is intentionally conservative."],
                audit_id=audit_id,
            )
        finally:
            write_audit_record(
                self.config.audit_log_path,
                {
                    "audit_id": audit_id,
                    "dataset_id": dataset_id,
                    "question": question,
                    "started_at": started,
                },
            )

    def execute_plan(
        self,
        plan: AnalysisPlan,
        dataset_id: str = "default",
        audit_id: str | None = None,
    ) -> StructuredResult:
        dataset = self._dataset(dataset_id)
        audit_id = audit_id or new_audit_id()
        try:
            result = execute_plan(
                plan,
                dataset=dataset,
                limits=self.config.limits,
                audit_id=audit_id,
            )
        except PlanValidationError as exc:
            result = StructuredResult(kind="error", answer=str(exc), plan=plan, audit_id=audit_id)
        write_audit_record(
            self.config.audit_log_path,
            {
                "audit_id": audit_id,
                "dataset_id": dataset_id,
                "plan": plan.model_dump(by_alias=True),
                "kind": result.kind,
            },
        )
        return result

    def _dataset(self, dataset_id: str) -> Dataset:
        try:
            return self.registry.get(dataset_id)
        except KeyError:
            if dataset_id == "default":
                return self.registry.get(self.config.dataset.id)
            raise
