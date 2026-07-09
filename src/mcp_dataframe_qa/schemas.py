from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

FilterOp = Literal["==", "!=", "<", "<=", ">", ">=", "in", "not_in", "contains"]
MetricFn = Literal["count", "sum", "avg", "mean", "median", "min", "max", "nunique"]
SortDirection = Literal["asc", "desc"]
ResultKind = Literal["scalar", "table", "error"]


class FilterCondition(BaseModel):
    column: str
    op: FilterOp
    value: Any


class Metric(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    fn: MetricFn
    column: str = "*"
    name: str | None = Field(default=None, alias="as")

    @property
    def output_name(self) -> str:
        if self.name:
            return self.name
        if self.fn == "count" and self.column == "*":
            return "count"
        return f"{self.fn}_{self.column}"


class SortSpec(BaseModel):
    column: str
    direction: SortDirection = "desc"


class AnalysisPlan(BaseModel):
    filters: list[FilterCondition] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    metrics: list[Metric] = Field(default_factory=list)
    sort: list[SortSpec] = Field(default_factory=list)
    limit: int | None = None


class TableResult(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]


class ChartSpec(BaseModel):
    kind: str
    x: str
    y: str


class StructuredResult(BaseModel):
    kind: ResultKind
    answer: str
    value: Any | None = None
    table: TableResult | None = None
    chart: ChartSpec | None = None
    plan: AnalysisPlan | None = None
    warnings: list[str] = Field(default_factory=list)
    audit_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True)
