from typing import Any, Dict, List, Literal, Optional

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
    name: Optional[str] = Field(default=None, alias="as")

    @property
    def output_name(self) -> str:
        if self.name:
            return self.name
        if self.fn == "count" and self.column == "*":
            return "count"
        return "%s_%s" % (self.fn, self.column)


class SortSpec(BaseModel):
    column: str
    direction: SortDirection = "desc"


class AnalysisPlan(BaseModel):
    filters: List[FilterCondition] = Field(default_factory=list)
    group_by: List[str] = Field(default_factory=list)
    metrics: List[Metric] = Field(default_factory=list)
    sort: List[SortSpec] = Field(default_factory=list)
    limit: Optional[int] = None


class TableResult(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]


class ChartSpec(BaseModel):
    kind: str
    x: str
    y: str


class StructuredResult(BaseModel):
    kind: ResultKind
    answer: str
    value: Optional[Any] = None
    table: Optional[TableResult] = None
    chart: Optional[ChartSpec] = None
    plan: Optional[AnalysisPlan] = None
    warnings: List[str] = Field(default_factory=list)
    audit_id: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return self.model_dump(by_alias=True)
