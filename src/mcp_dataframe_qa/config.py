from pathlib import Path

from pydantic import BaseModel, Field


class ColumnConfig(BaseModel):
    description: str = ""
    semantic_type: str | None = None
    synonyms: list[str] = Field(default_factory=list)


class DatasetConfig(BaseModel):
    id: str = "default"
    path: str = "data/listings.csv"
    table_name: str = "dataframe"


class LimitsConfig(BaseModel):
    max_rows_returned: int = 100
    max_execution_ms: int = 3000
    max_cell_chars: int = 500
    max_preview_rows: int = 20


class AppConfig(BaseModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    columns: dict[str, ColumnConfig] = Field(default_factory=dict)
    audit_log_path: str | None = None


def load_config(path: str | None = None) -> AppConfig:
    if path is None:
        default_path = Path("dataframe_qa.yaml")
        if not default_path.exists():
            return AppConfig()
        path = str(default_path)

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read dataframe_qa.yaml") from exc

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return AppConfig.model_validate(raw)
