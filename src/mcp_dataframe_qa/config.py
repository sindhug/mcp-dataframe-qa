from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ColumnConfig(BaseModel):
    description: str = ""
    semantic_type: Optional[str] = None
    synonyms: List[str] = Field(default_factory=list)


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
    columns: Dict[str, ColumnConfig] = Field(default_factory=dict)
    audit_log_path: Optional[str] = None


def load_config(path: Optional[str] = None) -> AppConfig:
    if path is None:
        default_path = Path("dataframe_qa.yaml")
        if not default_path.exists():
            return AppConfig()
        path = str(default_path)

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError("Config file not found: %s" % config_path)

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read dataframe_qa.yaml") from exc

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return AppConfig.model_validate(raw)
