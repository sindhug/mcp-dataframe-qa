from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

import pandas as pd

from mcp_dataframe_qa.config import ColumnConfig
from mcp_dataframe_qa.profiling import profile_dataframe


@dataclass
class Dataset:
    dataset_id: str
    table_name: str
    frame: pd.DataFrame
    path: Optional[Path] = None
    columns: Optional[Mapping[str, ColumnConfig]] = None

    def profile(self, max_examples: int = 5, max_cell_chars: int = 120) -> Dict:
        return profile_dataframe(
            self.frame,
            dataset_id=self.dataset_id,
            table_name=self.table_name,
            columns=self.columns or {},
            max_examples=max_examples,
            max_cell_chars=max_cell_chars,
        )


class DatasetRegistry:
    def __init__(self) -> None:
        self._datasets: Dict[str, Dataset] = {}

    def register(self, dataset: Dataset) -> None:
        self._datasets[dataset.dataset_id] = dataset

    def get(self, dataset_id: str = "default") -> Dataset:
        try:
            return self._datasets[dataset_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._datasets)) or "none"
            raise KeyError("Unknown dataset '%s'. Available datasets: %s" % (dataset_id, available)) from exc


def default_data_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "listings.csv"


def load_dataframe(path: str) -> pd.DataFrame:
    data_path = Path(path).expanduser()
    if not data_path.is_absolute():
        data_path = Path.cwd() / data_path
    if not data_path.exists():
        raise FileNotFoundError("Data file not found: %s" % data_path)

    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(data_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(data_path)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return pd.read_json(data_path, lines=suffix in {".jsonl", ".ndjson"})
    raise ValueError("Unsupported data file type '%s'. Use CSV, Parquet, or JSON." % suffix)


def dataset_from_path(
    path: Optional[str],
    dataset_id: str = "default",
    table_name: str = "dataframe",
    columns: Optional[Mapping[str, ColumnConfig]] = None,
) -> Dataset:
    resolved = Path(path).expanduser() if path else default_data_path()
    frame = load_dataframe(str(resolved))
    return Dataset(
        dataset_id=dataset_id,
        table_name=table_name,
        frame=frame,
        path=resolved,
        columns=columns or {},
    )
