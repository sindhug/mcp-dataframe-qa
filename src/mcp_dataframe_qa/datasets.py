import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mcp_dataframe_qa.config import ColumnConfig
from mcp_dataframe_qa.profiling import profile_dataframe


@dataclass
class Dataset:
    dataset_id: str
    table_name: str
    frame: pd.DataFrame
    path: Path | None = None
    columns: Mapping[str, ColumnConfig] | None = None

    def profile(self, max_examples: int = 5, max_cell_chars: int = 120) -> dict:
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
        self._datasets: dict[str, Dataset] = {}

    def register(self, dataset: Dataset) -> None:
        self._datasets[dataset.dataset_id] = dataset

    def get(self, dataset_id: str = "default") -> Dataset:
        try:
            return self._datasets[dataset_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._datasets)) or "none"
            message = f"Unknown dataset '{dataset_id}'. Available datasets: {available}"
            raise KeyError(message) from exc


def default_data_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "listings.csv"


def load_dataframe(path: str) -> pd.DataFrame:
    data_path = Path(path).expanduser()
    if not data_path.is_absolute():
        data_path = Path.cwd() / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(data_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(data_path)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return pd.read_json(data_path, lines=suffix in {".jsonl", ".ndjson"})
    raise ValueError(f"Unsupported data file type '{suffix}'. Use CSV, Parquet, or JSON.")


def parse_date_columns(frame: pd.DataFrame, columns: Mapping[str, ColumnConfig]) -> pd.DataFrame:
    """Parse columns annotated semantic_type: date into real datetime64.

    Without this, a "date" column is just a string with only equality and
    lexicographic ordering to work with, so date arithmetic (days between two
    dates, which month something happened) has nothing to compute against.
    """
    date_columns = [
        name
        for name, config in columns.items()
        if config.semantic_type == "date" and name in frame.columns
    ]
    if not date_columns:
        return frame

    frame = frame.copy()
    for name in date_columns:
        series = frame[name]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        before_non_null = int(series.notna().sum())
        parsed = pd.to_datetime(series, errors="coerce")
        after_non_null = int(parsed.notna().sum())
        if before_non_null > 0 and after_non_null < before_non_null * 0.5:
            print(
                f"Warning: column '{name}' is configured as semantic_type: date, but only "
                f"{after_non_null} of {before_non_null} non-null values parsed as dates. "
                "Date arithmetic on this column may be unreliable.",
                file=sys.stderr,
            )
        frame[name] = parsed
    return frame


def warn_column_mismatches(frame: pd.DataFrame, columns: Mapping[str, ColumnConfig]) -> None:
    """Warn when a config's columns and the actual dataframe's columns disagree.

    Silently ignoring a mismatch is what let a config for one dataset keep
    describing a completely different dataset without anyone noticing. A
    one-line warning on load is cheap and catches exactly that class of
    mistake, whether the config is stale or the data file changed.
    """
    if not columns:
        return

    configured = set(columns.keys())
    actual = set(frame.columns)

    described_but_missing = sorted(configured - actual)
    if described_but_missing:
        print(
            "Warning: config describes columns not present in this dataset, "
            f"they will be ignored: {', '.join(described_but_missing)}",
            file=sys.stderr,
        )

    present_but_undescribed = sorted(actual - configured)
    if present_but_undescribed:
        print(
            f"Warning: {len(present_but_undescribed)} column(s) in this dataset have no "
            "description in the config, the planner will have less context for them: "
            f"{', '.join(present_but_undescribed)}",
            file=sys.stderr,
        )


def dataset_from_path(
    path: str | None,
    dataset_id: str = "default",
    table_name: str = "dataframe",
    columns: Mapping[str, ColumnConfig] | None = None,
) -> Dataset:
    resolved = Path(path).expanduser() if path else default_data_path()
    frame = load_dataframe(str(resolved))
    frame = parse_date_columns(frame, columns or {})
    warn_column_mismatches(frame, columns or {})
    return Dataset(
        dataset_id=dataset_id,
        table_name=table_name,
        frame=frame,
        path=resolved,
        columns=columns or {},
    )
