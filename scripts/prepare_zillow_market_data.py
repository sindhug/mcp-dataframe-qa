#!/usr/bin/env python3
"""Prepare the bundled Zillow Research market dataset.

The source CSVs are published by Zillow Research in a wide time-series format.
This script downloads the public files when needed, melts them to one
region-month row, and writes a compact dataframe-friendly CSV.
"""

from __future__ import annotations

import argparse
import re
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


RAW_SOURCES: Dict[str, str] = {
    "for_sale_inventory": (
        "https://files.zillowstatic.com/research/public_csvs/invt_fs/"
        "Metro_invt_fs_uc_sfrcondo_sm_month.csv"
    ),
    "new_listings": (
        "https://files.zillowstatic.com/research/public_csvs/new_listings/"
        "Metro_new_listings_uc_sfrcondo_sm_month.csv"
    ),
    "median_list_price": (
        "https://files.zillowstatic.com/research/public_csvs/mlp/"
        "Metro_mlp_uc_sfrcondo_sm_month.csv"
    ),
}

ID_COLUMNS = ["RegionID", "SizeRank", "RegionName", "RegionType", "StateName"]
RENAMED_ID_COLUMNS = {
    "RegionID": "region_id",
    "SizeRank": "size_rank",
    "RegionName": "region_name",
    "RegionType": "region_type",
    "StateName": "state_name",
}
MERGE_COLUMNS = ["region_id", "size_rank", "region_name", "region_type", "state_name", "period"]
DATE_COLUMN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/zillow", help="Directory for source CSVs.")
    parser.add_argument(
        "--output",
        default="data/zillow_metro_market.csv",
        help="Path for the prepared tidy CSV.",
    )
    parser.add_argument(
        "--recent-months",
        type=int,
        default=0,
        help="Keep only the most recent N months. The default keeps the full source history.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download source CSVs even when matching files already exist.",
    )
    return parser.parse_args()


def source_path(raw_dir: Path, url: str) -> Path:
    return raw_dir / url.rsplit("/", 1)[-1]


def download_sources(raw_dir: Path, force: bool = False) -> Dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for metric, url in RAW_SOURCES.items():
        destination = source_path(raw_dir, url)
        paths[metric] = destination
        if destination.exists() and not force:
            continue

        request = urllib.request.Request(url, headers={"User-Agent": "mcp-dataframe-qa/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            destination.write_bytes(response.read())
    return paths


def date_columns(columns: Iterable[str]) -> List[str]:
    return [column for column in columns if DATE_COLUMN_RE.match(column)]


def melt_metric(path: Path, metric_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    dates = date_columns(frame.columns)
    missing_id_columns = [column for column in ID_COLUMNS if column not in frame.columns]
    if missing_id_columns:
        raise ValueError("%s is missing required columns: %s" % (path, missing_id_columns))
    if not dates:
        raise ValueError("%s does not contain date columns." % path)

    long = frame[ID_COLUMNS + dates].melt(
        id_vars=ID_COLUMNS,
        value_vars=dates,
        var_name="period",
        value_name=metric_name,
    )
    long = long.rename(columns=RENAMED_ID_COLUMNS)
    long["period"] = pd.to_datetime(long["period"], errors="raise").dt.strftime("%Y-%m-%d")
    long[metric_name] = pd.to_numeric(long[metric_name], errors="coerce")
    return long


def prepare_dataset(paths: Dict[str, Path], recent_months: Optional[int] = None) -> pd.DataFrame:
    prepared: Optional[pd.DataFrame] = None
    for metric, path in paths.items():
        metric_frame = melt_metric(path, metric)
        if prepared is None:
            prepared = metric_frame
        else:
            prepared = prepared.merge(metric_frame, on=MERGE_COLUMNS, how="outer")

    if prepared is None:
        raise ValueError("No source files were provided.")

    prepared["state_name"] = prepared["state_name"].fillna("US")
    period = pd.to_datetime(prepared["period"], errors="raise")
    prepared["year"] = period.dt.year
    prepared["month"] = period.dt.month

    if recent_months and recent_months > 0:
        keep_periods = sorted(prepared["period"].unique())[-recent_months:]
        prepared = prepared[prepared["period"].isin(keep_periods)]

    ordered_columns = [
        "region_id",
        "size_rank",
        "region_name",
        "region_type",
        "state_name",
        "period",
        "year",
        "month",
        "for_sale_inventory",
        "new_listings",
        "median_list_price",
    ]
    prepared = prepared[ordered_columns]
    prepared = prepared.sort_values(["period", "size_rank"], ascending=[False, True])
    return prepared.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    paths = download_sources(Path(args.raw_dir), force=args.force_download)
    prepared = prepare_dataset(paths, recent_months=args.recent_months)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output, index=False)

    print("Wrote %s rows to %s" % (len(prepared), output))
    print("Date range: %s to %s" % (prepared["period"].min(), prepared["period"].max()))


if __name__ == "__main__":
    main()
