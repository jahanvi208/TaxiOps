"""Download official (unchanged) NYC TLC yellow-taxi monthly Parquet files."""
from __future__ import annotations

import argparse
import re
import urllib.request
from pathlib import Path

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def download(month: str, outdir: Path) -> Path:
    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", month):
        raise ValueError(f"Invalid year-month: {month}")
    outdir.mkdir(parents=True, exist_ok=True)
    destination = outdir / f"yellow_tripdata_{month}.parquet"
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already present: {destination}")
        return destination
    url = f"{BASE_URL}/{destination.name}"
    partial = destination.with_suffix(".part")
    try:
        print(f"Downloading from official TLC mirror: {url}")
        urllib.request.urlretrieve(url, partial)
        if partial.stat().st_size == 0:
            raise OSError("Downloaded file is empty")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    print(f"Saved {destination} ({destination.stat().st_size:,} bytes)")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("months", nargs="+", help="YYYY-MM, e.g. 2025-01 2025-02")
    parser.add_argument("--outdir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    for requested_month in args.months:
        download(requested_month, args.outdir)
