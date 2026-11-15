"""Initialize Git/DVC, fetch official months, version files and push to local cache.

Safe to rerun; existing files and configured DVC remotes are reused.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from download_data import download

ROOT = Path(__file__).resolve().parent.parent


def command(*args: str):
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main():
    if not (ROOT / ".git").exists():
        command("git", "init")
    if not (ROOT / ".dvc").exists():
        command("dvc", "init")
    command("dvc", "remote", "add", "-f", "-d", "localstore", ".dvcstore")
    for month in ("2025-01", "2025-02"):
        download(month, ROOT / "data/raw")
        command("dvc", "add", f"data/raw/yellow_tripdata_{month}.parquet")
    command("dvc", "push")
    print("DVC cache populated. Commit .dvc/config and data/raw/*.dvc, not raw parquet data.")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        sys.exit(f"Missing required command: {exc}. Install Git and taxiops[train].")
