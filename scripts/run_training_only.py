"""Run the actual real-data tuner after Docker Compose starts the MLflow server.

This script is deliberately Docker-free: the containers run on a GitHub-hosted runner.
"""
from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    for _ in range(90):
        try:
            with urllib.request.urlopen('http://127.0.0.1:5000/', timeout=3) as response:
                if response.status == 200:
                    break
                time.sleep(2)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(2)
    else:
        sys.exit('MLflow server did not become reachable; inspect compose logs')
    args = [sys.executable, '-m', 'taxiops.train', *sys.argv[1:]]
    print('Running MLflow training:', ' '.join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
