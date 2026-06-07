from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_VENDOR_DIR = Path(".wtpy_trial_vendor")
TSINGHUA_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"

PACKAGES = [
    "wtpy==0.9.9.3",
    "chardet",
    "pyyaml",
    "tqdm",
    "pyquery",
    "py_mini_racer",
    "requests",
    "pytz",
    "psutil",
    "fastapi",
    "uvicorn",
    "starlette",
    "flask",
    "flask-cors",
    "simple-websocket",
    "xlsxwriter",
    "openpyxl",
    "deap",
    "scikit-learn",
]


def install_wtpy_vendor(vendor_dir: Path = DEFAULT_VENDOR_DIR, index_url: str = TSINGHUA_INDEX) -> Path:
    vendor_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--target",
        str(vendor_dir),
        "--upgrade",
        "--no-deps",
        "wtpy==0.9.9.3",
        "-i",
        index_url,
    ]
    subprocess.run(cmd, check=True)

    deps = [pkg for pkg in PACKAGES if not pkg.startswith("wtpy")]
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--target",
        str(vendor_dir),
        "--upgrade",
        *deps,
        "-i",
        index_url,
    ]
    subprocess.run(cmd, check=True)
    return vendor_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Install wtpy into an isolated vendor folder.")
    parser.add_argument("--vendor-dir", default=str(DEFAULT_VENDOR_DIR))
    parser.add_argument("--index-url", default=TSINGHUA_INDEX)
    args = parser.parse_args(argv)
    vendor_dir = install_wtpy_vendor(Path(args.vendor_dir), args.index_url)
    print(vendor_dir.resolve())


if __name__ == "__main__":
    main()
