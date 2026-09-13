#!/usr/bin/env python3
"""
CI Notebook Verification Runner.

Headlessly executes all Jupyter notebooks across tutorials, masterclass,
examples, and competition directories to guarantee zero runtime regressions.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT_DIR = Path(__file__).resolve().parent.parent

SEARCH_DIRS = [
    ROOT_DIR / "notebooks",
    ROOT_DIR / "docs" / "tutorials",
    ROOT_DIR / "docs" / "examples",
    ROOT_DIR / "examples",
    ROOT_DIR / "competitions",
]


def discover_notebooks() -> list[Path]:
    """Find all valid .ipynb files excluding checkpoint directories."""
    notebooks: list[Path] = []
    for d in SEARCH_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.ipynb")):
            if ".ipynb_checkpoints" not in p.parts:
                notebooks.append(p)
    return notebooks


def execute_notebook(nb_path: Path, timeout: int = 180) -> tuple[bool, float, str]:
    """Execute a single notebook headlessly with working directory set to notebook's dir."""
    t0 = time.perf_counter()
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        client = NotebookClient(nb, timeout=timeout, kernel_name="python3")
        client.execute(cwd=str(nb_path.parent))
        elapsed = time.perf_counter() - t0
        return True, elapsed, ""
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        err_msg = str(exc).encode("ascii", "replace").decode("ascii")
        return False, elapsed, err_msg


def main() -> int:
    notebooks = discover_notebooks()
    print("=" * 80)
    print(f"HEADLESS NOTEBOOK CI CHECK: {len(notebooks)} NOTEBOOK(S) DISCOVERED")
    print("=" * 80)

    if not notebooks:
        print("Error: No notebooks discovered!")
        return 1

    passed = 0
    failed = 0
    start_total = time.perf_counter()

    for nb_path in notebooks:
        rel_path = nb_path.relative_to(ROOT_DIR)
        print(f"Running: {rel_path} ...", end=" ", flush=True)

        success, elapsed, error_msg = execute_notebook(nb_path)
        if success:
            print(f"PASSED ({elapsed:.2f}s)")
            passed += 1
        else:
            print(f"FAILED ({elapsed:.2f}s)")
            print(f"   --> Error in {rel_path}:\n{error_msg}\n")
            failed += 1

    total_time = time.perf_counter() - start_total
    print("=" * 80)
    print(f"SUMMARY: {passed}/{len(notebooks)} notebooks passed in {total_time:.2f}s")
    if failed > 0:
        print(f"FAILURE: {failed} notebook(s) encountered execution errors.")
        print("=" * 80)
        return 1

    print("ALL NOTEBOOKS EXECUTED WITH 0 ERRORS.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
