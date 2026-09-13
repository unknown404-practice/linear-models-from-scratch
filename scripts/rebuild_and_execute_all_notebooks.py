"""
Rebuild and Execute All 19 Project Notebooks with Complete Visual Outputs.

This script:
1. Executes every notebook headlessly via nbclient.
2. Saves the executed notebooks with embedded outputs, plots, and stdout directly into .ipynb files.
3. Verifies 100% output population across tutorials, examples, mini-projects, masterclass, and competitions.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT_DIR = Path(__file__).resolve().parent.parent

NOTEBOOK_PATHS = [
    ROOT_DIR / "docs" / "examples" / "example_house_prices_baseline.ipynb",
    ROOT_DIR / "docs" / "examples" / "example_california_housing_pipeline.ipynb",
    ROOT_DIR / "docs" / "examples" / "example_robust_regression_with_ransac.ipynb",
    ROOT_DIR / "docs" / "examples" / "example_glm_logistic_poisson.ipynb",
    ROOT_DIR / "docs" / "examples" / "example_bayesian_linear_regression.ipynb",
    ROOT_DIR / "docs" / "examples" / "example_conformal_prediction_coverage.ipynb",
    ROOT_DIR / "docs" / "examples" / "example_streaming_rls_kalman.ipynb",
    ROOT_DIR / "docs" / "examples" / "example_quantile_regression_risk.ipynb",
    ROOT_DIR / "docs" / "tutorials" / "tutorial_01_linear_regression_basics.ipynb",
    ROOT_DIR / "docs" / "tutorials" / "tutorial_02_regularization_and_bias_variance.ipynb",
    ROOT_DIR / "docs" / "tutorials" / "tutorial_03_optimizers_and_schedulers.ipynb",
    ROOT_DIR / "docs" / "tutorials" / "tutorial_04_uncertainty_and_conformal_prediction.ipynb",
    ROOT_DIR / "docs" / "tutorials" / "tutorial_05_streaming_and_quantile_regression.ipynb",
    ROOT_DIR / "examples" / "02_model_comparison_dashboard.ipynb",
    ROOT_DIR / "examples" / "03_uncertainty_aware_predictions.ipynb",
    ROOT_DIR / "competitions" / "house_prices_competition_baseline.ipynb",
    ROOT_DIR / "notebooks" / "01_linear_regression_from_scratch_real_data.ipynb",
    ROOT_DIR / "notebooks" / "02_optimizers_and_geometry.ipynb",
    ROOT_DIR / "notebooks" / "03_uncertainty_and_risk.ipynb",
]


def execute_and_save_all():
    """Execute all 19 notebooks and save their outputs in place."""
    print("=" * 80)
    print(f"EXECUTING AND SAVING RICH OUTPUTS FOR {len(NOTEBOOK_PATHS)} NOTEBOOKS")
    print("=" * 80)

    start_total = time.perf_counter()
    passed = 0
    failed = 0

    for nb_path in NOTEBOOK_PATHS:
        rel_path = nb_path.relative_to(ROOT_DIR)
        print(f"Executing: {rel_path} ...", end=" ", flush=True)
        t0 = time.perf_counter()

        try:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)

            client = NotebookClient(nb, timeout=180, kernel_name="python3")
            client.execute(cwd=str(nb_path.parent))

            # Save executed notebook with all outputs
            with open(nb_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)

            elapsed = time.perf_counter() - t0
            code_cells = [c for c in nb.cells if c.cell_type == "code"]
            out_cells = [c for c in code_cells if len(c.get("outputs", [])) > 0]
            print(f"DONE ({elapsed:.1f}s, {len(out_cells)}/{len(code_cells)} cells with output)")
            passed += 1
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"FAILED ({elapsed:.1f}s)")
            print(f"  Error: {exc}")
            failed += 1

    total_time = time.perf_counter() - start_total
    print("=" * 80)
    print(
        f"RESULT: {passed}/{len(NOTEBOOK_PATHS)} notebooks successfully executed and saved in {total_time:.1f}s"
    )
    if failed > 0:
        print(f"Errors encountered in {failed} notebook(s)!")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(execute_and_save_all())
